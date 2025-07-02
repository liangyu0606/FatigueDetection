import os
import cv2
import dlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader


class YawDDPreprocessor:
    def __init__(self, predictor_path, seq_length=30):
        # 初始化dlib工具

        # dlib中自带的人脸检测器
        self.detector = dlib.get_frontal_face_detector()
        # 根据predictor_path指定的路径读取用于检测68个特征点的模型
        self.predictor = dlib.shape_predictor(predictor_path)
        # 每个训练样本包含的连续帧数，默认30帧
        self.seq_length = seq_length

        # 定义面部关键点索引
        # 点36: 左眼外眼角
        # 点37: 左眼上眼睑外侧
        # 点38: 左眼上眼睑内侧
        # 点39: 左眼内眼角
        # 点40: 左眼下眼睑内侧
        # 点41: 左眼下眼睑外侧
        self.left_eye_indices = list(range(36, 42)) # [36, 37, 38, 39, 40, 41]
        self.right_eye_indices = list(range(42, 48))
        # 外轮廓(48 - 59): 嘴唇外边缘的12个点
        # 内轮廓(60 - 67): 嘴唇内边缘的8个点
        self.mouth_indices = list(range(48, 68))

    def calculate_eye_aspect_ratio(self, landmarks):
        """计算眼睛纵横比(EAR)"""
        left_eye = landmarks[self.left_eye_indices]
        right_eye = landmarks[self.right_eye_indices]
        A = np.linalg.norm(left_eye[1] - left_eye[5])
        B = np.linalg.norm(left_eye[2] - left_eye[4])
        C = np.linalg.norm(left_eye[0] - left_eye[3])
        left_ear = (A + B) / (2.0 * C)
        A = np.linalg.norm(right_eye[1] - right_eye[5])
        B = np.linalg.norm(right_eye[2] - right_eye[4])
        C = np.linalg.norm(right_eye[0] - right_eye[3])
        right_ear = (A + B) / (2.0 * C)
        return (left_ear + right_ear) / 2.0

    def calculate_mouth_aspect_ratio(self, landmarks):
        """计算嘴巴纵横比(MAR)"""
        mouth = landmarks[self.mouth_indices]
        A = np.linalg.norm(mouth[2] - mouth[10])
        B = np.linalg.norm(mouth[4] - mouth[8])
        C = np.linalg.norm(mouth[0] - mouth[6])
        return (A + B) / (2.0 * C)

    def extract_landmarks(self, frame):
        """提取面部68个关键点"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        if not faces:
            return None
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        shape = self.predictor(gray, face)
        return np.array([[part.x, part.y] for part in shape.parts()])

    def extract_features_from_video(self, video_path, yawn_ranges):
        """从视频中提取特征序列，根据打哈欠范围进行处理"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Cannot open video {video_path}")
            cap.release()
            return np.array([])

        features = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 如果有打哈欠范围，处理指定范围
        if yawn_ranges:
            for start, end in yawn_ranges:
                start = max(0, start)
                end = min(total_frames - 1, end)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                sub_features = []
                current_frame = start
                while current_frame <= end and len(sub_features) < self.seq_length:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    landmarks = self.extract_landmarks(frame)
                    if landmarks is None:
                        current_frame += 1
                        continue
                    ear = self.calculate_eye_aspect_ratio(landmarks)
                    mar = self.calculate_mouth_aspect_ratio(landmarks)
                    nose = landmarks[30]
                    normalized_landmarks = (landmarks - nose).flatten() / frame.shape[0]
                    sub_features.append(np.concatenate([[ear, mar], normalized_landmarks]))
                    current_frame += 1

                # 如果提取到了特征，进行填充
                if len(sub_features) > 0:
                    if len(sub_features) < self.seq_length:
                        feature_dim = sub_features[0].shape[0]
                        padding = np.zeros((self.seq_length - len(sub_features), feature_dim))
                        sub_features = np.vstack([sub_features, padding])
                    features.append(sub_features[:self.seq_length])
        else:
            # 如果没有打哈欠范围，处理整个视频的前seq_length帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_features = []
            frame_count = 0
            while frame_count < self.seq_length:
                ret, frame = cap.read()
                if not ret:
                    break
                landmarks = self.extract_landmarks(frame)
                if landmarks is None:
                    frame_count += 1
                    continue
                ear = self.calculate_eye_aspect_ratio(landmarks)
                mar = self.calculate_mouth_aspect_ratio(landmarks)
                nose = landmarks[30]
                normalized_landmarks = (landmarks - nose).flatten() / frame.shape[0]
                frame_features.append(np.concatenate([[ear, mar], normalized_landmarks]))
                frame_count += 1

            # 如果提取到了特征，进行填充
            if len(frame_features) > 0:
                if len(frame_features) < self.seq_length:
                    feature_dim = frame_features[0].shape[0]
                    padding = np.zeros((self.seq_length - len(frame_features), feature_dim))
                    frame_features = np.vstack([frame_features, padding])
                features.append(frame_features[:self.seq_length])

        cap.release()
        return np.array(features) if features else np.array([])

    def process_dataset(self, data_dir):
        """处理整个数据集，生成特征和标签"""
        all_sequences = []
        all_labels = []

        # 处理所有目录以获得更多数据
        directories = [
            os.path.join(data_dir, "Mirror", "Male_mirror"),
            os.path.join(data_dir, "Mirror", "Female_mirror"),
            os.path.join(data_dir, "Dash", "Male_dash"),
            os.path.join(data_dir, "Dash", "Female_dash")
        ]

        for gender_path in directories:
            if not os.path.exists(gender_path):
                print(f"Directory not found: {gender_path}")
                continue

            label_file = os.path.join(gender_path, "labels.txt")
            dir_name = os.path.basename(gender_path)

            if os.path.exists(label_file):
                label_dict = {}
                with open(label_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:  # 跳过空行
                            continue
                        parts = line.split()
                        if len(parts) < 3:  # 确保有足够的部分
                            print(f"Warning: Skipping malformed line: {line}")
                            continue
                        video_name = parts[0]
                        label_type = int(parts[1])
                        yawn_str = parts[2]
                        yawn_ranges = []
                        if yawn_str != "-1,-1":
                            yawn_pairs = yawn_str.split(',')
                            for pair in yawn_pairs:
                                if '-' in pair:
                                    start, end = map(int, pair.split('-'))
                                    yawn_ranges.append((start, end))
                        label_dict[video_name] = (label_type, yawn_ranges)

                for video_file in tqdm(os.listdir(gender_path), desc=f"Processing {dir_name}"):
                    if not video_file.endswith(('.avi', '.mp4')):
                        continue
                    video_path = os.path.join(gender_path, video_file)
                    try:
                        if video_file in label_dict:
                            label_type, yawn_ranges = label_dict[video_file]
                            features = self.extract_features_from_video(video_path, yawn_ranges)

                            # 处理所有类型：0=正常，1=交谈，2=打哈欠
                            if label_type in [0, 1, 2] and len(features) > 0:
                                for sub_feature in features:
                                    all_sequences.append(sub_feature)
                                    # 将类型2（打哈欠）标记为疲劳(1)，其他标记为正常(0)
                                    all_labels.append(1 if label_type == 2 else 0)
                    except Exception as e:
                        print(f"Error processing {video_file}: {e}")
            else:
                print(f"Label file not found: {label_file}")
                # 如果没有标签文件，根据文件名推断
                for video_file in tqdm(os.listdir(gender_path), desc=f"Processing {dir_name} (no labels)"):
                    if not video_file.endswith(('.avi', '.mp4')):
                        continue
                    video_path = os.path.join(gender_path, video_file)
                    try:
                        # 根据文件名推断标签
                        if "Yawning" in video_file:
                            label = 1  # 疲劳
                            yawn_ranges = []  # 处理整个视频
                        else:
                            label = 0  # 正常
                            yawn_ranges = []

                        features = self.extract_features_from_video(video_path, yawn_ranges)
                        if len(features) > 0:
                            for sub_feature in features:
                                all_sequences.append(sub_feature)
                                all_labels.append(label)
                    except Exception as e:
                        print(f"Error processing {video_file}: {e}")

        print(f"Total sequences before augmentation: {len(all_sequences)}")
        print(f"Label distribution: {np.bincount(all_labels)}")

        # 数据增强
        if len(all_sequences) > 0:
            all_sequences, all_labels = self.augment_data(all_sequences, all_labels)

        return np.array(all_sequences), np.array(all_labels)

    def augment_data(self, sequences, labels):
        """数据增强：添加噪声、时间偏移等"""
        augmented_sequences = list(sequences)
        augmented_labels = list(labels)

        print("Applying data augmentation...")

        for i, (seq, label) in enumerate(zip(sequences, labels)):
            # 1. 添加高斯噪声
            noise_seq = seq + np.random.normal(0, 0.01, seq.shape)
            augmented_sequences.append(noise_seq)
            augmented_labels.append(label)

            # 2. 特征缩放（轻微变化）
            scale_factor = np.random.uniform(0.95, 1.05)
            scaled_seq = seq * scale_factor
            augmented_sequences.append(scaled_seq)
            augmented_labels.append(label)

            # 3. 时间偏移（如果序列长度允许）
            if seq.shape[0] > 5:
                shift = np.random.randint(-2, 3)
                if shift > 0:
                    shifted_seq = np.vstack([seq[shift:], seq[-shift:]])
                elif shift < 0:
                    shifted_seq = np.vstack([seq[:shift], seq[:-shift]])
                else:
                    shifted_seq = seq.copy()
                augmented_sequences.append(shifted_seq)
                augmented_labels.append(label)

        print(f"Data augmented: {len(sequences)} -> {len(augmented_sequences)} samples")
        return augmented_sequences, augmented_labels

    def split_and_scale_data(self, X, y, test_size=0.2, val_size=0.1):
        """分割数据集并标准化特征"""
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size / (1 - test_size),
            random_state=42, stratify=y_train
        )
        # 特征标准化
        scaler = StandardScaler()
        n_samples, n_timesteps, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(n_samples * n_timesteps, n_features)
        X_train_scaled = scaler.fit_transform(X_train_reshaped)
        X_train = X_train_scaled.reshape(n_samples, n_timesteps, n_features)
        X_val_reshaped = X_val.reshape(X_val.shape[0] * n_timesteps, n_features)
        X_val_scaled = scaler.transform(X_val_reshaped)
        X_val = X_val_scaled.reshape(X_val.shape[0], n_timesteps, n_features)
        X_test_reshaped = X_test.reshape(X_test.shape[0] * n_timesteps, n_features)
        X_test_scaled = scaler.transform(X_test_reshaped)
        X_test = X_test_scaled.reshape(X_test.shape[0], n_timesteps, n_features)
        return X_train, X_val, X_test, y_train, y_val, y_test