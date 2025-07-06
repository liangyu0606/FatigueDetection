"""
数据预处理模块 - 使用dlib进行人脸检测和特征点提取
"""
import os
import cv2
import dlib
import numpy as np
import pickle
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import logging

from config import *
from utils import (
    parse_annotation_line, extract_face_landmarks, normalize_landmarks,
    create_sliding_windows, interpolate_missing_landmarks, setup_logging
)

class FatigueDataPreprocessor:
    def __init__(self):
        self.logger = setup_logging()
        
        # 初始化dlib检测器和预测器
        self.detector = dlib.get_frontal_face_detector()
        if not os.path.exists(DLIB_PREDICTOR_PATH):
            raise FileNotFoundError(f"dlib预测器文件未找到: {DLIB_PREDICTOR_PATH}")
        self.predictor = dlib.shape_predictor(DLIB_PREDICTOR_PATH)
        
        self.logger.info("dlib检测器和预测器初始化完成")
    
    def load_annotations(self, annotation_file: str) -> List[Tuple[str, int, List[Tuple[int, int]]]]:
        """加载标注文件"""
        annotations = []
        
        if not os.path.exists(annotation_file):
            self.logger.warning(f"标注文件不存在: {annotation_file}")
            return annotations
        
        with open(annotation_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                filename, label, intervals = parse_annotation_line(line)
                if filename is not None:
                    annotations.append((filename, label, intervals))
        
        self.logger.info(f"加载标注文件: {annotation_file}, 共{len(annotations)}条记录")
        return annotations
    
    def process_video(self, video_path: str, annotations: List[Tuple[str, int, List[Tuple[int, int]]]]) -> Dict:
        """
        处理单个视频文件
        
        Args:
            video_path: 视频文件路径
            annotations: 该视频的标注信息
            
        Returns:
            processed_data: 处理后的数据字典
        """
        if not os.path.exists(video_path):
            self.logger.error(f"视频文件不存在: {video_path}")
            return None
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"无法打开视频文件: {video_path}")
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        self.logger.info(f"处理视频: {video_path}, 总帧数: {total_frames}, FPS: {fps}")
        
        # 提取所有帧的人脸和特征点
        faces = []
        landmarks = []
        frame_indices = []
        
        frame_idx = 0

        # 调试模式下限制处理的帧数
        if DEBUG_MODE:
            total_frames = min(total_frames, DEBUG_MAX_FRAMES)
            self.logger.info(f"调试模式：限制处理帧数为 {total_frames}")

        pbar = tqdm(total=total_frames, desc=f"提取特征")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 调试模式下的帧数限制
            if DEBUG_MODE and frame_idx >= DEBUG_MAX_FRAMES:
                break
            
            # 提取人脸和特征点
            face_img, face_landmarks = extract_face_landmarks(frame, self.detector, self.predictor)

            if face_img is not None and face_img.size > 0 and len(face_img.shape) == 3:
                try:
                    # 检查图像尺寸是否有效
                    if face_img.shape[0] > 0 and face_img.shape[1] > 0:
                        # Resize人脸图像
                        face_img_resized = cv2.resize(face_img, FACE_SIZE)
                        faces.append(face_img_resized)

                        # 归一化特征点
                        landmarks_norm = normalize_landmarks(face_landmarks, face_img.shape[:2])
                        landmarks.append(landmarks_norm)
                        frame_indices.append(frame_idx)
                    else:
                        # 图像尺寸无效
                        faces.append(None)
                        landmarks.append(None)
                        frame_indices.append(frame_idx)
                except Exception as e:
                    # resize失败，记录为None
                    self.logger.warning(f"帧 {frame_idx} resize失败: {e}")
                    faces.append(None)
                    landmarks.append(None)
                    frame_indices.append(frame_idx)
            else:
                faces.append(None)
                landmarks.append(None)
                frame_indices.append(frame_idx)
            
            frame_idx += 1
            pbar.update(1)
        
        cap.release()
        pbar.close()

        # 统计检测成功率
        valid_faces = sum(1 for face in faces if face is not None)
        detection_rate = valid_faces / len(faces) * 100 if faces else 0
        self.logger.info(f"人脸检测成功率: {detection_rate:.1f}% ({valid_faces}/{len(faces)})")

        if detection_rate < 50:
            self.logger.warning(f"人脸检测成功率较低，可能影响训练效果")

        # 插值缺失的特征点
        landmarks = interpolate_missing_landmarks(landmarks)

        # 处理缺失的人脸图像（使用前一帧或后一帧）
        faces = self._interpolate_missing_faces(faces)
        
        processed_data = {
            'video_path': video_path,
            'faces': faces,
            'landmarks': landmarks,
            'frame_indices': frame_indices,
            'total_frames': total_frames,
            'fps': fps,
            'annotations': annotations
        }
        
        return processed_data
    
    def _interpolate_missing_faces(self, faces: List[Optional[np.ndarray]]) -> List[np.ndarray]:
        """插值缺失的人脸图像"""
        interpolated_faces = []
        
        for i, face in enumerate(faces):
            if face is not None:
                interpolated_faces.append(face)
            else:
                # 寻找最近的有效人脸
                replacement_face = None
                
                # 向前搜索
                for j in range(i-1, -1, -1):
                    if faces[j] is not None:
                        replacement_face = faces[j]
                        break
                
                # 向后搜索
                if replacement_face is None:
                    for j in range(i+1, len(faces)):
                        if faces[j] is not None:
                            replacement_face = faces[j]
                            break
                
                # 如果还是没找到，创建黑色图像
                if replacement_face is None:
                    replacement_face = np.zeros((FACE_SIZE[1], FACE_SIZE[0], 3), dtype=np.uint8)
                
                interpolated_faces.append(replacement_face)
        
        return interpolated_faces
    
    def create_training_samples(self, processed_data: Dict) -> List[Dict]:
        """
        创建训练样本
        
        Args:
            processed_data: 处理后的视频数据
            
        Returns:
            samples: 训练样本列表
        """
        faces = processed_data['faces']
        landmarks = processed_data['landmarks']
        annotations = processed_data['annotations']
        total_frames = processed_data['total_frames']
        
        samples = []
        
        # 为每个标注创建样本
        for filename, label, yawn_intervals in annotations:
            # 创建帧标签数组
            frame_labels = np.zeros(total_frames, dtype=int)
            
            # 标记打哈欠帧
            if label == 2 and yawn_intervals:  # 打哈欠类别
                for start, end in yawn_intervals:
                    frame_labels[start:end+1] = 1
            
            # 创建滑动窗口
            windows = create_sliding_windows(total_frames, SEQUENCE_LENGTH, OVERLAP_RATIO)
            
            for start_idx, end_idx in windows:
                # 提取窗口内的数据
                window_faces = faces[start_idx:end_idx]
                window_landmarks = landmarks[start_idx:end_idx]
                window_labels = frame_labels[start_idx:end_idx]
                
                # 确保窗口长度一致
                if len(window_faces) < SEQUENCE_LENGTH:
                    # 填充到指定长度
                    pad_length = SEQUENCE_LENGTH - len(window_faces)
                    window_faces.extend([window_faces[-1]] * pad_length)
                    window_landmarks.extend([window_landmarks[-1]] * pad_length)
                    window_labels = np.pad(window_labels, (0, pad_length), mode='edge')
                
                # 确定窗口标签（调整策略）
                yawn_frames = np.sum(window_labels)
                if DEBUG_MODE:
                    # 调试模式：只要有打哈欠帧就标记为正样本
                    window_label = 1 if yawn_frames > 0 else 0
                else:
                    # 正式模式：多数投票
                    window_label = 1 if yawn_frames > SEQUENCE_LENGTH // 4 else 0  # 降低阈值
                
                sample = {
                    'faces': np.array(window_faces),
                    'landmarks': np.array(window_landmarks),
                    'label': window_label,
                    'video_path': processed_data['video_path'],
                    'start_frame': start_idx,
                    'end_frame': end_idx
                }
                
                samples.append(sample)
        
        return samples

    def process_dataset(self, dataset_path: str) -> List[Dict]:
        """
        处理整个数据集

        Args:
            dataset_path: 数据集路径

        Returns:
            all_samples: 所有训练样本
        """
        all_samples = []

        # 处理Dash和Mirror文件夹
        for folder_name in ['Dash', 'Mirror']:
            folder_path = os.path.join(dataset_path, folder_name)
            if not os.path.exists(folder_path):
                continue

            # 处理Female和Male子文件夹
            for gender_folder in os.listdir(folder_path):
                gender_path = os.path.join(folder_path, gender_folder)
                if not os.path.isdir(gender_path):
                    continue

                self.logger.info(f"处理文件夹: {gender_path}")

                # 加载标注文件
                annotation_file = os.path.join(gender_path, 'labels.txt')
                annotations = self.load_annotations(annotation_file)

                if not annotations:
                    continue

                # 按视频文件分组标注
                video_annotations = {}
                for filename, label, intervals in annotations:
                    if filename not in video_annotations:
                        video_annotations[filename] = []
                    video_annotations[filename].append((filename, label, intervals))

                # 处理每个视频
                video_count = 0
                for video_filename, video_annots in video_annotations.items():
                    # 调试模式下限制视频数量
                    if DEBUG_MODE and video_count >= DEBUG_MAX_VIDEOS:
                        self.logger.info(f"调试模式：已处理 {video_count} 个视频，跳过剩余视频")
                        break

                    video_path = os.path.join(gender_path, video_filename)

                    # 处理视频
                    processed_data = self.process_video(video_path, video_annots)
                    if processed_data is None:
                        continue

                    # 创建训练样本
                    samples = self.create_training_samples(processed_data)
                    all_samples.extend(samples)

                    self.logger.info(f"视频 {video_filename} 生成 {len(samples)} 个样本")
                    video_count += 1

        return all_samples

    def save_processed_data(self, samples: List[Dict], output_path: str):
        """保存处理后的数据"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(samples, f)

        self.logger.info(f"处理后的数据已保存到: {output_path}")

        # 统计信息
        positive_samples = sum(1 for sample in samples if sample['label'] == 1)
        negative_samples = len(samples) - positive_samples

        self.logger.info(f"总样本数: {len(samples)}")
        self.logger.info(f"正样本数 (打哈欠): {positive_samples}")
        self.logger.info(f"负样本数 (正常/说话): {negative_samples}")
        self.logger.info(f"正负样本比例: {positive_samples/negative_samples:.2f}")

if __name__ == "__main__":
    preprocessor = FatigueDataPreprocessor()

    # 处理数据集
    samples = preprocessor.process_dataset(DATASET_ROOT)

    # 保存处理后的数据
    output_file = os.path.join(PROCESSED_DATA_PATH, "processed_samples.pkl")
    preprocessor.save_processed_data(samples, output_file)
