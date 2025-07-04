"""
工具函数
"""
import cv2
import numpy as np
import dlib
from typing import List, Tuple, Optional
import logging

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def parse_annotation_line(line: str) -> Tuple[str, int, List[Tuple[int, int]]]:
    """
    解析标注文件的一行
    
    Args:
        line: 标注行，格式：filename label intervals
        
    Returns:
        filename: 视频文件名
        label: 类别标签
        intervals: 打哈欠区间列表 [(start, end), ...]
    """
    parts = line.strip().split()
    if len(parts) < 3:
        return None, None, None
        
    filename = parts[0]
    label = int(parts[1])
    interval_str = parts[2]
    
    intervals = []
    if interval_str != "-1,-1":
        # 解析多个区间：774-965,1623-1806,2387-2631
        for interval in interval_str.split(','):
            if '-' in interval:
                start, end = map(int, interval.split('-'))
                intervals.append((start, end))
    
    return filename, label, intervals

def extract_face_landmarks(image: np.ndarray, detector, predictor) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    提取人脸区域和68个特征点

    Args:
        image: 输入图像
        detector: dlib人脸检测器
        predictor: dlib特征点预测器

    Returns:
        face_image: 人脸区域图像
        landmarks: 68个特征点坐标 (68, 2)
    """
    try:
        # 检查输入图像
        if image is None or image.size == 0:
            return None, None

        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 人脸检测
        faces = detector(gray)

        if len(faces) == 0:
            return None, None

        # 取第一个检测到的人脸
        face = faces[0]

        # 提取人脸区域坐标
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # 边界检查
        img_h, img_w = image.shape[:2]
        x = max(0, x)
        y = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)

        # 检查区域是否有效
        if x2 <= x or y2 <= y:
            return None, None

        # 提取人脸区域
        face_image = image[y:y2, x:x2]

        # 检查提取的人脸图像
        if face_image.size == 0 or face_image.shape[0] == 0 or face_image.shape[1] == 0:
            return None, None

        # 提取特征点
        landmarks = predictor(gray, face)
        landmarks_array = np.array([[p.x - x, p.y - y] for p in landmarks.parts()])

        return face_image, landmarks_array

    except Exception as e:
        # 任何异常都返回None
        return None, None

def normalize_landmarks(landmarks: np.ndarray, face_size: Tuple[int, int]) -> np.ndarray:
    """
    归一化特征点坐标
    
    Args:
        landmarks: 原始特征点坐标
        face_size: 人脸图像尺寸
        
    Returns:
        normalized_landmarks: 归一化后的特征点坐标
    """
    if landmarks is None:
        return None
    
    # 归一化到[0, 1]范围
    landmarks_norm = landmarks.copy().astype(np.float32)
    landmarks_norm[:, 0] /= face_size[0]
    landmarks_norm[:, 1] /= face_size[1]
    
    return landmarks_norm

def augment_image(image: np.ndarray, landmarks: np.ndarray, augmentation_params: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    数据增强
    
    Args:
        image: 输入图像
        landmarks: 特征点
        augmentation_params: 增强参数
        
    Returns:
        augmented_image: 增强后的图像
        augmented_landmarks: 增强后的特征点
    """
    aug_image = image.copy()
    aug_landmarks = landmarks.copy() if landmarks is not None else None
    
    # 亮度调整
    if 'brightness_range' in augmentation_params:
        brightness = np.random.uniform(-augmentation_params['brightness_range'], 
                                     augmentation_params['brightness_range'])
        aug_image = cv2.convertScaleAbs(aug_image, alpha=1, beta=brightness * 255)
    
    # 对比度调整
    if 'contrast_range' in augmentation_params:
        contrast = np.random.uniform(1 - augmentation_params['contrast_range'],
                                   1 + augmentation_params['contrast_range'])
        aug_image = cv2.convertScaleAbs(aug_image, alpha=contrast, beta=0)
    
    # 水平翻转
    if augmentation_params.get('horizontal_flip', False) and np.random.random() > 0.5:
        aug_image = cv2.flip(aug_image, 1)
        if aug_landmarks is not None:
            aug_landmarks[:, 0] = 1.0 - aug_landmarks[:, 0]  # 假设landmarks已归一化
    
    return aug_image, aug_landmarks

def create_sliding_windows(total_frames: int, sequence_length: int, overlap_ratio: float) -> List[Tuple[int, int]]:
    """
    创建滑动窗口
    
    Args:
        total_frames: 总帧数
        sequence_length: 序列长度
        overlap_ratio: 重叠比例
        
    Returns:
        windows: 窗口列表 [(start, end), ...]
    """
    if total_frames < sequence_length:
        return [(0, total_frames)]
    
    step = int(sequence_length * (1 - overlap_ratio))
    windows = []
    
    start = 0
    while start + sequence_length <= total_frames:
        windows.append((start, start + sequence_length))
        start += step
    
    # 确保最后一个窗口包含到最后一帧
    if windows[-1][1] < total_frames:
        windows.append((total_frames - sequence_length, total_frames))
    
    return windows

def interpolate_missing_landmarks(landmarks_sequence: List[Optional[np.ndarray]]) -> List[np.ndarray]:
    """
    插值缺失的特征点
    
    Args:
        landmarks_sequence: 特征点序列，可能包含None
        
    Returns:
        interpolated_sequence: 插值后的特征点序列
    """
    # 找到有效的特征点
    valid_indices = [i for i, lm in enumerate(landmarks_sequence) if lm is not None]
    
    if len(valid_indices) == 0:
        return [np.zeros((68, 2)) for _ in landmarks_sequence]
    
    interpolated = []
    for i, landmarks in enumerate(landmarks_sequence):
        if landmarks is not None:
            interpolated.append(landmarks)
        else:
            # 线性插值
            if len(valid_indices) == 1:
                interpolated.append(landmarks_sequence[valid_indices[0]])
            else:
                # 找到最近的两个有效点进行插值
                left_idx = max([idx for idx in valid_indices if idx < i], default=valid_indices[0])
                right_idx = min([idx for idx in valid_indices if idx > i], default=valid_indices[-1])
                
                if left_idx == right_idx:
                    interpolated.append(landmarks_sequence[left_idx])
                else:
                    # 线性插值
                    alpha = (i - left_idx) / (right_idx - left_idx)
                    left_lm = landmarks_sequence[left_idx]
                    right_lm = landmarks_sequence[right_idx]
                    interp_lm = (1 - alpha) * left_lm + alpha * right_lm
                    interpolated.append(interp_lm)
    
    return interpolated
