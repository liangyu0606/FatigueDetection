"""
数据集加载器
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from typing import List, Dict, Tuple
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from config import *
from utils import augment_image

class YawnDataset(Dataset):
    """打哈欠检测数据集"""
    
    def __init__(self, samples: List[Dict], transform=None, augment=False):
        """
        Args:
            samples: 预处理后的样本列表
            transform: 数据变换
            augment: 是否进行数据增强
        """
        self.samples = samples
        self.transform = transform
        self.augment = augment
        
        # 统计类别分布
        self.labels = [sample['label'] for sample in samples]
        self.class_counts = np.bincount(self.labels)
        
        print(f"数据集大小: {len(samples)}")
        print(f"类别分布: {dict(enumerate(self.class_counts))}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        faces = sample['faces']  # (sequence_length, height, width, channels)
        landmarks = sample['landmarks']  # (sequence_length, 68, 2)
        label = sample['label']
        
        # 数据增强
        if self.augment:
            augmented_faces = []
            augmented_landmarks = []
            
            for i in range(len(faces)):
                face = faces[i]
                landmark = landmarks[i]
                
                # 应用增强
                aug_face, aug_landmark = augment_image(face, landmark, AUGMENTATION)
                augmented_faces.append(aug_face)
                augmented_landmarks.append(aug_landmark)
            
            faces = np.array(augmented_faces)
            landmarks = np.array(augmented_landmarks)
        
        # 转换为tensor
        faces = torch.from_numpy(faces).float()
        landmarks = torch.from_numpy(landmarks).float()
        label = torch.tensor(label, dtype=torch.long)
        
        # 调整维度顺序: (sequence_length, height, width, channels) -> (sequence_length, channels, height, width)
        faces = faces.permute(0, 3, 1, 2)
        
        # 展平特征点: (sequence_length, 68, 2) -> (sequence_length, 136)
        landmarks = landmarks.reshape(landmarks.size(0), -1)
        
        # 归一化像素值
        faces = faces / 255.0
        
        return faces, landmarks, label
    
    def get_class_weights(self):
        """计算类别权重用于处理不平衡数据"""
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(self.labels),
            y=self.labels
        )
        return torch.tensor(class_weights, dtype=torch.float32)

def load_processed_data(data_path: str) -> List[Dict]:
    """加载预处理后的数据"""
    with open(data_path, 'rb') as f:
        samples = pickle.load(f)
    return samples

def balance_dataset(samples: List[Dict], balance_ratio: float = 1.0) -> List[Dict]:
    """
    平衡数据集

    Args:
        samples: 原始样本
        balance_ratio: 正负样本比例 (positive/negative)

    Returns:
        balanced_samples: 平衡后的样本
    """
    positive_samples = [s for s in samples if s['label'] == 1]
    negative_samples = [s for s in samples if s['label'] == 0]

    print(f"原始数据 - 正样本: {len(positive_samples)}, 负样本: {len(negative_samples)}")

    if len(positive_samples) == 0:
        print("⚠️ 警告：没有正样本！")
        return samples

    if len(negative_samples) == 0:
        print("⚠️ 警告：没有负样本！")
        return samples

    # 调试模式下的特殊处理
    if DEBUG_MODE:
        # 如果正样本太少，大量上采样
        if len(negative_samples) / len(positive_samples) > 10:
            print("🔍 调试模式：正样本严重不足，进行大量上采样")
            target_positive = min(len(negative_samples) // 2, len(positive_samples) * 20)
            positive_samples = np.random.choice(
                positive_samples,
                size=target_positive,
                replace=True
            ).tolist()

            # 适当下采样负样本
            target_negative = len(positive_samples) * 2
            if target_negative < len(negative_samples):
                negative_samples = np.random.choice(
                    negative_samples,
                    size=target_negative,
                    replace=False
                ).tolist()
        else:
            # 正常平衡
            target_ratio = min(balance_ratio, 0.5)  # 调试模式下最多1:2
            if len(positive_samples) * target_ratio <= len(negative_samples):
                target_negative = int(len(positive_samples) / target_ratio)
                negative_samples = np.random.choice(
                    negative_samples,
                    size=target_negative,
                    replace=False
                ).tolist()
            else:
                target_positive = int(len(negative_samples) * target_ratio)
                positive_samples = np.random.choice(
                    positive_samples,
                    size=target_positive,
                    replace=True
                ).tolist()
    else:
        # 正式模式的平衡策略
        if len(positive_samples) * balance_ratio <= len(negative_samples):
            # 下采样负样本
            target_negative = int(len(positive_samples) / balance_ratio)
            negative_samples = np.random.choice(
                negative_samples,
                size=target_negative,
                replace=False
            ).tolist()
        else:
            # 上采样正样本
            target_positive = int(len(negative_samples) * balance_ratio)
            positive_samples = np.random.choice(
                positive_samples,
                size=target_positive,
                replace=True
            ).tolist()

    balanced_samples = positive_samples + negative_samples
    np.random.shuffle(balanced_samples)

    print(f"平衡后数据 - 正样本: {len(positive_samples)}, 负样本: {len(negative_samples)}")
    print(f"平衡比例: {len(positive_samples)/len(negative_samples):.3f}")

    return balanced_samples

def create_data_loaders(samples: List[Dict], 
                       train_ratio: float = TRAIN_SPLIT,
                       val_ratio: float = VAL_SPLIT,
                       test_ratio: float = TEST_SPLIT,
                       balance_train: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建数据加载器
    
    Args:
        samples: 预处理后的样本
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        balance_train: 是否平衡训练集
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # 确保比例和为1
    total_ratio = train_ratio + val_ratio + test_ratio
    train_ratio /= total_ratio
    val_ratio /= total_ratio
    test_ratio /= total_ratio
    
    # 分层划分数据集
    labels = [sample['label'] for sample in samples]
    
    # 首先分出训练集和临时集
    train_samples, temp_samples, train_labels, temp_labels = train_test_split(
        samples, labels,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=42
    )
    
    # 再从临时集中分出验证集和测试集
    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=temp_labels,
        random_state=42
    )
    
    print(f"数据集划分:")
    print(f"训练集: {len(train_samples)} 样本")
    print(f"验证集: {len(val_samples)} 样本")
    print(f"测试集: {len(test_samples)} 样本")
    
    # 平衡训练集
    if balance_train:
        train_samples = balance_dataset(train_samples)
    
    # 创建数据集
    train_dataset = YawnDataset(train_samples, augment=True)
    val_dataset = YawnDataset(val_samples, augment=False)
    test_dataset = YawnDataset(test_samples, augment=False)
    
    # 根据调试模式选择批次大小和工作进程数
    batch_size = DEBUG_BATCH_SIZE if DEBUG_MODE else BATCH_SIZE
    # Windows下使用单进程避免多进程问题
    num_workers = 0

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False  # CPU模式下关闭pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    return train_loader, val_loader, test_loader

def analyze_dataset(samples: List[Dict]):
    """分析数据集统计信息"""
    print("=== 数据集分析 ===")
    
    # 基本统计
    total_samples = len(samples)
    positive_samples = sum(1 for s in samples if s['label'] == 1)
    negative_samples = total_samples - positive_samples
    
    print(f"总样本数: {total_samples}")
    print(f"正样本数 (打哈欠): {positive_samples} ({positive_samples/total_samples*100:.1f}%)")
    print(f"负样本数 (正常/说话): {negative_samples} ({negative_samples/total_samples*100:.1f}%)")
    print(f"正负样本比例: {positive_samples/negative_samples:.3f}")
    
    # 视频来源统计
    video_sources = {}
    for sample in samples:
        video_path = sample['video_path']
        if 'Dash' in video_path:
            source = 'Dash'
        elif 'Mirror' in video_path:
            source = 'Mirror'
        else:
            source = 'Unknown'
        
        if 'Female' in video_path:
            gender = 'Female'
        elif 'Male' in video_path:
            gender = 'Male'
        else:
            gender = 'Unknown'
        
        key = f"{source}_{gender}"
        if key not in video_sources:
            video_sources[key] = {'total': 0, 'positive': 0}
        
        video_sources[key]['total'] += 1
        if sample['label'] == 1:
            video_sources[key]['positive'] += 1
    
    print("\n=== 按来源统计 ===")
    for source, stats in video_sources.items():
        pos_ratio = stats['positive'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"{source}: {stats['total']} 样本, {stats['positive']} 正样本 ({pos_ratio:.1f}%)")

if __name__ == "__main__":
    # 测试数据加载器
    data_path = os.path.join(PROCESSED_DATA_PATH, "processed_samples.pkl")
    
    if os.path.exists(data_path):
        samples = load_processed_data(data_path)
        analyze_dataset(samples)
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = create_data_loaders(samples)
        
        # 测试数据加载
        for faces, landmarks, labels in train_loader:
            print(f"批次形状 - 人脸: {faces.shape}, 特征点: {landmarks.shape}, 标签: {labels.shape}")
            break
    else:
        print(f"预处理数据文件不存在: {data_path}")
        print("请先运行 data_preprocessing.py 进行数据预处理")
