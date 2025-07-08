"""
CNN+LSTM模型架构 - 打哈欠检测
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from config import *

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 跳跃连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += residual  # 残差连接
        out = F.relu(out)

        return out

class CNNFeatureExtractor(nn.Module):
    """基于残差网络的CNN特征提取器"""

    def __init__(self, input_channels: int = 3):
        super(CNNFeatureExtractor, self).__init__()

        # 初始卷积层
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 残差层
        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 特征映射层
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE)
        )

        # 权重初始化
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)

        x = self.fc(x)

        return x

class LandmarkProcessor(nn.Module):
    """改进的特征点处理器 - 使用残差连接"""

    def __init__(self, landmark_dim: int = 68 * 2):
        super(LandmarkProcessor, self).__init__()

        # 输入投影
        self.input_proj = nn.Linear(landmark_dim, 256)

        # 残差块
        self.res_block1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256)
        )

        self.res_block2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128)
        )

        # 输出层
        self.output_proj = nn.Linear(128, 64)

        # 跳跃连接的投影层
        self.skip_proj1 = nn.Identity()  # 256 -> 256
        self.skip_proj2 = nn.Linear(256, 128)  # 256 -> 128

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: (batch_size, 68*2)
        x = F.relu(self.input_proj(x))  # (batch_size, 256)

        # 第一个残差块
        residual1 = self.skip_proj1(x)
        out1 = self.res_block1(x)
        x = F.relu(out1 + residual1)

        # 第二个残差块
        residual2 = self.skip_proj2(x)
        out2 = self.res_block2(x)
        x = F.relu(out2 + residual2)

        # 输出投影
        x = F.relu(self.output_proj(x))

        return x

class YawnDetectionModel(nn.Module):
    """改进的打哈欠检测模型 - 残差CNN+LSTM架构"""

    def __init__(self):
        super(YawnDetectionModel, self).__init__()

        # CNN特征提取器
        self.cnn_extractor = CNNFeatureExtractor()

        # 特征点处理器
        self.landmark_processor = LandmarkProcessor()

        # 改进的特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(512 + 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )

        # 多层LSTM
        self.lstm1 = nn.LSTM(
            input_size=256,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.lstm2 = nn.LSTM(
            input_size=LSTM_HIDDEN_SIZE * 2,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # LSTM层归一化
        self.ln1 = nn.LayerNorm(LSTM_HIDDEN_SIZE * 2)
        self.ln2 = nn.LayerNorm(LSTM_HIDDEN_SIZE * 2)

        # 改进的注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=LSTM_HIDDEN_SIZE * 2,
            num_heads=8,
            dropout=DROPOUT_RATE,
            batch_first=True
        )

        # 残差分类器
        self.classifier = nn.Sequential(
            nn.Linear(LSTM_HIDDEN_SIZE * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(128, NUM_CLASSES)
        )

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
        
    def forward(self, faces, landmarks):
        """
        改进的前向传播

        Args:
            faces: 人脸图像序列 (batch_size, sequence_length, channels, height, width)
            landmarks: 特征点序列 (batch_size, sequence_length, 68*2)

        Returns:
            output: 分类结果 (batch_size, num_classes)
        """
        batch_size, seq_len = faces.size(0), faces.size(1)

        # 重塑输入以便CNN处理
        faces = faces.reshape(batch_size * seq_len, *faces.shape[2:])
        landmarks = landmarks.reshape(batch_size * seq_len, -1)

        # CNN特征提取
        cnn_features = self.cnn_extractor(faces)  # (batch_size * seq_len, 512)

        # 特征点处理
        landmark_features = self.landmark_processor(landmarks)  # (batch_size * seq_len, 64)

        # 特征融合
        fused_features = torch.cat([cnn_features, landmark_features], dim=1)
        fused_features = self.feature_fusion(fused_features)

        # 重塑为序列格式
        fused_features = fused_features.reshape(batch_size, seq_len, -1)

        # 多层LSTM处理
        lstm_out1, _ = self.lstm1(fused_features)
        lstm_out1 = self.ln1(lstm_out1)  # 层归一化

        # 残差连接（如果维度匹配）
        if lstm_out1.size(-1) == fused_features.size(-1):
            lstm_out1 = lstm_out1 + fused_features

        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.ln2(lstm_out2)  # 层归一化

        # 残差连接
        lstm_out2 = lstm_out2 + lstm_out1

        # 注意力机制
        attn_out, attn_weights = self.attention(lstm_out2, lstm_out2, lstm_out2)

        # 多种池化策略结合
        # 1. 全局平均池化
        avg_pooled = torch.mean(attn_out, dim=1)
        # 2. 全局最大池化
        max_pooled, _ = torch.max(attn_out, dim=1)
        # 3. 最后时间步
        last_step = attn_out[:, -1, :]

        # 组合不同的池化结果
        pooled_features = (avg_pooled + max_pooled + last_step) / 3

        # 分类
        output = self.classifier(pooled_features)

        return output
    
    def get_attention_weights(self, faces, landmarks):
        """获取注意力权重用于可视化"""
        with torch.no_grad():
            batch_size, seq_len = faces.size(0), faces.size(1)
            
            # 重塑输入
            faces = faces.reshape(batch_size * seq_len, *faces.shape[2:])
            landmarks = landmarks.reshape(batch_size * seq_len, -1)
            
            # 特征提取和融合
            cnn_features = self.cnn_extractor(faces)
            landmark_features = self.landmark_processor(landmarks)
            fused_features = torch.cat([cnn_features, landmark_features], dim=1)
            fused_features = F.relu(self.feature_fusion(fused_features))
            fused_features = fused_features.reshape(batch_size, seq_len, -1)
            
            # LSTM处理
            lstm_out, _ = self.lstm(fused_features)
            
            # 注意力权重
            _, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
            
            return attn_weights

class ImprovedFocalLoss(nn.Module):
    """改进的Focal Loss - 处理类别不平衡和标签平滑"""

    def __init__(self, alpha=None, gamma=2, label_smoothing=0.1, reduction='mean'):
        super(ImprovedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 标签平滑
        num_classes = inputs.size(-1)
        if self.label_smoothing > 0:
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

            # 使用KL散度计算平滑损失
            log_probs = F.log_softmax(inputs, dim=-1)
            loss = -torch.sum(smooth_targets * log_probs, dim=-1)
        else:
            loss = F.cross_entropy(inputs, targets, reduction='none')

        # Focal权重
        probs = F.softmax(inputs, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma

        # 类别权重
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
            focal_weight = alpha_t * focal_weight

        focal_loss = focal_weight * loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑交叉熵损失"""

    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=-1)
        num_classes = inputs.size(-1)

        # 创建平滑标签
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / (num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = -torch.sum(smooth_targets * log_probs, dim=-1)
        return loss.mean()

def create_model():
    """创建模型实例"""
    model = YawnDetectionModel()
    return model

def count_parameters(model):
    """统计模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # 测试模型
    model = create_model()
    
    # 创建测试输入
    batch_size = 2
    faces = torch.randn(batch_size, SEQUENCE_LENGTH, 3, FACE_SIZE[0], FACE_SIZE[1])
    landmarks = torch.randn(batch_size, SEQUENCE_LENGTH, 68 * 2)
    
    # 前向传播
    output = model(faces, landmarks)
    
    print(f"模型参数数量: {count_parameters(model):,}")
    print(f"输入形状 - 人脸: {faces.shape}, 特征点: {landmarks.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出: {output}")
