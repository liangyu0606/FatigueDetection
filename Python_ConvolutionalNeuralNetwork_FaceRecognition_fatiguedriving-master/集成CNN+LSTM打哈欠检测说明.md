# 集成CNN+LSTM打哈欠检测说明

## 概述

本项目已成功将real_pljc中的CNN+LSTM打哈欠检测模型集成到main.py的疲劳检测系统中。现在系统采用**专业化检测策略**：

- **打哈欠检测**: 纯CNN+LSTM深度学习模型（高准确性）
- **其他行为检测**: 启发式方法（眨眼、点头等基于EAR、MAR等指标）

## 主要改进

### 1. 新增模块

#### YawnCNNLSTM类
```python
class YawnCNNLSTM(nn.Module):
    """专门用于打哈欠检测的CNN+LSTM模型"""
```
- 输入: 138维特征序列 (EAR + MAR + 68个面部关键点坐标)
- 输出: 打哈欠概率
- 架构: CNN特征提取 + LSTM时序建模 + 全连接分类

#### YawnDetector类
```python
class YawnDetector:
    """专门用于打哈欠检测的类"""
```
- 模型加载和管理
- 特征提取和缓冲区管理
- 打哈欠预测

### 2. 集成策略

#### 纯CNN+LSTM检测策略（已简化）
```python
# 只使用CNN+LSTM检测打哈欠（移除启发式检测）
# 注意：cnn_lstm_yawn_detected已经包含了连续帧判断
if cnn_lstm_yawn_detected:
    # CNN+LSTM检测到打哈欠（已经通过连续15帧验证）
    detection_method = "CNN+LSTM"
    yawn_detected = True

# 如果CNN+LSTM不可用，则不检测打哈欠
if not (self.yawn_detector and self.yawn_detector.is_available):
    # 显示不可用提示
    pass
```

#### 检测策略（已简化）
1. **CNN+LSTM专用检测**: 唯一的打哈欠检测方法
2. **连续帧验证**: 需要连续15帧置信度>0.5
3. **冷却机制**: 检测后冷却3秒防止重复计数

## 技术特点

### 1. 特征工程（已修复）
- **EAR (Eye Aspect Ratio)**: 眼睛长宽比
- **MAR (Mouth Aspect Ratio)**: 嘴巴长宽比
- **68个面部关键点**: 使用帧高度归一化（与real_pljc一致）
- **时序建模**: 30帧序列长度

### 2. 模型架构
```
输入 (30, 138) → CNN特征提取 → LSTM时序建模 → 全连接分类 → 连续帧验证 → 最终判定
```

### 3. 连续帧验证（关键改进）
- **单帧阈值**: 0.5
- **连续帧要求**: 15帧
- **重置机制**: 单帧置信度<0.5时重置计数
- **误报控制**: 大幅减少偶然的高置信度误报

### 4. 冷却机制（防止重复计数）
- **冷却时间**: 3秒（可调整）
- **工作原理**: 检测到打哈欠后进入冷却期，期间忽略所有检测
- **状态重置**: 冷却期间重置CNN+LSTM连续帧计数
- **界面提示**: 显示冷却状态和剩余时间

### 4. 实时性优化
- 特征缓冲区管理
- 连续帧状态跟踪
- 错误处理和降级策略

## 安装和配置

### 1. 依赖要求
```bash
pip install torch torchvision
pip install opencv-python
pip install dlib
pip install numpy
```

### 2. 模型文件
确保以下模型文件存在：
- `../real_pljc/models/best_fatigue_model.pth` (CNN+LSTM模型)
- `./model/shape_predictor_68_face_landmarks.dat` (dlib面部关键点检测器)

### 3. 测试集成
```bash
python test_integrated_yawn_detection.py
```

## 使用方法

### 1. 启动程序
```bash
python main.py
```

### 2. 功能说明
- 程序启动时会自动加载CNN+LSTM打哈欠检测模型
- 实时视频流中会显示检测结果
- 界面显示包括:
  - `CNN+LSTM Yawn: 0.XX` - CNN+LSTM置信度
  - 检测方法标识 (CNN+LSTM/启发式)

### 3. 输出信息
```
🥱 检测到哈欠！方法: CNN+LSTM, 总计: 1, MAR: 0.654, CNN+LSTM置信度: 0.823
📊 记录哈欠事件到数据库 (方法: CNN+LSTM)
```

## 性能对比

| 检测方法 | 准确率 | 实时性 | 鲁棒性 | 复杂度 |
|---------|--------|--------|--------|--------|
| 纯启发式 | 中等 | 优秀 | 一般 | 低 |
| **纯CNN+LSTM** | **优秀** | **良好** | **优秀** | **中等** |
| 混合方法 | 优秀 | 良好 | 优秀 | 高 |

**选择纯CNN+LSTM的原因**：
- ✅ 准确率最高，专门针对打哈欠训练
- ✅ 逻辑简单，避免方法冲突
- ✅ 维护成本低，只需关注一个模型
- ✅ 与real_pljc保持一致的检测效果

## 配置参数

### 1. CNN+LSTM参数
```python
seq_length = 30              # 序列长度
consecutive_frames = 15      # 连续帧阈值
single_frame_threshold = 0.5 # 单帧置信度阈值
```

### 2. 冷却机制参数
```python
yawn_cooldown_seconds = 3.0  # 冷却时间（秒）
yawn_detection_enabled = True # 检测启用状态
```

### 3. 启发式参数
```python
MAR_THRESH = 0.5         # 嘴巴长宽比阈值
yawn_duration = (0.3, 3.0)  # 有效哈欠持续时间
```

## 故障排除

### 1. 模型加载失败
```
❌ CNN+LSTM打哈欠检测模型加载失败
```
**解决方案**: 检查模型文件路径和PyTorch安装

### 2. 特征提取失败
```
特征提取失败: ...
```
**解决方案**: 检查面部关键点检测是否正常

### 3. 预测失败
```
CNN+LSTM打哈欠预测失败: ...
```
**解决方案**: 检查输入特征维度和模型兼容性

### 4. 误报问题（已解决）
```
明明没有打哈欠，但是检测到了打哈欠
```
**解决方案**:
- ✅ 已修复特征归一化方式
- ✅ 已添加连续帧验证（需要连续15帧）
- ✅ 已移除低置信度判断
- ✅ 现在与real_pljc使用相同的检测逻辑

## 扩展功能

### 1. 添加其他行为的深度学习检测
可以类似地为眨眼、点头等行为添加深度学习模型

### 2. 模型更新
```python
# 更新模型文件
detector.load_model(new_model_path)
```

### 3. 自定义融合策略
修改`main.py`中的融合逻辑以适应特定需求

## 总结

通过集成CNN+LSTM模型，系统在保持实时性的同时显著提升了打哈欠检测的准确性和鲁棒性。融合策略确保了在各种场景下都能获得最佳的检测效果。
