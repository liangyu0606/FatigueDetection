# 断点训练功能说明

## 🔄 断点训练功能

疲劳检测模型现在支持断点训练功能，可以从上次中断的地方继续训练，避免因为意外中断而丢失训练进度。

## 📋 功能特性

### ✅ 自动保存检查点
- **最新模型**: `latest_model.pth` - 每轮训练后自动保存
- **最佳模型**: `best_model.pth` - 验证准确率最高时保存
- **定期备份**: `checkpoint_epoch_X.pth` - 每10轮保存一次备份

### ✅ 完整状态保存
检查点文件包含以下信息：
- 模型参数状态
- 优化器状态
- 学习率调度器状态
- 训练轮次
- 训练历史记录（损失、准确率）
- 训练配置参数
- 保存时间戳

### ✅ 智能恢复
- 自动检测现有检查点
- 从正确的轮次继续训练
- 保持训练历史连续性
- 重新配置学习率调度器

## 🚀 使用方法

### 方法1：使用主脚本（推荐）
```bash
python main.py --mode train
```
- 自动检测是否存在检查点
- 提示用户选择继续训练或开始新训练

### 方法2：使用专用断点训练脚本
```bash
python checkpoint_training.py
```

#### 查看可用检查点
```bash
python checkpoint_training.py --list
```

#### 查看检查点详细信息
```bash
python checkpoint_training.py --info latest_model.pth
```

#### 强制开始新训练
```bash
python checkpoint_training.py --force
```

### 方法3：直接调用训练脚本
```bash
# 继续训练
python train.py --resume

# 开始新训练
python train.py
```

## 📁 文件结构

```
models/
├── latest_model.pth          # 最新检查点
├── best_model.pth            # 最佳模型
├── checkpoint_epoch_10.pth   # 第10轮备份
├── checkpoint_epoch_20.pth   # 第20轮备份
└── ...
```

## 🔧 检查点内容

每个检查点文件包含：

```python
{
    'epoch': 15,                          # 训练轮次
    'model_state_dict': {...},            # 模型参数
    'optimizer_state_dict': {...},        # 优化器状态
    'scheduler_state_dict': {...},        # 学习率调度器状态
    'val_acc': 0.8542,                   # 验证准确率
    'best_val_acc': 0.8654,              # 最佳验证准确率
    'train_losses': [...],                # 训练损失历史
    'val_losses': [...],                  # 验证损失历史
    'train_accuracies': [...],            # 训练准确率历史
    'val_accuracies': [...],              # 验证准确率历史
    'timestamp': '2024-01-15T10:30:00',   # 保存时间
    'config': {                           # 训练配置
        'learning_rate': 0.001,
        'batch_size': 32,
        'sequence_length': 30,
        'num_epochs': 100
    }
}
```

## ⚠️ 注意事项

### 1. 数据一致性
- 确保使用相同的数据集
- 确保数据预处理方式一致

### 2. 配置兼容性
- 模型架构必须相同
- 超参数建议保持一致

### 3. 存储空间
- 检查点文件较大（通常几十MB）
- 定期清理旧的备份文件

### 4. 训练中断
- 使用 `Ctrl+C` 安全中断训练
- 避免强制终止进程

## 🛠️ 故障排除

### 问题1：检查点加载失败
```
错误: 加载检查点失败
```
**解决方案**:
- 检查文件是否损坏
- 确认模型架构是否匹配
- 删除损坏的检查点文件重新训练

### 问题2：内存不足
```
错误: CUDA out of memory
```
**解决方案**:
- 减小批次大小
- 使用梯度累积
- 清理GPU缓存

### 问题3：学习率异常
```
警告: 学习率调度器状态不匹配
```
**解决方案**:
- 检查训练轮次设置
- 重新创建学习率调度器
- 使用 `--force` 开始新训练

## 📊 最佳实践

### 1. 定期备份
- 重要训练阶段手动备份
- 使用版本控制管理检查点

### 2. 监控训练
- 观察损失曲线连续性
- 检查学习率变化
- 验证准确率趋势

### 3. 实验管理
- 记录训练配置
- 保存实验日志
- 对比不同检查点性能

## 🎯 示例工作流

### 长期训练项目
```bash
# 第一天：开始训练
python main.py --mode train

# 第二天：继续训练
python checkpoint_training.py
# 选择继续训练

# 查看训练进度
python checkpoint_training.py --list

# 如果需要调整参数，开始新训练
python checkpoint_training.py --force
```

### 实验对比
```bash
# 保存当前最佳模型
cp models/best_model.pth models/experiment_1_best.pth

# 尝试新的训练策略
python checkpoint_training.py --force

# 对比结果
python evaluate.py --model models/experiment_1_best.pth
python evaluate.py --model models/best_model.pth
```

## 📈 性能优化

### 1. 检查点频率
- 默认每轮保存（适合短期训练）
- 长期训练可调整为每N轮保存

### 2. 存储优化
- 压缩检查点文件
- 只保存必要状态
- 定期清理旧文件

### 3. 加载速度
- 使用SSD存储检查点
- 预加载到内存
- 并行加载数据

通过断点训练功能，您可以更灵活地管理长期训练任务，提高训练效率和可靠性！
