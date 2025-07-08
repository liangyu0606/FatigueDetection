# 疲劳检测系统 (Fatigue Detection System)

基于深度学习的实时疲劳检测系统，使用LSTM+残差网络检测眼睛闭合和打哈欠行为，支持用户管理和管理员功能。

## 🌟 功能特点

- **实时检测**: 基于摄像头的实时疲劳状态监控
- **多模态检测**: 结合眼睛闭合程度(EAR)和打哈欠行为(MAR)
- **深度学习**: 使用LSTM+残差网络处理时序特征
- **双界面系统**: 用户界面和管理员界面
- **用户管理**: 支持用户注册、登录和权限管理
- **数据标注**: 内置数据标注工具
- **模型训练**: 完整的数据预处理和训练流水线
- **系统日志**: 详细的运行日志和错误追踪

## 📁 项目结构

```
FatigueDetection/
├── fatigue_gui_user.py         # 用户界面主程序
├── fatigue_gui_admin.py        # 管理员界面主程序
├── config.py                   # 系统配置文件
├── model.py                    # 深度学习模型定义
├── utils.py                    # 工具函数
├── database_config.py          # 数据库配置
├── system_logger.py            # 系统日志模块
├── train.py                    # 模型训练脚本
├── evaluate.py                 # 模型评估脚本
├── resume_training.py          # 恢复训练脚本
├── checkpoint_training.py      # 检查点训练脚本
├── data_preprocessing.py       # 数据预处理
├── improved_labeling_tool.py   # 数据标注工具
├── dataset.py                  # 数据集处理
├── requirements.txt            # 依赖包列表
├── dataset/                    # 原始数据集
│   ├── Dash/                  # Dash数据集
│   └── Mirror/                # Mirror数据集
├── data/                      # 处理后的数据
├── output/                    # 输出目录
│   ├── models/               # 训练好的模型
│   ├── logs/                 # 日志文件
│   └── processed_data/       # 处理后的数据
└── static/                    # 静态资源
    └── warning.mp3           # 警告音频
```

## 🚀 快速开始

### 环境要求

- Python 3.7+
- 摄像头设备
- 8GB+ 内存推荐

### 安装依赖

```bash
pip install -r requirements.txt
```

### 下载预训练模型

下载dlib面部特征点检测模型：
```bash
# 下载 shape_predictor_68_face_landmarks.dat
# 放置到 output/ 目录下
```

### 运行系统

#### 用户界面
```bash
python fatigue_gui_user.py
```

#### 管理员界面
```bash
python fatigue_gui_admin.py
```

## 💻 系统界面

### 用户界面功能
- **用户注册/登录**: 安全的用户认证系统
- **实时检测**: 摄像头实时疲劳监控
- **状态显示**: 疲劳状态、EAR/MAR值实时显示
- **参数调节**: 可调节检测阈值和敏感度
- **统计信息**: FPS、检测次数等统计数据
- **夜间模式**: 低光环境检测优化

### 管理员界面功能
- **用户管理**: 查看、编辑、删除用户
- **数据管理**: 数据集管理和预处理
- **模型训练**: 训练新模型或更新现有模型
- **系统监控**: 系统性能和使用情况监控
- **日志查看**: 详细的系统运行日志

## 🧠 技术架构

### 核心算法
1. **面部特征提取**: 使用dlib检测68个面部特征点
2. **特征计算**: 
   - EAR (Eye Aspect Ratio): 眼睛长宽比
   - MAR (Mouth Aspect Ratio): 嘴部长宽比
3. **时序建模**: LSTM+残差网络处理连续帧特征
4. **疲劳判断**: 集成多个模型的预测结果

### 深度学习模型
- **主模型**: LSTM+残差网络的疲劳检测模型
- **眼睛检测**: 专门的眼睛闭合检测模型
- **哈欠检测**: 专门的哈欠行为检测模型
- **集成预测**: 多模型融合提高准确率

### 数据库设计
- **用户表**: 存储用户信息和权限
- **检测记录**: 存储检测历史和统计数据
- **系统日志**: 记录系统运行状态

## 🔧 模型训练

### 数据预处理
```bash
python data_preprocessing.py --dataset dataset --output data
```

### 训练模型
```bash
python train.py --data data --epochs 100
```

### 评估模型
```bash
python evaluate.py --model output/models/best_model.h5
```

### 恢复训练
```bash
python resume_training.py --checkpoint output/models/checkpoint.h5
```

## 📊 数据标注

使用内置标注工具标注新数据：
```bash
python improved_labeling_tool.py
```

标注功能：
- 视频播放控制
- 帧级别标注
- 批量标注
- 标注验证

## ⚙️ 配置说明

### config.py 主要配置项
```python
# 数据路径
DATASET_ROOT = "dataset"
DLIB_PREDICTOR_PATH = "output/shape_predictor_68_face_landmarks.dat"

# 模型参数
SEQUENCE_LENGTH = 30    # LSTM输入序列长度
LSTM_HIDDEN_SIZE = 256  # LSTM隐藏层大小
DROPOUT_RATE = 0.3      # Dropout比例

# 训练参数
BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
```

### 检测阈值调节
- **EAR阈值**: 眼睛闭合检测敏感度
- **MAR阈值**: 哈欠检测敏感度
- **疲劳阈值**: 综合疲劳判断阈值
- **连续帧数**: 连续检测帧数要求

## 🔍 故障排除

### 常见问题

1. **摄像头无法打开**
   - 检查摄像头连接
   - 确认摄像头权限
   - 尝试不同的摄像头ID

2. **模型加载失败**
   - 检查模型文件路径
   - 确认模型文件完整性
   - 重新训练模型

3. **检测精度低**
   - 调整检测阈值
   - 改善光照条件
   - 重新训练模型

4. **性能问题**
   - 降低视频分辨率
   - 调整序列长度
   - 使用GPU加速

### 日志查看
- GUI界面：查看系统日志面板
- 文件日志：`output/logs/` 目录
- 控制台输出：运行时终端信息

## 📈 性能优化

### 建议配置
- **CPU**: Intel i5或同等性能
- **内存**: 8GB以上
- **摄像头**: 720p 30fps
- **存储**: SSD推荐

### 优化建议
1. 使用GPU加速训练 (安装tensorflow-gpu)
2. 调整序列长度平衡精度和速度
3. 根据硬件性能调整摄像头分辨率
4. 定期清理日志文件

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 📞 技术支持

如有问题或建议，请：
1. 查看文档和FAQ
2. 检查系统日志
3. 提交Issue描述问题
4. 提供系统环境信息

---

**注意**: 首次运行前请确保已正确安装所有依赖包并下载必要的预训练模型。
