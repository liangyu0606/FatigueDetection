"""
配置文件 - 打哈欠检测项目
"""
import os

# 数据路径配置
DATASET_ROOT = "dataset"
DASH_PATH = os.path.join(DATASET_ROOT, "Dash")
MIRROR_PATH = os.path.join(DATASET_ROOT, "Mirror")

# dlib模型路径
DLIB_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# 数据预处理参数
FACE_SIZE = (128, 128)  # 人脸图像resize尺寸
SEQUENCE_LENGTH = 30    # LSTM输入序列长度（帧数）
OVERLAP_RATIO = 0.5     # 滑动窗口重叠比例

# 模型参数
CNN_CHANNELS = [32, 64, 128]  # CNN各层通道数
LSTM_HIDDEN_SIZE = 256        # LSTM隐藏层大小
LSTM_LAYERS = 2               # LSTM层数
DROPOUT_RATE = 0.3            # Dropout比例
NUM_CLASSES = 2               # 分类数：0-正常/说话，1-打哈欠

# 训练参数
BATCH_SIZE = 8  # 减小批次大小以节省内存
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# 调试模式参数（已关闭，进行完整训练）
DEBUG_MODE = False  # 设置为False进行完整训练
DEBUG_MAX_VIDEOS = 3  # 调试模式下每个文件夹最多处理的视频数
DEBUG_MAX_FRAMES = 600  # 调试模式下每个视频最多处理的帧数
DEBUG_EPOCHS = 5  # 调试模式下的训练轮数
DEBUG_BATCH_SIZE = 4  # 调试模式下的批次大小

# 数据增强参数
AUGMENTATION = {
    'brightness_range': 0.2,
    'contrast_range': 0.2,
    'rotation_range': 10,
    'horizontal_flip': True
}

# 输出路径
OUTPUT_DIR = "output"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "models")
LOG_PATH = os.path.join(OUTPUT_DIR, "logs")
PROCESSED_DATA_PATH = os.path.join(OUTPUT_DIR, "processed_data")

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
