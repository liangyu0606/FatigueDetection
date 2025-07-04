import datetime
import math
import os
import sys
import threading
import time

import cv2
import dlib
import numpy as np
import pygame
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtGui import QImage
from PySide6.QtCore import Qt
from imutils import face_utils
from scipy.spatial import distance as dist
from camera_config import camera_config

# 尝试导入深度学习库，如果失败则使用占位符
try:
    # 首先定义TENSORFLOW_AVAILABLE变量
    TENSORFLOW_AVAILABLE = False

    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Conv2D, MaxPooling2D, Flatten

    # 如果导入成功，设置TENSORFLOW_AVAILABLE为True
    TENSORFLOW_AVAILABLE = True

    # 只有当TensorFlow可用时才尝试导入Input层
    if TENSORFLOW_AVAILABLE:
        from tensorflow.keras.layers import Input
        pass

except ImportError as e:
    print(f"TensorFlow导入失败: {e}")
    TENSORFLOW_AVAILABLE = False

# 尝试导入PyTorch用于CNN+LSTM打哈欠检测
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
    print("✅ PyTorch可用，将启用CNN+LSTM打哈欠检测")
except ImportError as e:
    print(f"PyTorch导入失败: {e}")
    PYTORCH_AVAILABLE = False


# CNN+LSTM模型定义（用于打哈欠检测）
if PYTORCH_AVAILABLE:
    class YawnCNNLSTM(nn.Module):
        """专门用于打哈欠检测的CNN+LSTM模型"""
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(YawnCNNLSTM, self).__init__()

            # CNN layers - Extract spatial features
            self.cnn = nn.Sequential(
                nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2)
            )

            # CNN output channels is 256
            cnn_output_size = 256

            # LSTM layers - Process temporal features
            self.lstm = nn.LSTM(
                input_size=cnn_output_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2,
                bidirectional=True
            )

            # Fully connected layers - Output classification results
            self.fc = nn.Sequential(
                nn.Linear(hidden_size * 2, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, output_size)
            )

        def forward(self, x):
            batch_size, seq_len, features = x.size()

            # 重塑输入以通过CNN: (batch_size, features, seq_len)
            x = x.permute(0, 2, 1)

            # 通过CNN
            cnn_out = self.cnn(x)

            # 重塑CNN输出以通过LSTM: (batch_size, seq_len, cnn_features)
            cnn_out = cnn_out.permute(0, 2, 1)

            # 通过LSTM
            lstm_out, _ = self.lstm(cnn_out)

            # 取最后一个时间步的输出
            lstm_out = lstm_out[:, -1, :]

            # 通过全连接层
            output = self.fc(lstm_out)

            return output


    class YawnDetector:
        """专门用于打哈欠检测的类"""
        def __init__(self, model_path=None, seq_length=30, consecutive_frames=15):
            self.model = None
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if PYTORCH_AVAILABLE else None
            self.seq_length = seq_length
            self.features_buffer = []
            self.is_available = False

            # 添加连续帧判断逻辑（参考real_pljc）
            self.consecutive_frames = consecutive_frames
            self.fatigue_frames = 0
            self.frame_count = 0

            if PYTORCH_AVAILABLE and model_path and os.path.exists(model_path):
                self.load_model(model_path)

        def load_model(self, model_path):
            """加载训练好的CNN+LSTM模型"""
            try:
                # 初始化模型（与训练时的结构保持一致）
                self.model = YawnCNNLSTM(
                    input_size=138,  # 2 (EAR, MAR) + 68*2 (landmark coordinates)
                    hidden_size=64,
                    num_layers=1,
                    output_size=1
                ).to(self.device)

                # 加载模型权重
                self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
                self.model.eval()
                self.is_available = True
                print(f"✅ CNN+LSTM打哈欠检测模型加载成功: {model_path}")

            except Exception as e:
                print(f"❌ CNN+LSTM打哈欠检测模型加载失败: {e}")
                self.model = None
                self.is_available = False

        def extract_features(self, landmarks, ear, mar, frame_height=480):
            """提取特征向量（与real_pljc保持一致）"""
            if landmarks is None:
                return None

            try:
                # 归一化关键点坐标（以鼻尖为基准，使用帧高度归一化）
                nose = landmarks[30]  # 鼻尖关键点
                normalized_landmarks = (landmarks - nose).flatten() / frame_height  # 使用帧高度归一化

                # 组合特征：EAR, MAR + 68个关键点的x,y坐标
                features = np.concatenate([[ear, mar], normalized_landmarks])
                return features

            except Exception as e:
                print(f"特征提取失败: {e}")
                return None

        def update_buffer(self, features):
            """更新特征缓冲区"""
            if features is not None:
                self.features_buffer.append(features)
                if len(self.features_buffer) > self.seq_length:
                    self.features_buffer.pop(0)

        def predict_yawn(self, detection_enabled=True):
            """预测是否打哈欠（与real_pljc保持一致的逻辑）"""
            if not self.is_available or len(self.features_buffer) < self.seq_length:
                return False, 0.0

            try:
                # 准备输入序列
                input_seq = np.array([self.features_buffer])
                input_tensor = torch.FloatTensor(input_seq).to(self.device)

                # 模型预测
                with torch.no_grad():
                    logits = self.model(input_tensor).item()
                    prediction = torch.sigmoid(torch.tensor(logits)).item()

                # 更新疲劳状态（参考real_pljc的连续帧判断逻辑）
                self.frame_count += 1
                if prediction >= 0.5:  # 单帧预测阈值
                    self.fatigue_frames += 1
                else:
                    self.fatigue_frames = 0

                # 判定疲劳需要连续帧数达到阈值
                is_fatigued = self.fatigue_frames >= self.consecutive_frames

                # 如果检测被禁用（冷却期），重置连续帧计数
                if not detection_enabled and is_fatigued:
                    self.fatigue_frames = 0
                    is_fatigued = False

                return is_fatigued, prediction

            except Exception as e:
                print(f"CNN+LSTM打哈欠预测失败: {e}")
                return False, 0.0

        def reset_state(self):
            """重置检测器状态（用于冷却期）"""
            self.fatigue_frames = 0
            self.frame_count = 0


# 如果PyTorch不可用，定义占位符类
if not PYTORCH_AVAILABLE:
    class YawnCNNLSTM:
        def __init__(self, *args, **kwargs):
            pass

    class YawnDetector:
        def __init__(self, *args, **kwargs):
            self.is_available = False


# 处理TensorFlow导入失败的情况
try:
    # 这个try块是为了兼容原有的except ImportError
    pass
except ImportError:
    # 如果导入失败，定义基本占位符
    class Sequential:
        def __init__(self):
            pass

        def add(self, layer):
            pass

    class ResNet50:
        def __init__(self, *args, **kwargs):
            pass

    class LSTM:
        def __init__(self, *args, **kwargs):
            pass

    class Dense:
        def __init__(self, *args, **kwargs):
            pass

    class Conv2D:
        def __init__(self, *args, **kwargs):
            pass

    class MaxPooling2D:
        def __init__(self, *args, **kwargs):
            pass

    class Flatten:
        def __init__(self, *args, **kwargs):
            pass

    tf = None
    TENSORFLOW_AVAILABLE = False

import main_ui


# pyside6-uic -o main_ui.py main.ui

class MainUI(QtWidgets.QWidget, main_ui.Ui_Form):
    # 信号，在UI线程中，不能在其他线程直接操作UI
    thread_signal = QtCore.Signal(dict)

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 初始化摄像头列表
        self.cameras = []
        self.init_camera_list()

        # 连接信号
        self.pushButton.clicked.connect(self.button_clicked)

        # 初始化疲劳统计模块
        try:
            from fatigue_statistics import FatigueStatistics
            self.fatigue_stats = FatigueStatistics()
            print("✅ 疲劳统计模块已初始化")
        except Exception as e:
            print(f"⚠️ 疲劳统计模块初始化失败: {e}")
            self.fatigue_stats = None

        # 初始化深度学习模型
        self.resnet = None
        self.cnn_model = None
        self.lstm_model = None

        # 初始化CNN疲劳检测器
        self.cnn_detector = None

        # 初始化CNN+LSTM打哈欠检测器
        self.yawn_detector = None

        if TENSORFLOW_AVAILABLE:
            try:
                self.init_models()
                self.init_cnn_detector()
            except Exception as e:
                print(f"模型初始化失败: {e}")
        else:
            print("TensorFlow未安装，使用简化功能")

        # 初始化CNN+LSTM打哈欠检测器
        if PYTORCH_AVAILABLE:
            try:
                self.init_yawn_detector()
            except Exception as e:
                print(f"CNN+LSTM打哈欠检测器初始化失败: {e}")

        # 连接信号
        # self.thread_signal.connect(self.thread_singnal_slot)

        # 六个功能是否要用
        self.fun = [True] * 6

        # 兼容新旧界面的复选框名称
        try:
            # 尝试使用新界面的复选框
            if hasattr(self, 'checkBox'):
                self.checkBox_11 = self.checkBox
                self.checkBox_12 = self.checkBox_2
                self.checkBox_21 = self.checkBox_3
                self.checkBox_22 = self.checkBox_4
                # 为缺失的复选框创建占位符
                if not hasattr(self, 'checkBox_31'):
                    from PySide6.QtWidgets import QCheckBox
                    self.checkBox_31 = QCheckBox()
                    self.checkBox_32 = QCheckBox()

            self.checkBox_11.setChecked(self.fun[0])
            self.checkBox_12.setChecked(self.fun[1])
            self.checkBox_21.setChecked(self.fun[2])
            self.checkBox_22.setChecked(self.fun[3])
            if hasattr(self, 'checkBox_31'):
                self.checkBox_31.setChecked(self.fun[4])
            if hasattr(self, 'checkBox_32'):
                self.checkBox_32.setChecked(self.fun[5])

            self.checkBox_11.stateChanged.connect(self.select_changed)
            self.checkBox_12.stateChanged.connect(self.select_changed)
            self.checkBox_21.stateChanged.connect(self.select_changed)
            self.checkBox_22.stateChanged.connect(self.select_changed)
            if hasattr(self, 'checkBox_31'):
                self.checkBox_31.stateChanged.connect(self.select_changed)
            if hasattr(self, 'checkBox_32'):
                self.checkBox_32.stateChanged.connect(self.select_changed)
        except AttributeError as e:
            print(f"界面组件初始化警告: {e}")
            # 创建默认的功能状态
            pass

        # 阈值
        self.values = [3,2,3,5,2]

        # 兼容新旧界面的spinBox名称
        try:
            # 为缺失的spinBox创建占位符
            if not hasattr(self, 'spinBox_2'):
                from PySide6.QtWidgets import QSpinBox
                self.spinBox_2 = QSpinBox()
                self.spinBox_2.setValue(2)

            if hasattr(self, 'spinBox_1'):
                self.spinBox_1.setValue(self.values[0])
                self.spinBox_1.valueChanged.connect(self.value_changed)
            if hasattr(self, 'spinBox_2'):
                self.spinBox_2.setValue(self.values[1])
                self.spinBox_2.valueChanged.connect(self.value_changed)
            if hasattr(self, 'spinBox_3'):
                self.spinBox_3.setValue(self.values[2])
                self.spinBox_3.valueChanged.connect(self.value_changed)
            if hasattr(self, 'spinBox_4'):
                self.spinBox_4.setValue(self.values[3])
                self.spinBox_4.valueChanged.connect(self.value_changed)
            if hasattr(self, 'spinBox_5'):
                self.spinBox_5.setValue(self.values[4])
                self.spinBox_5.valueChanged.connect(self.value_changed)
        except AttributeError as e:
            print(f"SpinBox初始化警告: {e}")

        self.thread_signal.connect(self.thread_singnal_slot)

        # 兼容新旧界面的图片显示标签
        if not hasattr(self, 'label_img'):
            from PySide6.QtWidgets import QLabel
            self.label_img = QLabel()
            self.label_img.setScaledContents(True)
            # 设置最小尺寸确保有足够空间显示人脸
            self.label_img.setMinimumSize(640, 480)
            # 如果有视频显示区域，可以将label_img添加到其中
            print("创建了默认的图片显示标签")
        else:
            self.label_img.setScaledContents(True)
            # 确保现有的label_img也有合适的最小尺寸
            if self.label_img.minimumSize().width() < 640:
                self.label_img.setMinimumSize(640, 480)

        if hasattr(self, 'plainTextEdit_tip'):
            self.plainTextEdit_tip.appendPlainText('等待开始\n')
        else:
            print("等待开始")


        """参数"""
        # 默认为摄像头0
        self.VIDEO_STREAM = 0
        self.CAMERA_STYLE = False  # False未打开摄像头，True摄像头已打开

        # 优化后的眨眼检测参数 - 进一步提高敏感度
        self.EYE_AR_THRESH = 0.20  # 进一步降低阈值，提高敏感度（原0.22）
        self.EYE_AR_CONSEC_FRAMES = 2  # 保持较低的连续帧要求
        self.EYE_AR_UPPER_THRESH = 0.40  # 适当提高上限，避免过滤正常眨眼

        # 优化后的打哈欠检测参数 - 进一步提高敏感度
        self.MAR_THRESH = 0.40  # 进一步降低阈值，提高敏感度（原0.45）
        self.MAR_DURATION_THRESH = 0.6  # 进一步降低哈欠持续时间阈值（秒）
        self.MOUTH_AR_CONSEC_FRAMES = 2  # 减少连续帧要求

        # 优化后的瞌睡点头检测参数
        self.HAR_THRESH_LOW = 15.0  # 轻微点头角度阈值（度）
        self.HAR_THRESH_HIGH = 25.0  # 明显点头角度阈值（度）
        self.NOD_AR_CONSEC_FRAMES = 4  # 增加连续帧要求，减少误检

        # 其他检测参数
        self.AR_CONSEC_FRAMES_check = 3
        self.OUT_AR_CONSEC_FRAMES_check = 5

        """计数"""
        # 初始化帧计数器和眨眼总数
        self.COUNTER = 0
        self.TOTAL = 0
        # 初始化帧计数器和打哈欠总数
        self.mCOUNTER = 0
        self.mTOTAL = 0
        # 初始化帧计数器和点头总数
        self.hCOUNTER = 0
        self.hTOTAL = 0
        # 离职时间长度
        self.oCOUNTER = 0

        # 新增：改进的检测状态跟踪
        self.yawn_start_time = None  # 哈欠开始时间
        self.last_ear_values = []  # 最近的EAR值历史
        self.last_mar_values = []  # 最近的MAR值历史
        self.fatigue_score = 0.0  # 疲劳评分
        self.baseline_ear = 0.3  # 基线EAR值（将动态调整）
        self.baseline_mar = 0.4  # 基线MAR值（将动态调整）

        # 打哈欠冷却机制
        self.last_yawn_time = None  # 上次检测到打哈欠的时间
        self.yawn_cooldown_seconds = 3.0  # 打哈欠冷却时间（秒）
        self.yawn_detection_enabled = True  # 打哈欠检测是否启用

        # 自适应阈值调整
        self.calibration_frames = 0  # 校准帧数
        self.calibration_period = 300  # 校准周期（帧数）
        self.adaptive_mode = True  # 是否启用自适应模式

        """姿态"""
        # 世界坐标系(UVW)：填写3D参考点，该模型参考http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
        self.object_pts = np.float32([[6.825897, 6.760612, 4.402142],  #33左眉左上角
                                 [1.330353, 7.122144, 6.903745],  #29左眉右角
                                 [-1.330353, 7.122144, 6.903745], #34右眉左角
                                 [-6.825897, 6.760612, 4.402142], #38右眉右上角
                                 [5.311432, 5.485328, 3.987654],  #13左眼左上角
                                 [1.789930, 5.393625, 4.413414],  #17左眼右上角
                                 [-1.789930, 5.393625, 4.413414], #25右眼左上角
                                 [-5.311432, 5.485328, 3.987654], #21右眼右上角
                                 [2.005628, 1.409845, 6.165652],  #55鼻子左上角
                                 [-2.005628, 1.409845, 6.165652], #49鼻子右上角
                                 [2.774015, -2.080775, 5.048531], #43嘴左上角
                                 [-2.774015, -2.080775, 5.048531],#39嘴右上角
                                 [0.000000, -3.116408, 6.097667], #45嘴中央下角
                                 [0.000000, -7.415691, 4.070434]])#6下巴角

        # 相机坐标系(XYZ)：添加相机内参
        self.K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
                 0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
                 0.0, 0.0, 1.0]# 等价于矩阵[fx, 0, cx; 0, fy, cy; 0, 0, 1]
        # 图像中心坐标系(uv)：相机畸变参数[k1, k2, p1, p2, k3]
        self.D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

        # 像素坐标系(xy)：填写凸轮的本征和畸变系数
        self.cam_matrix = np.array(self.K).reshape(3, 3).astype(np.float32)
        self.dist_coeffs = np.array(self.D).reshape(5, 1).astype(np.float32)

        # 重新投影3D点的世界坐标轴以验证结果姿势
        self.reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                                       [10.0, 10.0, -10.0],
                                       [10.0, -10.0, -10.0],
                                       [10.0, -10.0, 10.0],
                                       [-10.0, 10.0, 10.0],
                                       [-10.0, 10.0, -10.0],
                                       [-10.0, -10.0, -10.0],
                                       [-10.0, -10.0, 10.0]])
        # 绘制正方体12轴
        self.line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
                          [4, 5], [5, 6], [6, 7], [7, 4],
                          [0, 4], [1, 5], [2, 6], [3, 7]]


        # 线程
        self.thread = None
        self.sound_thread = None
        self.is_running = True  # 添加运行状态标志

    def safe_emit_signal(self, data):
        """安全地发送信号"""
        try:
            if self.is_running and hasattr(self, 'thread_signal'):
                # 进一步减少调试输出，只在重要事件时打印
                if data['type'] == 'msg' and ('疲劳' in str(data.get('value', '')) or 'CNN检测' in str(data.get('value', ''))):
                    print(f"发送重要信号: {data['type']} - {data.get('value', '')}")
                elif data['type'] == 'res' and hasattr(self, 'frame_count') and self.frame_count % 300 == 0:  # 每10秒打印一次状态
                    print(f"系统状态正常: FPS={data.get('value', ['', '', '0'])[2]}")

                self.thread_signal.emit(data)
                return True
            else:
                # 只在第一次出现错误时记录，避免重复日志
                if not hasattr(self, '_signal_error_logged'):
                    print(f"检测已停止，停止信号发送")
                    self._signal_error_logged = True
                return False
        except RuntimeError as e:
            # Qt对象已被销毁的情况
            if not hasattr(self, '_runtime_error_logged'):
                print(f"UI已关闭，停止信号发送")
                self._runtime_error_logged = True
            return False
        except Exception as e:
            if not hasattr(self, '_unknown_error_logged'):
                print(f"信号发送异常: {e}")
                self._unknown_error_logged = True
            return False

    def _optimize_camera_brightness(self):
        """优化摄像头设置 - 使用配置文件参数"""
        if self.cap is None or not self.cap.isOpened():
            return

        print("正在优化摄像头设置...")

        try:
            # 使用配置文件中的摄像头属性
            camera_props = camera_config.get_camera_properties()

            for prop, value in camera_props.items():
                try:
                    self.cap.set(prop, value)
                except Exception as e:
                    # 某些属性可能不被支持，继续设置其他属性
                    pass

            print(f"摄像头优化设置完成:")
            print(f"  分辨率: {camera_config.CAMERA_WIDTH}x{camera_config.CAMERA_HEIGHT}")
            print(f"  FPS: {camera_config.CAMERA_FPS}")
            print(f"  亮度: {camera_config.BRIGHTNESS}")
            print(f"  对比度: {camera_config.CONTRAST}")
            print(f"  增益: {camera_config.GAIN}")
            print("  已针对暗环境和帧率进行优化")

        except Exception as e:
            print(f"摄像头参数设置失败: {e}")
            # 即使设置失败也继续，使用默认参数

    def _enhance_dark_frame(self, frame):
        """增强暗图像 - 使用配置文件参数"""
        if frame is None:
            return frame

        # 检查图像亮度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)

        # 如果图像太暗，应用多步骤增强
        if mean_brightness < camera_config.DARK_THRESHOLD:
            enhanced_frame = frame.copy()

            # 步骤1: 基础亮度和对比度调整
            enhanced_frame = cv2.convertScaleAbs(
                enhanced_frame,
                alpha=camera_config.BRIGHTNESS_ALPHA,
                beta=camera_config.BRIGHTNESS_BETA
            )

            # 步骤2: 应用CLAHE（对比度限制自适应直方图均衡化）
            clahe = cv2.createCLAHE(
                clipLimit=camera_config.CLAHE_CLIP_LIMIT,
                tileGridSize=camera_config.CLAHE_TILE_SIZE
            )

            # 在LAB色彩空间中处理亮度通道
            lab = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            enhanced_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # 步骤3: Gamma校正进一步提亮暗部
            gamma = camera_config.GAMMA_CORRECTION
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            enhanced_frame = cv2.LUT(enhanced_frame, table)

            # 步骤4: 与原图混合，保持自然效果
            enhanced_frame = cv2.addWeighted(
                frame,
                1.0 - camera_config.ENHANCEMENT_WEIGHT,
                enhanced_frame,
                camera_config.ENHANCEMENT_WEIGHT,
                0
            )

            return enhanced_frame

        return frame

    def init_camera_list(self):
        """初始化摄像头列表"""
        self.cameras = []
        print("正在扫描可用摄像头...")

        for i in range(5):
            print(f"  测试摄像头索引 {i}...")
            cap = None
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    # 测试是否能读取帧
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        self.cameras.append(i)
                        print(f"    ✓ 摄像头 {i} 可用")
                    else:
                        print(f"    ✗ 摄像头 {i} 无法读取帧")
                else:
                    print(f"    ✗ 摄像头 {i} 无法打开")

                if cap is not None:
                    cap.release()

            except Exception as e:
                print(f"    ✗ 摄像头 {i} 测试异常: {e}")
                if cap is not None:
                    try:
                        cap.release()
                    except:
                        pass

        print(f"找到 {len(self.cameras)} 个摄像头设备")
        if self.cameras:
            print("可用摄像头:", self.cameras)
        else:
            print("未找到摄像头设备")

    def _init_camera_robust(self, camera_index):
        """简化的摄像头初始化方法 - 避免闪烁"""
        print(f"正在初始化摄像头 {camera_index}...")

        # 优先使用DirectShow后端（Windows最稳定）
        try:
            print(f"  使用DirectShow后端...")
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

            if cap.isOpened():
                # 立即测试读取
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    print(f"摄像头 {camera_index} 初始化成功")
                    return cap
                else:
                    print(f"  无法读取帧")
                    cap.release()
            else:
                print(f"  无法打开摄像头")
                if cap is not None:
                    cap.release()

        except Exception as e:
            print(f"  DirectShow初始化失败: {e}")
            if 'cap' in locals() and cap is not None:
                try:
                    cap.release()
                except:
                    pass

        # 如果DirectShow失败，尝试默认后端
        try:
            print(f"  尝试默认后端...")
            cap = cv2.VideoCapture(camera_index)

            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    print(f"摄像头 {camera_index} 初始化成功（默认后端）")
                    return cap
                else:
                    cap.release()
            else:
                if cap is not None:
                    cap.release()

        except Exception as e:
            print(f"  默认后端初始化失败: {e}")
            if 'cap' in locals() and cap is not None:
                try:
                    cap.release()
                except:
                    pass

        print(f"摄像头 {camera_index} 初始化失败")
        return None

    def _check_camera_health(self):
        """检查摄像头健康状态"""
        if not hasattr(self, 'cap') or self.cap is None:
            return False

        try:
            # 检查摄像头是否仍然打开
            if not self.cap.isOpened():
                return False

            # 尝试获取一些基本属性
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

            if width <= 0 or height <= 0:
                return False

            return True

        except Exception as e:
            print(f"摄像头健康检查失败: {e}")
            return False

    def _reconnect_camera(self):
        """重新连接摄像头"""
        print("尝试重新连接摄像头...")

        # 释放当前摄像头
        if hasattr(self, 'cap') and self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None

        # 短暂等待
        time.sleep(0.2)

        # 尝试重新打开
        try:
            self.cap = cv2.VideoCapture(self.VIDEO_STREAM, cv2.CAP_DSHOW)
            if self.cap.isOpened():
                # 重新设置基本参数
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                # 测试读取
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    print("摄像头重连成功")
                    return True
                else:
                    print("摄像头重连后无法读取帧")
                    self.cap.release()
                    self.cap = None
                    return False
            else:
                print("摄像头重连失败")
                return False

        except Exception as e:
            print(f"摄像头重连异常: {e}")
            return False



    def init_models(self):
        """初始化深度学习模型"""
        self.resnet = self._init_resnet()
        self.cnn_model = self._init_cnn_model()
        self.lstm_model = self._init_lstm_model()

        if all([self.resnet, self.cnn_model, self.lstm_model]):
            print("所有模型初始化完成")
        else:
            print("部分模型初始化失败")

    def init_cnn_detector(self):
        """初始化CNN疲劳检测器"""
        try:
            # 尝试导入CNN检测器
            from simple_cnn_detector import CNNFatigueDetector

            # 检查是否有训练好的模型
            model_path = './model/fatigue_model_mobilenet.h5'
            if os.path.exists(model_path):
                self.cnn_detector = CNNFatigueDetector(model_path)
                if self.cnn_detector.is_available():
                    print("✅ CNN疲劳检测器已加载")
                else:
                    print("⚠️ CNN检测器加载失败")
                    self.cnn_detector = None
            else:
                print("⚠️ 未找到训练好的CNN模型，将创建简化版本")
                # 创建一个使用预训练ResNet的简化检测器
                self.cnn_detector = self._create_simple_detector()

        except ImportError as e:
            print(f"CNN检测器导入失败: {e}")
            self.cnn_detector = None
        except Exception as e:
            print(f"CNN检测器初始化失败: {e}")
            self.cnn_detector = None

    def _create_simple_detector(self):
        """创建简化的疲劳检测器"""
        class SimpleFatigueDetector:
            def __init__(self, resnet_model):
                self.resnet = resnet_model
                self.available = resnet_model is not None

            def is_available(self):
                return self.available

            def predict_fatigue(self, face_image):
                """使用ResNet特征进行简单的疲劳检测"""
                if not self.available or face_image is None or face_image.size == 0:
                    return None

                try:
                    # 预处理图像
                    img = cv2.resize(face_image, (224, 224))
                    img = tf.keras.applications.resnet50.preprocess_input(img)

                    # 提取特征
                    features = self.resnet.predict(np.expand_dims(img, axis=0), verbose=0)

                    # 简单的疲劳判断（基于特征的统计特性）
                    feature_mean = np.mean(features)
                    feature_std = np.std(features)

                    # 简化的疲劳判断逻辑
                    fatigue_score = feature_mean * feature_std
                    confidence = min(abs(fatigue_score) * 100, 1.0)

                    is_fatigue = fatigue_score < -0.1  # 阈值可调整

                    return {
                        'predicted_class': 'drowsy' if is_fatigue else 'alert',
                        'confidence': confidence,
                        'fatigue_level': '疲劳' if is_fatigue else '正常',
                        'fatigue_detected': is_fatigue,
                        'feature_mean': feature_mean,
                        'feature_std': feature_std
                    }

                except Exception as e:
                    print(f"简化检测器预测失败: {e}")
                    return None

        return SimpleFatigueDetector(self.resnet)

    def init_yawn_detector(self):
        """初始化CNN+LSTM打哈欠检测器"""
        try:
            # 检查是否有训练好的模型 - 现在模型文件在model文件夹中
            model_path = './model/best_fatigue_model.pth'
            if not os.path.exists(model_path):
                # 尝试其他可能的路径
                model_path = '../real_pljc/models/best_fatigue_model.pth'
            if not os.path.exists(model_path):
                # 尝试相对路径
                model_path = './real_pljc/models/best_fatigue_model.pth'
            if not os.path.exists(model_path):
                # 尝试绝对路径
                model_path = 'D:/code/PythonProject2/real_pljc/models/best_fatigue_model.pth'

            if os.path.exists(model_path):
                self.yawn_detector = YawnDetector(model_path)
                if self.yawn_detector.is_available:
                    print("✅ CNN+LSTM打哈欠检测器已加载")
                else:
                    print("⚠️ CNN+LSTM打哈欠检测器加载失败")
                    self.yawn_detector = None
            else:
                print("⚠️ 未找到CNN+LSTM打哈欠检测模型")
                self.yawn_detector = None

        except Exception as e:
            print(f"CNN+LSTM打哈欠检测器初始化失败: {e}")
            self.yawn_detector = None

    def _init_resnet(self):
        """初始化ResNet50模型"""
        try:
            print("正在加载ResNet50模型...")
            resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
            print("ResNet50模型加载成功")
            return resnet
        except Exception as e:
            print(f"ResNet50模型加载失败: {e}")
            return None

    def _init_cnn_model(self):
        """初始化CNN模型"""
        try:
            print("正在创建CNN模型...")
            cnn_model = Sequential()

            if TENSORFLOW_AVAILABLE:
                cnn_model.add(Input(shape=(64, 64, 3)))

            cnn_model.add(Conv2D(32, (3, 3), activation='relu'))
            cnn_model.add(MaxPooling2D((2, 2)))
            cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
            cnn_model.add(MaxPooling2D((2, 2)))
            cnn_model.add(Flatten())
            cnn_model.add(Dense(128, activation='relu'))
            cnn_model.add(Dense(1, activation='sigmoid'))
            print("CNN模型创建成功")
            return cnn_model
        except Exception as e:
            print(f"CNN模型创建失败: {e}")
            return None

    def _init_lstm_model(self):
        """初始化LSTM模型"""
        try:
            print("正在创建LSTM模型...")
            lstm_model = Sequential()

            if TENSORFLOW_AVAILABLE:
                lstm_model.add(Input(shape=(10, 2048)))

            lstm_model.add(LSTM(64, return_sequences=True))
            lstm_model.add(LSTM(32))
            lstm_model.add(Dense(1, activation='sigmoid'))
            print("LSTM模型创建成功")
            return lstm_model
        except Exception as e:
            print(f"LSTM模型创建失败: {e}")
            return None

    def extract_resnet_features(self, face_img):
        """使用ResNet提取人脸特征"""
        if self.resnet is None or face_img.size == 0:
            return None

        # 预处理图像
        img = cv2.resize(face_img, (224, 224))
        img = tf.keras.applications.resnet50.preprocess_input(img)
        features = self.resnet.predict(np.expand_dims(img, axis=0))
        return features

    def detect_cnn_fatigue(self, face_img):
        """使用CNN进行疲劳检测"""
        if self.cnn_model is None or face_img.size == 0:
            return None

        # 预处理图像
        img = cv2.resize(face_img, (64, 64))
        img = img / 255.0  # 归一化
        prediction = self.cnn_model.predict(np.expand_dims(img, axis=0))
        return '疲劳' if prediction[0][0] > 0.5 else '正常'

    def analyze_lstm_fatigue(self, features_sequence):
        """使用LSTM分析时序特征"""
        if self.lstm_model is None or not features_sequence:
            return None

        # 确保序列长度一致
        if len(features_sequence) != 10:
            return None

        prediction = self.lstm_model.predict(np.expand_dims(features_sequence, axis=0))
        return '深度疲劳' if prediction[0][0] > 0.7 else '轻度疲劳'

    def calculate_fatigue_score(self, blink_count, yawn_count, nod_count, time_window=60):
        """计算综合疲劳评分"""
        # 权重设置（基于疲劳检测研究）
        blink_weight = 0.3  # 眨眼权重
        yawn_weight = 0.5   # 哈欠权重（更重要）
        nod_weight = 0.4    # 点头权重

        # 标准化到每分钟的频率
        blink_rate = (blink_count / time_window) * 60
        yawn_rate = (yawn_count / time_window) * 60
        nod_rate = (nod_count / time_window) * 60

        # 正常基线值（每分钟）
        normal_blink_rate = 15  # 正常眨眼频率
        normal_yawn_rate = 0.5  # 正常哈欠频率
        normal_nod_rate = 1     # 正常点头频率

        # 计算偏离度
        blink_deviation = max(0, blink_rate - normal_blink_rate) / normal_blink_rate
        yawn_deviation = max(0, yawn_rate - normal_yawn_rate) / normal_yawn_rate
        nod_deviation = max(0, nod_rate - normal_nod_rate) / normal_nod_rate

        # 综合评分
        fatigue_score = (blink_deviation * blink_weight +
                        yawn_deviation * yawn_weight +
                        nod_deviation * nod_weight)

        return min(fatigue_score, 1.0)  # 限制在0-1之间

    def get_fatigue_level(self, score):
        """根据评分获取疲劳等级"""
        if score < 0.2:
            return '正常'
        elif score < 0.4:
            return '轻微疲劳'
        elif score < 0.7:
            return '中度疲劳'
        else:
            return '重度疲劳'

    def adaptive_threshold_adjustment(self, ear, mar):
        """自适应阈值调整"""
        if not self.adaptive_mode:
            return

        self.calibration_frames += 1

        # 在校准期间收集数据
        if self.calibration_frames <= self.calibration_period:
            # 更新基线值
            self.update_baseline_values(ear, mar)

            # 校准完成后调整阈值
            if self.calibration_frames == self.calibration_period:
                self._adjust_thresholds()
                print(f"自适应校准完成 - EAR基线: {self.baseline_ear:.3f}, MAR基线: {self.baseline_mar:.3f}")

        # 定期重新校准（每1000帧）
        elif self.calibration_frames % 1000 == 0:
            self._adjust_thresholds()

    def _adjust_thresholds(self):
        """根据基线值调整检测阈值"""
        if len(self.last_ear_values) >= 10:
            # 动态调整眨眼阈值
            ear_std = np.std(self.last_ear_values)
            self.EYE_AR_THRESH = max(0.2, self.baseline_ear - 2 * ear_std)
            self.EYE_AR_UPPER_THRESH = self.baseline_ear + 2 * ear_std

        if len(self.last_mar_values) >= 10:
            # 动态调整哈欠阈值
            mar_std = np.std(self.last_mar_values)
            self.MAR_THRESH = max(0.5, self.baseline_mar + 1.5 * mar_std)

    def get_detection_confidence(self, ear, mar):
        """计算检测置信度"""
        ear_confidence = 1.0
        mar_confidence = 1.0

        if len(self.last_ear_values) >= 10:
            ear_std = np.std(self.last_ear_values)
            ear_z_score = abs(ear - self.baseline_ear) / (ear_std + 1e-6)
            ear_confidence = min(1.0, ear_z_score / 3.0)  # 3-sigma规则

        if len(self.last_mar_values) >= 10:
            mar_std = np.std(self.last_mar_values)
            mar_z_score = abs(mar - self.baseline_mar) / (mar_std + 1e-6)
            mar_confidence = min(1.0, mar_z_score / 3.0)

        return (ear_confidence + mar_confidence) / 2.0

    def get_head_pose(self,shape):# 头部姿态估计
        # （像素坐标集合）填写2D参考点，注释遵循https://ibug.doc.ic.ac.uk/resources/300-W/
        # 17左眉左上角/21左眉右角/22右眉左上角/26右眉右上角/36左眼左上角/39左眼右上角/42右眼左上角/
        # 45右眼右上角/31鼻子左上角/35鼻子右上角/48左上角/54嘴右上角/57嘴中央下角/8下巴角
        image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                                shape[39], shape[42], shape[45], shape[31], shape[35],
                                shape[48], shape[54], shape[57], shape[8]])
        # solvePnP计算姿势——求解旋转和平移矩阵：
        # rotation_vec表示旋转矩阵，translation_vec表示平移矩阵，cam_matrix与K矩阵对应，dist_coeffs与D矩阵对应。
        _, rotation_vec, translation_vec = cv2.solvePnP(self.object_pts, image_pts, self.cam_matrix, self.dist_coeffs)
        # projectPoints重新投影误差：原2d点和重投影2d点的距离（输入3d点、相机内参、相机畸变、r、t，输出重投影2d点）
        reprojectdst, _ = cv2.projectPoints(self.reprojectsrc, rotation_vec, translation_vec, self.cam_matrix,self.dist_coeffs)
        reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))# 以8行2列显示

        # 计算欧拉角calc euler angle
        # 参考https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#decomposeprojectionmatrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec) #罗德里格斯公式（将旋转矩阵转换为旋转向量）
        pose_mat = cv2.hconcat((rotation_mat, translation_vec)) # 水平拼接，vconcat垂直拼接
        # decomposeProjectionMatrix将投影矩阵分解为旋转矩阵和相机矩阵
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

        # 确保从数组中提取标量值
        pitch = math.radians(euler_angle[0].item())
        yaw = math.radians(euler_angle[1].item())
        roll = math.radians(euler_angle[2].item())

        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))
        #print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))

        return reprojectdst, euler_angle# 投影误差，欧拉角
    def eye_aspect_ratio(self,eye):
        # 垂直眼标志（X，Y）坐标
        A = dist.euclidean(eye[1], eye[5])# 计算两个集合之间的欧式距离
        B = dist.euclidean(eye[2], eye[4])
        # 计算水平之间的欧几里得距离
        # 水平眼标志（X，Y）坐标
        C = dist.euclidean(eye[0], eye[3])
        # 眼睛长宽比的计算
        ear = (A + B) / (2.0 * C)
        # 返回眼睛的长宽比
        return ear

    def mouth_aspect_ratio(self,mouth):# 嘴部
        A = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59
        B = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
        C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
        mar = (A + B) / (2.0 * C)
        return mar

    def update_baseline_values(self, ear, mar):
        """动态更新基线值"""
        # 保持最近50个值的历史
        self.last_ear_values.append(ear)
        self.last_mar_values.append(mar)

        if len(self.last_ear_values) > 50:
            self.last_ear_values.pop(0)
        if len(self.last_mar_values) > 50:
            self.last_mar_values.pop(0)

        # 更新基线值（使用中位数，更稳定）
        if len(self.last_ear_values) >= 10:
            self.baseline_ear = np.median(self.last_ear_values)
        if len(self.last_mar_values) >= 10:
            self.baseline_mar = np.median(self.last_mar_values)

    def is_valid_blink(self, ear):
        """改进的眨眼检测 - 简化版本"""
        # 使用固定阈值，更容易触发
        thresh = self.EYE_AR_THRESH

        # 检查是否在合理范围内（过滤异常值）
        if ear > self.EYE_AR_UPPER_THRESH or ear < 0.1:  # 添加下限检查
            return False

        is_blink = ear < thresh

        # 添加调试信息
        if not hasattr(self, '_blink_detail_counter'):
            self._blink_detail_counter = 0
        self._blink_detail_counter += 1

        if self._blink_detail_counter % 60 == 0:  # 每60帧打印一次详细信息
            print(f"🔍 眨眼检测详情 - EAR: {ear:.3f}, 阈值: {thresh:.3f}, 是否眨眼: {is_blink}")

        return is_blink

    def is_valid_yawn(self, mar, current_time):
        """改进的打哈欠检测 - 更宽松的条件"""
        # 使用简化的阈值检测
        thresh = self.MAR_THRESH

        is_mouth_open = mar > thresh

        if is_mouth_open:
            if self.yawn_start_time is None:
                self.yawn_start_time = current_time
                print(f"🔍 开始检测哈欠，MAR: {mar:.3f}, 阈值: {thresh:.3f}")
            return False  # 还在张嘴过程中
        else:
            if self.yawn_start_time is not None:
                # 检查持续时间
                duration = (current_time - self.yawn_start_time).total_seconds()
                self.yawn_start_time = None

                print(f"🔍 哈欠持续时间: {duration:.2f}秒")
                # 进一步放宽哈欠持续时间要求：0.3-3.0秒
                is_valid = 0.3 <= duration <= 3.0
                if is_valid:
                    print(f"✅ 有效哈欠，持续时间: {duration:.2f}秒")
                else:
                    print(f"❌ 无效哈欠，持续时间: {duration:.2f}秒（要求0.3-3.0秒）")
                return is_valid
            return False


    def _learning_face(self):
        """dlib的初始化调用 - 增强版本"""
        try:
            # 检查模型文件是否存在
            model_path = "./model/shape_predictor_68_face_landmarks.dat"
            if not os.path.exists(model_path):
                data = {
                    'type':'msg',
                    'value':u"❌ 模型文件不存在，请下载 shape_predictor_68_face_landmarks.dat\n"
                }
                self.safe_emit_signal(data)
                return

            # 初始化检测器
            print("🔍 正在初始化dlib人脸检测器...")
            self.detector = dlib.get_frontal_face_detector()
            print("🔍 正在加载特征点预测器...")
            self.predictor = dlib.shape_predictor(model_path)

            # 验证模型加载
            if self.detector is None or self.predictor is None:
                print("❌ 检测器或预测器为None")
                data = {
                    'type':'msg',
                    'value':u"❌ 人脸检测模型加载失败\n"
                }
                self.safe_emit_signal(data)
                return

            print("✅ 检测器初始化成功")
            print(f"   检测器类型: {type(self.detector)}")
            print(f"   预测器类型: {type(self.predictor)}")

            data = {
                'type':'msg',
                'value':u"✅ 人脸检测模型加载成功!!!\n"
            }
            self.safe_emit_signal(data)

        except Exception as e:
            data = {
                'type':'msg',
                'value':f"❌ 模型初始化失败: {e}\n"
            }
            self.safe_emit_signal(data)
            return

        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

        self.cap = None

        # 简化的摄像头初始化
        print(f"尝试打开摄像头，索引: {self.VIDEO_STREAM}")

        # 直接尝试打开摄像头，不过度复杂化
        success = False
        for camera_index in [self.VIDEO_STREAM, 0, 1]:  # 尝试当前索引、0、1
            try:
                print(f"  尝试摄像头索引 {camera_index}...")
                self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

                if self.cap.isOpened():
                    # 基本设置
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    # 测试读取
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        print(f"  ✓ 摄像头 {camera_index} 工作正常")
                        self.VIDEO_STREAM = camera_index
                        self.CAMERA_STYLE = True
                        success = True
                        data['value'] = f"打开摄像头成功(索引{camera_index})!!!"
                        break
                    else:
                        print(f"  ✗ 摄像头 {camera_index} 无法读取")
                        self.cap.release()
                        self.cap = None
                else:
                    print(f"  ✗ 摄像头 {camera_index} 无法打开")
                    if self.cap is not None:
                        self.cap.release()
                        self.cap = None

            except Exception as e:
                print(f"  ✗ 摄像头 {camera_index} 异常: {e}")
                if hasattr(self, 'cap') and self.cap is not None:
                    try:
                        self.cap.release()
                    except:
                        pass
                    self.cap = None

        if not success:
            data['value'] = u"摄像头打开失败!!!"
            print("未找到可用的摄像头设备")
        else:
            # 简化的参数设置
            self._optimize_camera_brightness()
        self.safe_emit_signal(data)

        # 打印最终摄像头状态
        if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            backend = self.cap.getBackendName()
            print(f"摄像头状态: 已打开")
            print(f"  分辨率: {width}x{height}")
            print(f"  设置FPS: {fps}")
            print(f"  后端: {backend}")

            # 如果FPS为0，尝试手动设置
            if fps == 0:
                print("检测到FPS为0，尝试手动设置...")
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                new_fps = self.cap.get(cv2.CAP_PROP_FPS)
                print(f"  重新设置后FPS: {new_fps}")
        else:
            print("未找到可用的视频源")

        # 初始化FPS计算变量
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.actual_fps = 0

        # 初始化时间变量
        t_time = datetime.datetime.now()
        e_time = datetime.datetime.now()
        h_time = datetime.datetime.now()

        # 成功打开视频，循环读取视频流
        print("开始视频流处理...")

        # 初始化性能监控变量
        frame_count = 0
        error_count = 0
        last_status_time = time.time()

        while self.is_running:
            try:
                # 在循环开始时立即检查停止标志
                if not self.is_running:
                    break

                start_time = datetime.datetime.now()
                res = ['-' for _ in range(9)]

                # 初始化CNN检测结果变量，确保在整个循环中都有定义
                cnn_result = None

                # 简单的摄像头状态检查
                if not hasattr(self, 'cap') or self.cap is None:
                    print("摄像头未初始化")
                    break

                # 简化的摄像头检查 - 只在必要时检查
                if not self.cap.isOpened():
                    print("摄像头连接丢失，尝试重新连接...")
                    if not self._reconnect_camera():
                        print("摄像头重连失败，停止检测")
                        break
                    continue

                # 优化的帧读取 - 减少卡顿
                flag, im_rd = self.cap.read()

                if not flag or im_rd is None or im_rd.size == 0:
                    error_count += 1
                    # 减少错误报告频率
                    if error_count % 30 == 0:
                        print(f"读取帧失败，错误计数: {error_count}")

                    # 如果连续错误太多，尝试重连
                    if error_count > 100:
                        print("错误过多，尝试重连摄像头...")
                        if not self._reconnect_camera():
                            break
                        error_count = 0

                    # 使用配置文件中的延迟时间
                    time.sleep(camera_config.MAIN_LOOP_DELAY)
                    continue

                # 成功读取帧，重置错误计数
                if error_count > 0:
                    error_count = 0

                frame_count += 1

                # 跳帧处理 - 使用配置文件参数
                skip_detection = (frame_count % camera_config.FRAME_SKIP_DETECTION != 0)

                # 每10秒报告一次状态，减少输出频率
                current_time_float = time.time()
                if current_time_float - last_status_time >= 10.0:
                    fps = frame_count / (current_time_float - last_status_time)
                    print(f"处理状态: {frame_count} 帧, FPS: {fps:.1f}, 错误: {error_count}")
                    frame_count = 0
                    last_status_time = current_time_float

                # 改进的图像处理 - 增强人脸检测
                try:
                    # 验证图像格式和尺寸
                    if len(im_rd.shape) != 3 or im_rd.shape[2] != 3:
                        print(f"图像格式异常: {im_rd.shape}")
                        continue

                    height, width = im_rd.shape[:2]
                    if height < 100 or width < 100:
                        print(f"图像尺寸过小: {width}x{height}")
                        continue

                    # 应用暗图像增强 - 解决摄像头太暗的问题
                    im_rd = self._enhance_dark_frame(im_rd)

                    # 转换为灰度图像
                    img_gray = cv2.cvtColor(im_rd, cv2.COLOR_BGR2GRAY)

                    # 验证灰度图像
                    if img_gray is None or img_gray.size == 0:
                        print("灰度转换失败")
                        continue

                    # 增强图像质量以提高人脸检测率
                    mean_brightness = np.mean(img_gray)

                    # 创建增强版本的图像
                    enhanced_gray = img_gray.copy()

                    # 增强的亮度调整 - 针对暗环境优化
                    if mean_brightness < 120:  # 提高阈值，更积极地增强暗图像
                        # 图像太暗，大幅增加亮度和对比度
                        enhanced_gray = cv2.convertScaleAbs(enhanced_gray, alpha=1.6, beta=40)

                        # 使用CLAHE进行局部对比度增强
                        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                        enhanced_gray = clahe.apply(enhanced_gray)

                        # 额外的Gamma校正来提亮暗部
                        gamma = 0.7  # 小于1的gamma值会提亮图像
                        inv_gamma = 1.0 / gamma
                        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                        enhanced_gray = cv2.LUT(enhanced_gray, table)

                    elif mean_brightness > 200:
                        # 图像太亮，降低亮度
                        enhanced_gray = cv2.convertScaleAbs(enhanced_gray, alpha=0.8, beta=-15)

                    # 轻微降噪，保持边缘清晰
                    enhanced_gray = cv2.bilateralFilter(enhanced_gray, 5, 50, 50)

                except Exception as e:
                    print(f"图像处理失败: {e}")
                    continue

                # 优化的人脸检测 - 跳帧处理减少卡顿
                faces = []

                # 使用跳帧策略：只在特定帧进行人脸检测，其他帧使用缓存结果
                if not skip_detection:
                    try:
                        # 检查检测器是否已初始化
                        if not hasattr(self, 'detector') or self.detector is None:
                            if frame_count % 60 == 0:  # 减少错误信息频率
                                print("❌ 人脸检测器未初始化")
                            faces = []
                        else:
                            # 多策略人脸检测，提高检测成功率
                            faces = []

                            # 策略1: 使用增强后的图像，不同上采样级别
                            for upsample in [0, 1]:  # 先尝试0（更快），再尝试1
                                if len(faces) == 0:
                                    try:
                                        faces = self.detector(enhanced_gray, upsample)
                                        if len(faces) > 0:
                                            break
                                    except:
                                        continue

                            # 策略2: 如果增强图像失败，尝试原始灰度图像
                            if len(faces) == 0:
                                try:
                                    faces = self.detector(img_gray, 0)
                                except:
                                    pass

                            # 策略3: 尝试直方图均衡化
                            if len(faces) == 0:
                                try:
                                    equalized = cv2.equalizeHist(img_gray)
                                    faces = self.detector(equalized, 0)
                                except:
                                    pass

                            # 策略4: 尝试缩放图像（有时小图像检测效果更好）
                            if len(faces) == 0:
                                try:
                                    small_gray = cv2.resize(img_gray, (320, 240))
                                    small_faces = self.detector(small_gray, 0)
                                    # 将坐标缩放回原图
                                    faces = []
                                    for face in small_faces:
                                        scaled_face = dlib.rectangle(
                                            int(face.left() * 2),
                                            int(face.top() * 2),
                                            int(face.right() * 2),
                                            int(face.bottom() * 2)
                                        )
                                        faces.append(scaled_face)
                                except:
                                    pass

                            # 过滤检测结果 - 更宽松的过滤条件
                            if len(faces) > 0:
                                filtered_faces = []
                                for face in faces:
                                    width = face.right() - face.left()
                                    height = face.bottom() - face.top()
                                    # 放宽尺寸限制，提高检测成功率
                                    if 30 <= width <= 600 and 30 <= height <= 600:
                                        # 确保人脸在图像范围内
                                        if (face.left() >= 0 and face.top() >= 0 and
                                            face.right() < im_rd.shape[1] and face.bottom() < im_rd.shape[0]):
                                            filtered_faces.append(face)
                                faces = filtered_faces

                            # 缓存检测结果供跳帧使用
                            if hasattr(self, 'last_faces'):
                                self.last_faces = faces
                            else:
                                self.last_faces = faces

                    except Exception as e:
                        if frame_count % 60 == 0:  # 减少错误信息频率
                            print(f"人脸检测失败: {e}")
                        faces = []
                else:
                    # 跳帧时使用上次的检测结果
                    if hasattr(self, 'last_faces'):
                        faces = self.last_faces
                    else:
                        faces = []
                # 如果检测到人脸
                if (len(faces) != 0):
                    res[0] = '识别到人脸'
                    # enumerate方法同时返回数据对象的索引和数据，k为索引，d为faces中的对象
                    for _, d in enumerate(faces):
                        # 用红色矩形框出人脸
                        cv2.rectangle(im_rd, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 1)
                        # 使用预测器得到68点数据的坐标
                        shape = self.predictor(im_rd, d)
                        # 圆圈显示每个特征点
                        for i in range(68):
                            cv2.circle(im_rd, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), -1, 8)
                        # 将脸部特征信息转换为数组array的格式
                        shape = face_utils.shape_to_np(shape)

                        # 提取人脸区域用于CNN分析
                        face_img = im_rd[d.top():d.bottom(), d.left():d.right()]

                        # 使用CNN进行疲劳检测 - 降低调用频率以提高性能
                        # 只在每10帧调用一次CNN检测，减少计算负担
                        if (self.cnn_detector and self.cnn_detector.is_available() and
                            frame_count % 10 == 0):
                            try:
                                cnn_result = self.cnn_detector.predict_fatigue(face_img)
                                if cnn_result:
                                    # 在界面上显示CNN检测结果
                                    cnn_text = f"CNN: {cnn_result['fatigue_level']} ({cnn_result['confidence']:.2f})"
                                    cv2.putText(im_rd, cnn_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                            except Exception as e:
                                if frame_count % 60 == 0:  # 减少错误信息频率
                                    print(f"CNN检测失败: {e}")
                                cnn_result = None  # 确保在异常情况下重置为None

                        # 预先计算EAR和MAR，因为CNN+LSTM模型需要这些值
                        # 提取左眼和右眼坐标
                        leftEye = shape[lStart:lEnd]
                        rightEye = shape[rStart:rEnd]
                        # 构造函数计算左右眼的EAR值，使用平均值作为最终的EAR
                        leftEAR = self.eye_aspect_ratio(leftEye)
                        rightEAR = self.eye_aspect_ratio(rightEye)
                        ear = (leftEAR + rightEAR) / 2.0

                        # 嘴巴坐标和MAR计算
                        mouth = shape[mStart:mEnd]
                        mar = self.mouth_aspect_ratio(mouth)

                        # 获取当前时间（确保在所有地方都可用）
                        current_time = datetime.datetime.now()

                        """
                        打哈欠 - 集成CNN+LSTM和启发式检测（带冷却机制）
                        """
                        if self.fun[1]:
                            # 使用cv2.convexHull获得凸包位置，使用drawContours画出轮廓位置进行画图操作
                            mouthHull = cv2.convexHull(mouth)
                            cv2.drawContours(im_rd, [mouthHull], -1, (0, 255, 0), 1)

                            # 检查打哈欠冷却状态
                            if self.last_yawn_time is not None:
                                time_since_last_yawn = (current_time - self.last_yawn_time).total_seconds()
                                if time_since_last_yawn < self.yawn_cooldown_seconds:
                                    self.yawn_detection_enabled = False
                                    # 显示冷却状态
                                    cooldown_remaining = self.yawn_cooldown_seconds - time_since_last_yawn
                                    cv2.putText(im_rd, f"Yawn Cooldown: {cooldown_remaining:.1f}s", (10, 160),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                else:
                                    self.yawn_detection_enabled = True

                            # CNN+LSTM打哈欠检测（只在启用时进行）
                            cnn_lstm_yawn_detected = False
                            cnn_lstm_confidence = 0.0

                            if self.yawn_detector and self.yawn_detector.is_available:
                                try:
                                    # 提取特征并更新缓冲区（传递帧高度用于归一化）
                                    frame_height = im_rd.shape[0]
                                    features = self.yawn_detector.extract_features(shape, ear, mar, frame_height)
                                    if features is not None:
                                        self.yawn_detector.update_buffer(features)

                                        # 进行CNN+LSTM预测（传递检测启用状态）
                                        cnn_lstm_yawn_detected, cnn_lstm_confidence = self.yawn_detector.predict_yawn(self.yawn_detection_enabled)

                                        # 在界面上显示CNN+LSTM检测结果
                                        if frame_count % 10 == 0:  # 每10帧显示一次
                                            status = "ENABLED" if self.yawn_detection_enabled else "COOLDOWN"
                                            cnn_lstm_text = f"CNN+LSTM: {cnn_lstm_confidence:.2f} (连续:{self.yawn_detector.fatigue_frames}) [{status}]"
                                            color = (0, 0, 255) if cnn_lstm_yawn_detected else (0, 255, 0)
                                            cv2.putText(im_rd, cnn_lstm_text, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                                except Exception as e:
                                    if frame_count % 60 == 0:  # 减少错误信息频率
                                        print(f"CNN+LSTM打哈欠检测失败: {e}")

                            # 只使用CNN+LSTM检测打哈欠（移除启发式检测）
                            yawn_detected = False
                            detection_method = ""

                            # 注意：cnn_lstm_yawn_detected已经包含了连续帧判断
                            if cnn_lstm_yawn_detected:
                                # CNN+LSTM检测到打哈欠（已经通过连续帧验证）
                                yawn_detected = True
                                detection_method = "CNN+LSTM"
                                print(f"🔍 CNN+LSTM检测到打哈欠: 置信度={cnn_lstm_confidence:.3f}, 连续帧={self.yawn_detector.fatigue_frames}")

                            # 如果CNN+LSTM不可用，显示提示信息
                            if not (self.yawn_detector and self.yawn_detector.is_available):
                                # 在界面上显示CNN+LSTM不可用的提示
                                cv2.putText(im_rd, "CNN+LSTM Yawn Detection: UNAVAILABLE", (10, 140),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                if self._yawn_debug_counter % 300 == 0:  # 每300帧提示一次
                                    print("⚠️ CNN+LSTM打哈欠检测不可用，请检查模型文件和PyTorch安装")

                            # 添加哈欠检测调试信息
                            if not hasattr(self, '_yawn_debug_counter'):
                                self._yawn_debug_counter = 0
                            self._yawn_debug_counter += 1
                            if self._yawn_debug_counter % 60 == 0:  # 每60帧打印一次
                                if self.yawn_detector and self.yawn_detector.is_available:
                                    print(f"🔍 哈欠检测 - MAR: {mar:.3f}, CNN+LSTM: {cnn_lstm_confidence:.3f}, 连续帧: {self.yawn_detector.fatigue_frames}")
                                else:
                                    print(f"🔍 哈欠检测 - CNN+LSTM不可用，跳过检测")

                            # 只在检测启用且检测到打哈欠时才计数和记录
                            if yawn_detected and self.yawn_detection_enabled:
                                self.mTOTAL += 1
                                print(f"🥱 检测到哈欠！方法: {detection_method}, 总计: {self.mTOTAL}, MAR: {mar:.3f}, CNN+LSTM置信度: {cnn_lstm_confidence:.3f}")
                                self.safe_emit_signal({'type':'msg','value':time.strftime('%Y-%m-%d %H:%M ', time.localtime()) + f"打哈欠({detection_method})"})
                                res[4] = '哈欠'

                                # 设置冷却时间
                                self.last_yawn_time = current_time
                                self.yawn_detection_enabled = False
                                print(f"🔒 打哈欠检测进入冷却期 {self.yawn_cooldown_seconds} 秒")

                                # 记录哈欠事件到统计数据库
                                if self.fatigue_stats:
                                    try:
                                        from fatigue_statistics import FatigueEvent
                                        # 使用CNN+LSTM的置信度
                                        confidence = cnn_lstm_confidence
                                        event = FatigueEvent(
                                            timestamp=current_time,
                                            event_type='yawn',
                                            value=1.0,
                                            confidence=confidence,
                                            additional_data={
                                                'mouth_aspect_ratio': mar,
                                                'detection_method': detection_method,
                                                'cnn_lstm_confidence': cnn_lstm_confidence,
                                                'consecutive_frames': self.yawn_detector.fatigue_frames if self.yawn_detector else 0
                                            }
                                        )
                                        self.fatigue_stats.record_event(event)
                                        print(f"📊 记录哈欠事件到数据库 (方法: {detection_method})")
                                    except Exception as e:
                                        print(f"记录哈欠事件失败: {e}")
                            elif mar > self.MAR_THRESH:
                                res[4] = '张嘴'
                                # 添加张嘴状态的调试信息
                                if self._yawn_debug_counter % 120 == 0:  # 每120帧打印一次
                                    print(f"🔍 检测到张嘴但未达到哈欠条件 - MAR: {mar:.3f}")
                            else:
                                res[4] = '闭嘴'
                            # cv2.putText(im_rd, "COUNTER: {}".format(self.mCOUNTER), (150, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            #             0.7, (0, 0, 255), 2)
                            # cv2.putText(im_rd, "MAR: {:.2f}".format(mar), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            #             (0, 0, 255), 2)
                            # cv2.putText(im_rd, "Yawning: {}".format(self.mTOTAL), (450, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            #             (255, 255, 0), 2)
                        else:
                            pass
                        """
                        眨眼 - 改进版本
                        """
                        if self.fun[0]:
                            # 执行自适应阈值调整
                            self.adaptive_threshold_adjustment(ear, mar)

                            leftEyeHull = cv2.convexHull(leftEye)
                            rightEyeHull = cv2.convexHull(rightEye)
                            # 使用cv2.convexHull获得凸包位置，使用drawContours画出轮廓位置进行画图操作
                            cv2.drawContours(im_rd, [leftEyeHull], -1, (0, 255, 0), 1)
                            cv2.drawContours(im_rd, [rightEyeHull], -1, (0, 255, 0), 1)

                        # 使用改进的眨眼检测
                        if self.is_valid_blink(ear):
                            self.COUNTER += 1
                            res[5] = '闭眼'
                            # 添加调试信息
                            if not hasattr(self, '_blink_debug_counter'):
                                self._blink_debug_counter = 0
                            self._blink_debug_counter += 1
                            if self._blink_debug_counter % 30 == 0:  # 每30帧打印一次
                                print(f"🔍 眨眼检测中 - COUNTER: {self.COUNTER}, EAR: {ear:.3f}, 阈值: {self.EYE_AR_THRESH}")
                        else:
                            # 如果连续帧数达到阈值，则表示进行了一次眨眼活动
                            if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                                self.TOTAL += 1
                                print(f"👁️ 检测到眨眼！总计: {self.TOTAL}, EAR: {ear:.3f}, COUNTER: {self.COUNTER}")
                                self.safe_emit_signal({'type':'msg','value':time.strftime('%Y-%m-%d %H:%M ', time.localtime()) + u"眨眼"})

                                # 记录眨眼事件到统计数据库
                                if self.fatigue_stats:
                                    try:
                                        from fatigue_statistics import FatigueEvent
                                        event = FatigueEvent(
                                            timestamp=datetime.datetime.now(),
                                            event_type='blink',
                                            value=1.0,
                                            confidence=0.8,
                                            additional_data={'eye_aspect_ratio': ear}
                                        )
                                        self.fatigue_stats.record_event(event)
                                        print(f"📊 记录眨眼事件到数据库")
                                    except Exception as e:
                                        print(f"记录眨眼事件失败: {e}")
                            # 重置眼帧计数器
                            self.COUNTER = 0
                            res[5] = '睁眼'
                        # 第十四步：进行画图操作，同时使用cv2.putText将眨眼次数进行显示
                        # cv2.putText(im_rd, "Faces: {}".format(len(faces)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        #             (0, 0, 255), 2)
                        # cv2.putText(im_rd, "COUNTER: {}".format(self.COUNTER), (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        #             (0, 0, 255), 2)
                        # cv2.putText(im_rd, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        #             (0, 0, 255), 2)
                        # cv2.putText(im_rd, "Blinks: {}".format(self.TOTAL), (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        #             (255, 255, 0), 2)
                    else:
                        pass
                    """
                    瞌睡点头 - 改进版本
                    """
                    if self.fun[2]:
                        # 获取头部姿态
                        _, euler_angle = self.get_head_pose(shape)
                        pitch = abs(euler_angle[0, 0])  # 取pitch旋转角度的绝对值
                        yaw = abs(euler_angle[1, 0])    # 取yaw旋转角度的绝对值

                        # 改进的点头检测：考虑多个角度
                        nod_level = '正'

                        # 轻微点头
                        if self.HAR_THRESH_LOW <= pitch <= self.HAR_THRESH_HIGH and yaw < 20:
                            self.hCOUNTER += 1
                            nod_level = '轻微倾斜'
                        # 明显点头
                        elif pitch > self.HAR_THRESH_HIGH and yaw < 25:
                            self.hCOUNTER += 2  # 明显点头计数更多
                            nod_level = '明显倾斜'
                        else:
                            # 如果连续帧数达到阈值，则表示瞌睡点头一次
                            if self.hCOUNTER >= self.NOD_AR_CONSEC_FRAMES:
                                self.hTOTAL += 1
                                self.safe_emit_signal({'type': 'msg', 'value': time.strftime('%Y-%m-%d %H:%M ',
                                                                                                   time.localtime()) + u"瞌睡点头"})

                                # 记录点头事件到统计数据库
                                if self.fatigue_stats:
                                    try:
                                        from fatigue_statistics import FatigueEvent
                                        event = FatigueEvent(
                                            timestamp=datetime.datetime.now(),
                                            event_type='nod',
                                            value=1.0,
                                            confidence=0.7,
                                            additional_data={'pitch': float(pitch), 'yaw': float(yaw)}
                                        )
                                        self.fatigue_stats.record_event(event)
                                        print(f"📊 记录点头事件到数据库")
                                    except Exception as e:
                                        print(f"记录点头事件失败: {e}")
                            # 重置点头帧计数器
                            self.hCOUNTER = 0
                            nod_level = '正'

                        res[3] = nod_level
                        # 绘制正方体12轴(视频流尺寸过大时，reprojectdst会超出int范围，建议压缩检测视频尺寸)
                        # for start, end in self.line_pairs:
                        #     x1, y1 = reprojectdst[start]
                        #     x2, y2 = reprojectdst[end]
                            #cv2.line(im_rd, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255))
                        # 显示角度结果
                        # cv2.putText(im_rd, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (10, 90),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), thickness=2)  # GREEN
                        # cv2.putText(im_rd, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (150, 90),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness=2)  # BLUE
                        # cv2.putText(im_rd, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (300, 90),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), thickness=2)  # RED
                        # cv2.putText(im_rd, "Nod: {}".format(self.hTOTAL), (450, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        #             (255, 255, 0), 2)
                    else:
                        pass

                # 改进的疲劳状态判断
                res[6] = str(self.TOTAL)   # 眨眼次数
                res[7] = str(self.mTOTAL)  # 哈欠次数
                res[8] = str(self.hTOTAL)  # 点头次数

                # 添加调试信息，每30秒打印一次计数状态
                if not hasattr(self, '_debug_counter_time'):
                    self._debug_counter_time = current_time

                if (current_time - self._debug_counter_time).total_seconds() >= 30:
                    print(f"🔍 当前计数状态 - 眨眼: {self.TOTAL}, 哈欠: {self.mTOTAL}, 点头: {self.hTOTAL}")
                    print(f"🔍 res数组: {res}")
                    self._debug_counter_time = current_time

                # 计算综合疲劳评分
                current_time = datetime.datetime.now()
                time_window = 60  # 60秒时间窗口

                self.fatigue_score = self.calculate_fatigue_score(
                    self.TOTAL, self.mTOTAL, self.hTOTAL, time_window
                )

                # 根据评分确定疲劳等级
                traditional_fatigue_level = self.get_fatigue_level(self.fatigue_score)

                # 集成CNN检测结果
                if cnn_result and cnn_result.get('confidence', 0) > 0.6:
                    # 检查是否检测到疲劳
                    is_fatigue = (cnn_result.get('predicted_class') == 'drowsy' or
                                 cnn_result.get('fatigue_level') == '疲劳')

                    if is_fatigue:
                        # 如果CNN检测到疲劳且置信度高，优先使用CNN结果
                        res[1] = cnn_result.get('fatigue_level', '疲劳')
                        # 发送CNN检测消息
                        self.safe_emit_signal({
                            'type': 'msg',
                            'value': f"CNN检测到疲劳: {cnn_result.get('fatigue_level', '疲劳')} (置信度: {cnn_result.get('confidence', 0):.2f})"
                        })

                        # 记录CNN疲劳检测事件
                        if self.fatigue_stats:
                            try:
                                from fatigue_statistics import FatigueEvent
                                event = FatigueEvent(
                                    timestamp=current_time,
                                    event_type='fatigue_state',
                                    value=2.0,  # CNN检测到的疲劳
                                    confidence=cnn_result.get('confidence', 0),
                                    additional_data={'source': 'CNN', 'level': cnn_result.get('fatigue_level', '疲劳')}
                                )
                                self.fatigue_stats.record_event(event)
                                print(f"📊 记录CNN疲劳事件到数据库")
                            except Exception as e:
                                print(f"记录CNN疲劳事件失败: {e}")
                    else:
                        # CNN检测为正常，使用传统方法结果
                        res[1] = traditional_fatigue_level
                else:
                    # CNN置信度不够或无结果，使用传统方法结果
                    res[1] = traditional_fatigue_level

                # 记录疲劳状态到数据库（每10次更新记录一次，避免过于频繁）
                if self.fatigue_stats and not hasattr(self, '_fatigue_record_counter'):
                    self._fatigue_record_counter = 0

                if self.fatigue_stats:
                    self._fatigue_record_counter += 1
                    if self._fatigue_record_counter >= 10:  # 每10次记录一次
                        try:
                            fatigue_level_map = {
                                '正常': 0,
                                '轻微疲劳': 1,
                                '中度疲劳': 2,
                                '重度疲劳': 3
                            }
                            fatigue_level = fatigue_level_map.get(res[1], 0)
                            attention_level = max(0.0, 1.0 - fatigue_level * 0.25)

                            self.fatigue_stats.record_fatigue_state(
                                fatigue_level=fatigue_level,
                                drowsiness_prob=fatigue_level * 0.25,
                                attention_level=attention_level,
                                confidence=0.8
                            )
                            self._fatigue_record_counter = 0

                            # 如果检测到疲劳，额外记录疲劳事件
                            if fatigue_level > 0:
                                from fatigue_statistics import FatigueEvent
                                event = FatigueEvent(
                                    timestamp=current_time,
                                    event_type='fatigue_state',
                                    value=float(fatigue_level),
                                    confidence=0.8,
                                    additional_data={'source': 'traditional', 'level': res[1]}
                                )
                                self.fatigue_stats.record_event(event)
                                print(f"📊 记录疲劳状态到数据库: {res[1]}")
                        except Exception as e:
                            print(f"记录疲劳状态失败: {e}")

                # 特殊情况检测：长时间闭眼或头部倾斜
                if res[3] in ['轻微倾斜', '明显倾斜'] and res[5] == '闭眼':
                    if (current_time - h_time).total_seconds() >= self.values[3]:
                        res[1] = '重度疲劳'
                else:
                    h_time = current_time

                if res[5] == '闭眼':
                    if (current_time - e_time).total_seconds() >= self.values[1]:
                        res[1] = '重度疲劳'
                else:
                    e_time = current_time

                # 每30分钟重置计数器（延长重置间隔，避免频繁清零）
                reset_interval = 1800  # 30分钟 = 1800秒
                if (current_time - t_time).total_seconds() >= reset_interval:
                    print(f"📊 重置计数器 - 眨眼: {self.TOTAL}, 哈欠: {self.mTOTAL}, 点头: {self.hTOTAL}")
                    # 保存当前统计到历史记录
                    if self.fatigue_stats:
                        try:
                            self.fatigue_stats.save_session_summary()
                            print("📊 会话统计已保存")
                        except Exception as e:
                            print(f"保存会话统计失败: {e}")

                    # 重置计数器
                    self.TOTAL = 0
                    self.mTOTAL = 0
                    self.hTOTAL = 0
                    t_time = current_time

                # 没有检测到人脸的处理
                if len(faces) == 0:
                    res[0] = '未识别到'
                    # 没有检测到人脸
                    self.oCOUNTER += 1

                    # 添加更友好的提示信息
                    cv2.putText(im_rd, "No Face Detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(im_rd, "Please face the camera", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

                    # 减少"没有识别到人脸"消息的频率
                    if self.oCOUNTER >= self.OUT_AR_CONSEC_FRAMES_check * 2:  # 增加阈值
                        self.safe_emit_signal({'type': 'msg', 'value': time.strftime('%Y-%m-%d %H:%M ',
                                                                                           time.localtime()) + u"没有识别到人脸，请调整位置"})
                        self.oCOUNTER = 0

                    # 不要重置计数器，保持历史数据
                    # self.TOTAL = 0
                    # self.mTOTAL = 0
                    # self.hTOTAL = 0

            # 确定疲劳提示:眨眼50次，打哈欠15次，瞌睡点头30次
            # if self.TOTAL >= 50 or self.mTOTAL >= 15 or self.hTOTAL >= 30:
            #     cv2.putText(im_rd, "SLEEP!!!", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            #     self.m_textCtrl3.AppendText(u"疲劳")

                # 优化的图像显示处理 - 减少卡顿
                try:
                    # 验证图像有效性
                    if im_rd is None or im_rd.size == 0:
                        if frame_count % 60 == 0:  # 减少错误信息频率
                            print("图像数据无效，跳过显示")
                        continue

                    height, width = im_rd.shape[:2]

                    # 确保图像有效
                    if height > 0 and width > 0:
                        # 转换颜色空间从BGR到RGB
                        RGBImg = cv2.cvtColor(im_rd, cv2.COLOR_BGR2RGB)

                        # 验证转换后的图像
                        if RGBImg is None or RGBImg.size == 0:
                            if frame_count % 60 == 0:  # 减少错误信息频率
                                print("RGB转换失败")
                            continue

                        # 减少调试信息的频率，避免控制台刷屏
                        if self.frame_count % 300 == 0:  # 每300帧打印一次（约每10秒）
                            print(f"发送图像数据: {width}x{height}, 人脸数: {len(faces)}")

                        # 发送图像数据到UI线程 - 确保视频流畅显示
                        # 每帧都发送图像数据以保证视频连续性
                        data = {'type':'img','value':RGBImg}
                        self.safe_emit_signal(data)
                    else:
                        if frame_count % 60 == 0:  # 减少错误信息频率
                            print("图像尺寸无效，跳过显示")

                except Exception as e:
                    if frame_count % 60 == 0:  # 减少错误信息频率
                        print(f"图像显示处理失败: {e}")
                    # 不要continue，让程序继续处理其他数据
                    pass

                end_time = datetime.datetime.now()

                # 计算实际FPS
                self.frame_count += 1
                current_time_float = time.time()
                if current_time_float - self.fps_start_time >= 3.0:  # 每3秒更新一次
                    self.actual_fps = self.frame_count / (current_time_float - self.fps_start_time)
                    print(f"实际FPS: {self.actual_fps:.1f}")
                    self.frame_count = 0
                    self.fps_start_time = current_time_float

                # 帧数 - 使用实际FPS或计算的瞬时FPS
                frame_time_ms = (end_time - start_time).total_seconds() * 1000
                if frame_time_ms > 0:
                    instant_fps = 1000 / frame_time_ms
                    res[2] = str(int(instant_fps))
                else:
                    res[2] = str(int(self.actual_fps)) if hasattr(self, 'actual_fps') else "0"

                data = {'type': 'res', 'value': res}
                self.safe_emit_signal(data)

            except Exception as e:
                print(f"主循环异常: {e}")
                error_count += 1
                if error_count > 100:
                    print("错误过多，停止检测")
                    break
                time.sleep(0.1)  # 异常后稍作等待
                continue

        # 释放摄像头
        if hasattr(self, 'cap') and self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                print(f"摄像头释放失败: {e}")
            finally:
                self.cap = None  # 无论如何都设为None

    def value_changed(self):
        try:
            if hasattr(self, 'spinBox_1'):
                self.values[0] = self.spinBox_1.value()
            if hasattr(self, 'spinBox_2'):
                self.values[1] = self.spinBox_2.value()
            if hasattr(self, 'spinBox_3'):
                self.values[2] = self.spinBox_3.value()
            if hasattr(self, 'spinBox_4'):
                self.values[3] = self.spinBox_4.value()
            if hasattr(self, 'spinBox_5'):
                self.values[4] = self.spinBox_5.value()
        except AttributeError:
            pass

    def select_changed(self):
        try:
            if hasattr(self, 'checkBox_11'):
                self.fun[0] = self.checkBox_11.isChecked()
            if hasattr(self, 'checkBox_12'):
                self.fun[1] = self.checkBox_12.isChecked()
            if hasattr(self, 'checkBox_21'):
                self.fun[2] = self.checkBox_21.isChecked()
            if hasattr(self, 'checkBox_22'):
                self.fun[3] = self.checkBox_22.isChecked()
            if hasattr(self, 'checkBox_31'):
                self.fun[4] = self.checkBox_31.isChecked()
            if hasattr(self, 'checkBox_32'):
                self.fun[5] = self.checkBox_32.isChecked()
        except AttributeError:
            pass
        pass

    def button_clicked(self):
        """处理按钮点击事件 - 切换开始/停止检测"""
        if self.thread is not None and self.thread.is_alive():
            # 当前正在检测，点击停止
            self.stop_detection()
            if hasattr(self, 'plainTextEdit_tip'):
                self.plainTextEdit_tip.appendPlainText('检测已停止')
        else:
            # 当前未检测，点击开始
            self.start_detection()
            if hasattr(self, 'plainTextEdit_tip'):
                self.plainTextEdit_tip.appendPlainText('开始检测')

    def start_detection(self):
        """开始检测"""
        # 启动检测
        self.is_running = True
        self.thread = threading.Thread(target=self._learning_face, daemon=True)
        self.thread.start()

        # 更新按钮状态为检测中
        self.pushButton.setText("⏹️ 停止检测")
        self.pushButton.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FF9800, stop:1 #F57C00);
                border: none;
                border-radius: 10px;
                color: white;
                font-weight: bold;
                padding: 15px 30px;
                font-size: 16pt;
                min-height: 40px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FFB74D, stop:1 #FF9800);
            }
        """)



    def stop_detection(self):
        """停止检测"""
        print("正在停止检测...")

        # 首先设置停止标志
        self.is_running = False

        # 结束统计会话
        if hasattr(self, 'fatigue_stats') and self.fatigue_stats:
            try:
                self.fatigue_stats.end_session()
                print("✅ 疲劳统计会话已结束")
            except Exception as e:
                print(f"结束统计会话失败: {e}")

        # 重置错误日志标志，允许下次启动时重新记录
        if hasattr(self, '_signal_error_logged'):
            delattr(self, '_signal_error_logged')
        if hasattr(self, '_runtime_error_logged'):
            delattr(self, '_runtime_error_logged')
        if hasattr(self, '_unknown_error_logged'):
            delattr(self, '_unknown_error_logged')

        # 关闭摄像头
        if hasattr(self, 'cap') and self.cap is not None:
            try:
                self.cap.release()
                print("摄像头已释放")
            except Exception as e:
                print(f"摄像头释放失败: {e}")
            finally:
                self.cap = None
                self.CAMERA_STYLE = False

        # 等待线程结束
        if self.thread is not None and self.thread.is_alive():
            try:
                print("等待检测线程停止...")
                self.thread.join(timeout=3)  # 等待最多3秒
                if self.thread.is_alive():
                    print("⚠️ 检测线程未能在3秒内停止，但系统已设置停止标志")
                else:
                    print("✅ 检测线程已正常停止")
            except Exception as e:
                print(f"等待线程停止时出错: {e}")

        # 重置按钮状态为开始检测
        self.pushButton.setText("🚀 开始检测")
        self.pushButton.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4CAF50, stop:1 #45a049);
                border: none;
                border-radius: 10px;
                color: white;
                font-weight: bold;
                padding: 15px 30px;
                font-size: 16pt;
                min-height: 40px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5CBF60, stop:1 #4CAF50);
            }
        """)

        # 清空视频显示
        if hasattr(self, 'label_img'):
            self.label_img.clear()
            self.label_img.setText("等待视频流...")
            self.label_img.setAlignment(Qt.AlignCenter)

        print("✅ 检测已停止")

    def thread_sound(self):
        """播放警报声音"""
        try:
            # 尝试播放音频文件
            audio_file = '1.mp3'
            if os.path.exists(audio_file):
                pygame.mixer.init()
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                time.sleep(15)
                pygame.mixer.music.stop()
            else:
                # 如果音频文件不存在，使用系统提示音
                self.play_system_beep()
        except Exception as e:
            print(f"音频播放失败: {e}")
            # 降级到系统提示音
            self.play_system_beep()

    def play_system_beep(self):
        """播放系统提示音作为替代"""
        try:
            # 在Windows上使用系统提示音
            import winsound
            # 播放系统警告声
            for _ in range(3):  # 播放3次
                winsound.Beep(1000, 500)  # 1000Hz, 500ms
                time.sleep(0.2)
        except ImportError:
            # 如果winsound不可用，打印警告
            print("⚠️ 疲劳警报：请注意休息！")
        except Exception as e:
            print(f"系统提示音播放失败: {e}")
            print("⚠️ 疲劳警报：请注意休息！")

    def paly_sound(self):
        if self.sound_thread is not None and self.sound_thread.is_alive():
            # self.plainTextEdit_tip('播放声音中')
            pass
        else:
            self.plainTextEdit_tip.appendPlainText('疲劳驾驶 播放声音')
            self.sound_thread = threading.Thread(target=self.thread_sound,daemon=True)
            self.sound_thread.start()
        pass

    def thread_singnal_slot(self, d):
        if d['type']=='img':
            try:
                RGBImg = d['value']

                # 减少调试输出频率
                if not hasattr(self, '_img_debug_counter'):
                    self._img_debug_counter = 0
                self._img_debug_counter += 1

                if self._img_debug_counter % 120 == 0:  # 每120帧打印一次
                    print(f"接收到图像信号: {type(RGBImg)}")

                # 验证图像数据
                if RGBImg is None or RGBImg.size == 0:
                    if self._img_debug_counter % 30 == 0:  # 错误信息减少频率
                        print("接收到无效图像数据")
                    return

                # 确保图像是连续的内存布局
                if not RGBImg.flags['C_CONTIGUOUS']:
                    RGBImg = np.ascontiguousarray(RGBImg)

                height, width, _ = RGBImg.shape
                bytes_per_line = 3 * width

                if self._img_debug_counter % 120 == 0:  # 每120帧打印一次
                    print(f"处理图像: {width}x{height}")

                # 将图片转化成Qt可读格式
                qimage = QImage(RGBImg.data, width, height, bytes_per_line, QImage.Format_RGB888)

                if qimage.isNull():
                    print("QImage创建失败")
                    return

                piximage = QtGui.QPixmap.fromImage(qimage)

                if piximage.isNull():
                    print("QPixmap创建失败")
                    return

                # 检查label_img是否存在
                if not hasattr(self, 'label_img') or self.label_img is None:
                    if self._img_debug_counter % 60 == 0:  # 减少错误信息频率
                        print("label_img不存在，无法显示图像")
                    return

                if self._img_debug_counter % 120 == 0:  # 减少调试信息频率
                    print(f"label_img存在，尺寸: {self.label_img.size()}")

                # 直接设置图像到标签，让setScaledContents处理缩放
                self.label_img.setPixmap(piximage)
                if self._img_debug_counter % 120 == 0:  # 减少调试信息频率
                    print(f"图像已设置到label_img: {width}x{height}")

            except Exception as e:
                print(f"图像显示失败: {e}")
                import traceback
                traceback.print_exc()
            self.label_img.setAlignment(Qt.AlignCenter)
        elif d['type'] == 'msg':
            if hasattr(self, 'plainTextEdit_tip'):
                self.plainTextEdit_tip.appendPlainText(d['value'])
            else:
                print(f"系统消息: {d['value']}")
        elif d['type'] == 'res':
            # 兼容新旧界面的标签更新
            try:
                if hasattr(self, 'label_11'):
                    self.label_11.setText(d['value'][0])
                if hasattr(self, 'label_12'):
                    self.label_12.setText(d['value'][1])
                if hasattr(self, 'label_13'):
                    self.label_13.setText(d['value'][2])
                if hasattr(self, 'label_21'):
                    self.label_21.setText(d['value'][3])
                if hasattr(self, 'label_22'):
                    self.label_22.setText(d['value'][4])
                if hasattr(self, 'label_23'):
                    self.label_23.setText(d['value'][5])
                if hasattr(self, 'label_31') and len(d['value']) > 6:
                    self.label_31.setText(d['value'][6])
                if hasattr(self, 'label_32') and len(d['value']) > 7:
                    self.label_32.setText(d['value'][7])
                if hasattr(self, 'label_33') and len(d['value']) > 8:
                    self.label_33.setText(d['value'][8])
            except (IndexError, AttributeError) as e:
                print(f"标签更新警告: {e}")
            # 设置疲劳状态颜色
            try:
                if len(d['value']) > 1 and hasattr(self, 'label_12'):
                    if d['value'][1] == '轻微疲劳':
                        self.label_12.setStyleSheet("color:orange;")
                    elif d['value'][1] == '中度疲劳':
                        self.label_12.setStyleSheet("color:yellow;")
                    elif d['value'][1] == '重度疲劳':
                        self.label_12.setStyleSheet("color:red;")
                        self.paly_sound()
                    else:
                        self.label_12.setStyleSheet("color:green;")
            except (IndexError, AttributeError):
                pass

        pass

    def close(self) -> bool:
        # 设置运行状态为False，停止线程
        self.is_running = False

        # 等待线程结束
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2)  # 等待最多2秒

        # 释放OpenCV摄像头
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            print("摄像头已释放")

        return super(MainUI, self).close()

    def closeEvent(self, event):
        """窗口关闭事件"""
        self.is_running = False

        # 等待线程结束
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2)

        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()

        event.accept()



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainUI()
    window.show()
    sys.exit(app.exec())