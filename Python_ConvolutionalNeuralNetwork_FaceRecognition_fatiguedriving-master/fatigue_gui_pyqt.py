"""
疲劳检测GUI界面 - PyQt版本
保持原有的检测逻辑，只更换前端界面技术从Tkinter到PyQt
"""
import cv2
import torch
import numpy as np
import dlib
from collections import deque
import time
import threading
import os
import sys

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QGridLayout, QLabel, QPushButton, 
                            QProgressBar, QTextEdit, QGroupBox, QFrame,
                            QScrollArea, QSplitter, QMessageBox, QSlider)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QColor

from config import *
from model import create_model
from utils import extract_face_landmarks, normalize_landmarks

class VideoThread(QThread):
    """视频处理线程 - 保持原有检测逻辑"""
    frame_ready = pyqtSignal(np.ndarray, bool, float, int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.running = False
        
    def run(self):
        """检测循环 - 完全保持原有逻辑"""
        while self.running:
            try:
                ret, frame = self.parent.cap.read()
                if not ret:
                    print("❌ 无法读取摄像头画面")
                    break

                # 转换为RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 预处理（保持原有逻辑）
                face_img, landmarks_norm, face_rect = self.parent._preprocess_frame(frame)
                face_detected = face_img is not None

                # 获取原始landmarks用于MAR/EAR计算
                if face_detected:
                    original_face_img, original_landmarks = extract_face_landmarks(frame, self.parent.detector, self.parent.predictor)
                
                yawn_prob = 0.0
                prediction = 0
                
                if face_detected:
                    self.parent.face_buffer.append(face_img)
                    self.parent.landmark_buffer.append(landmarks_norm)

                    # 如果缓冲区满了，进行预测（保持原有逻辑）
                    if len(self.parent.face_buffer) >= SEQUENCE_LENGTH:
                        yawn_prob, model_prediction = self.parent._predict_yawn()
                        self.parent.total_predictions += 1

                        # 计算当前帧的嘴部长宽比和眼部长宽比（使用原始landmarks）
                        current_mar = self.parent._calculate_mouth_aspect_ratio(original_landmarks)
                        current_ear = self.parent._calculate_eye_aspect_ratio(original_landmarks)

                        # 检测眨眼（保持原有逻辑）
                        blink_detected = self.parent._detect_blink(current_ear)

                        # 新的检测逻辑：模型预测 + MAR阈值的组合判断（保持原有逻辑）
                        model_says_yawn = yawn_prob > self.parent.yawn_threshold.get()
                        mar_says_yawn = current_mar > self.parent.mar_threshold.get()

                        # 最终判断：两个条件都满足才认为是打哈欠（保持原有逻辑）
                        final_prediction = 1 if (model_says_yawn and mar_says_yawn) else 0

                        # 更新连续检测计数 - 使用平滑衰减机制（保持原有逻辑）
                        current_time = time.time()
                        if final_prediction == 1:
                            # 检测到打哈欠：增加计数，更新最后检测时间
                            self.parent.consecutive_yawns += 1
                            self.parent.last_detection_time = current_time
                            self.parent.no_detection_frames = 0  # 重置未检测帧数
                            print(f"🔍 打哈欠检测: 模型={yawn_prob:.3f}({'✓' if model_says_yawn else '✗'}), MAR={current_mar:.3f}({'✓' if mar_says_yawn else '✗'}), 连续={self.parent.consecutive_yawns}")
                        else:
                            # 未检测到打哈欠：使用平滑衰减
                            self.parent.no_detection_frames += 1
                            
                            # 如果有之前的检测记录，则开始衰减
                            if self.parent.consecutive_yawns > 0:
                                # 计算衰减量：基于时间的衰减
                                if self.parent.last_detection_time > 0:
                                    time_since_last = current_time - self.parent.last_detection_time
                                    # 每秒衰减decay_rate帧，但至少保持1秒不衰减
                                    if time_since_last > 1.0:  # 1秒后开始衰减
                                        decay_amount = int((time_since_last - 1.0) * self.parent.decay_rate)
                                        self.parent.consecutive_yawns = max(0, self.parent.consecutive_yawns - decay_amount)
                                        
                                        if self.parent.consecutive_yawns == 0:
                                            print(f"📉 进度条衰减至零（未检测{self.parent.no_detection_frames}帧，时间间隔{time_since_last:.1f}秒）")
                                        else:
                                            print(f"📉 进度条衰减: {self.parent.consecutive_yawns}（未检测{self.parent.no_detection_frames}帧）")
                                else:
                                    # 如果没有时间记录，立即开始衰减
                                    if self.parent.no_detection_frames > 30:  # 30帧后开始衰减（约1秒）
                                        self.parent.consecutive_yawns = max(0, self.parent.consecutive_yawns - 1)
                            else:
                                # 如果consecutive_yawns已经是0，保持为0
                                self.parent.consecutive_yawns = 0

                        # 检查是否触发警报（保持原有逻辑）
                        if (self.parent.consecutive_yawns >= self.parent.consecutive_threshold and
                            (current_time - self.parent.last_yawn_time) > self.parent.alert_cooldown.get()):
                            self.parent.yawn_count += 1
                            self.parent.last_yawn_time = current_time
                            self.parent.recent_yawns.append(current_time)
                            print(f"🚨 触发警报！连续{self.parent.consecutive_yawns}帧检测到打哈欠")

                        # 更新prediction变量用于GUI显示
                        prediction = final_prediction

                    # 在人脸上绘制特征点和人脸框（使用归一化的landmarks）
                    frame = self.parent._draw_face_landmarks(frame, face_rect, landmarks_norm)
                else:
                    # 未检测到人脸时的衰减逻辑（保持原有逻辑）
                    if self.parent.consecutive_yawns > 0:
                        current_time = time.time()
                        self.parent.no_detection_frames += 1
                        
                        # 如果有之前的检测记录，则开始衰减
                        if self.parent.last_detection_time > 0:
                            time_since_last = current_time - self.parent.last_detection_time
                            # 未检测到人脸时，衰减更快一些
                            if time_since_last > 0.5:  # 0.5秒后开始衰减
                                decay_amount = int((time_since_last - 0.5) * self.parent.decay_rate * 1.5)  # 衰减速度1.5倍
                                old_consecutive = self.parent.consecutive_yawns
                                self.parent.consecutive_yawns = max(0, self.parent.consecutive_yawns - decay_amount)
                                
                                if old_consecutive != self.parent.consecutive_yawns:
                                    if self.parent.consecutive_yawns == 0:
                                        print(f"📉 未检测到人脸，进度条衰减至零（未检测{self.parent.no_detection_frames}帧）")
                                    else:
                                        print(f"📉 未检测到人脸，进度条衰减: {self.parent.consecutive_yawns}")
                        else:
                            # 如果没有时间记录，较快衰减
                            if self.parent.no_detection_frames > 15:  # 15帧后开始衰减（约0.5秒）
                                self.parent.consecutive_yawns = max(0, self.parent.consecutive_yawns - 1)

                # 保存状态变量供GUI更新使用
                self.parent._last_face_detected = face_detected
                self.parent._last_yawn_prob = yawn_prob
                self.parent._last_prediction = prediction
                if face_detected and len(self.parent.face_buffer) >= SEQUENCE_LENGTH:
                    # 保存MAR和EAR用于GUI显示（使用原始landmarks）
                    self.parent._last_mar = self.parent._calculate_mouth_aspect_ratio(original_landmarks)
                    self.parent._last_ear = self.parent._calculate_eye_aspect_ratio(original_landmarks)

                # 发送信号更新GUI
                self.frame_ready.emit(frame, face_detected, yawn_prob, prediction)
                
                self.msleep(33)  # 约30fps
                
            except Exception as e:
                print(f"❌ 检测循环错误: {e}")
                break

class FatigueDetectionGUI(QMainWindow):
    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化主窗口
        self.setWindowTitle("疲劳驾驶检测系统 - PyQt版本")
        self.setGeometry(100, 100, 1400, 900)
        
        # 加载模型（保持原有逻辑）
        self.model = self._load_model()
        
        # 初始化dlib（保持原有逻辑）
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(DLIB_PREDICTOR_PATH)
        
        # 摄像头
        self.cap = None
        self.video_thread = None
        self.camera_index = self._detect_available_camera()
        
        # 检测参数（保持原有逻辑，使用简单的类来模拟tkinter变量）
        class SimpleVar:
            def __init__(self, value):
                self._value = value
            def get(self):
                return self._value
            def set(self, value):
                self._value = value

        self.yawn_threshold = SimpleVar(0.6)
        self.mar_threshold = SimpleVar(0.6)
        self.consecutive_threshold = 30
        self.alert_cooldown = SimpleVar(5.0)
        self.current_mode = "balanced"
        
        # 缓冲区（保持原有逻辑）
        self.face_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.landmark_buffer = deque(maxlen=SEQUENCE_LENGTH)
        
        # 状态变量（保持原有逻辑）
        self.yawn_count = 0
        self.blink_count = 0
        self.total_predictions = 0
        self.consecutive_yawns = 0
        self.session_start_time = None
        self.last_yawn_time = 0
        
        # 进度条平滑控制变量（保持原有逻辑）
        self.last_detection_time = 0
        self.decay_rate = 2.0
        self.no_detection_frames = 0
        
        # 疲劳状态评估相关（保持原有逻辑）
        self.recent_yawns = []
        self.recent_blinks = []
        self.fatigue_window = 60
        self.last_blink_time = 0
        self.eye_closed_frames = 0
        self.eye_closed_threshold = 10
        self.long_eye_closed_threshold = 90
        self.eye_closed_start_time = None
        
        # 创建GUI
        self._create_gui()
        
        # 设置样式
        self._set_style()
        
        # 定时器更新GUI
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_gui)
        self.timer.start(100)  # 100ms更新一次
        
        # 应用默认的平衡模式
        self._apply_preset('balanced')

    def _detect_available_camera(self):
        """自动检测可用的摄像头（保持原有逻辑）"""
        print("🔍 正在检测可用摄像头...")
        
        for index in range(6):
            try:
                cap = cv2.VideoCapture(index)
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret and frame is not None:
                        print(f"✅ 检测到可用摄像头: 索引 {index}")
                        return index
                else:
                    cap.release()
            except Exception as e:
                continue
        
        print("❌ 未检测到可用摄像头，使用默认索引 0")
        return 0

    def _load_model(self):
        """加载模型（保持原有逻辑）"""
        try:
            model = create_model().to(self.device)
            checkpoint = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print("✅ 模型加载成功")
            return model
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            QMessageBox.critical(self, "错误", f"模型加载失败: {e}")
            return None

    def _create_gui(self):
        """创建GUI界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)

        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # 左侧面板
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)

        # 右侧面板
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)

        # 设置分割比例
        splitter.setSizes([800, 600])

    def _create_left_panel(self):
        """创建左侧面板"""
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # 视频显示区域
        video_group = QGroupBox("视频预览")
        video_group.setFont(QFont("Arial", 12, QFont.Bold))
        video_layout = QVBoxLayout(video_group)

        self.video_label = QLabel("等待摄像头启动...")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #333333;
                color: white;
                font-size: 16px;
                border: 2px solid #555555;
            }
        """)
        self.video_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.video_label)

        left_layout.addWidget(video_group)

        # 控制按钮
        control_group = QGroupBox("控制面板")
        control_group.setFont(QFont("Arial", 10, QFont.Bold))
        control_layout = QHBoxLayout(control_group)

        self.start_btn = QPushButton("🚀 开始检测")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 12px;
                font-weight: bold;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.start_btn.clicked.connect(self._start_detection)

        self.stop_btn = QPushButton("⏹️ 停止检测")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-size: 12px;
                font-weight: bold;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.stop_btn.clicked.connect(self._stop_detection)
        self.stop_btn.setEnabled(False)

        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addStretch()

        left_layout.addWidget(control_group)

        # 当前设置
        settings_group = QGroupBox("当前设置")
        settings_group.setFont(QFont("Arial", 10, QFont.Bold))
        settings_layout = QVBoxLayout(settings_group)

        self.current_mode_label = QLabel("当前模式: ⚖️ 平衡模式")
        self.current_mode_label.setFont(QFont("Arial", 11, QFont.Bold))
        self.current_mode_label.setStyleSheet("color: #4CAF50;")

        self.current_params_label = QLabel("模型阈值: 0.60 | MAR阈值: 0.60 | 连续阈值: 30帧 | 冷却: 5.0秒")
        self.current_params_label.setFont(QFont("Arial", 9))
        self.current_params_label.setStyleSheet("color: #666666;")

        settings_layout.addWidget(self.current_mode_label)
        settings_layout.addWidget(self.current_params_label)

        left_layout.addWidget(settings_group)

        # 快速预设
        preset_group = QGroupBox("快速预设")
        preset_group.setFont(QFont("Arial", 10, QFont.Bold))
        preset_layout = QHBoxLayout(preset_group)

        # 预设按钮
        sensitive_btn = QPushButton("🔥 敏感模式")
        sensitive_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF5722;
                color: white;
                font-weight: bold;
                padding: 8px 15px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #E64A19;
            }
        """)
        sensitive_btn.clicked.connect(lambda: self._apply_preset('sensitive'))

        balanced_btn = QPushButton("⚖️ 平衡模式")
        balanced_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px 15px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        balanced_btn.clicked.connect(lambda: self._apply_preset('balanced'))

        conservative_btn = QPushButton("🛡️ 保守模式")
        conservative_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 8px 15px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        conservative_btn.clicked.connect(lambda: self._apply_preset('conservative'))

        preset_layout.addWidget(sensitive_btn)
        preset_layout.addWidget(balanced_btn)
        preset_layout.addWidget(conservative_btn)

        left_layout.addWidget(preset_group)

        return left_widget

    def _create_right_panel(self):
        """创建右侧面板"""
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # 实时监测区域（3x2方格布局）
        monitor_group = QGroupBox("实时监测")
        monitor_group.setFont(QFont("Arial", 12, QFont.Bold))
        monitor_layout = QVBoxLayout(monitor_group)

        # 创建3x2方格布局
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(10)

        # 人脸检测状态
        self.face_status_widget = self._create_status_card("人脸检测", "等待中", "#ffffff", "#333333")
        grid_layout.addWidget(self.face_status_widget, 0, 0)

        # 打哈欠概率
        self.prob_status_widget = self._create_status_card("打哈欠概率", "0.000", "#e8f5e8", "#2e7d32")
        grid_layout.addWidget(self.prob_status_widget, 0, 1)

        # 嘴部状态
        self.mouth_status_widget = self._create_status_card("嘴部状态", "正常", "#fff3e0", "#e65100")
        grid_layout.addWidget(self.mouth_status_widget, 0, 2)

        # 眼部状态
        self.eye_status_widget = self._create_status_card("眼部状态", "正常", "#e3f2fd", "#1565c0")
        grid_layout.addWidget(self.eye_status_widget, 1, 0)

        # 疲劳状态
        self.fatigue_status_widget = self._create_status_card("疲劳状态", "正常", "#fce4ec", "#ad1457")
        grid_layout.addWidget(self.fatigue_status_widget, 1, 1)

        # 连续检测
        self.consecutive_status_widget = self._create_status_card("连续检测", "0/30", "#f3e5f5", "#6a1b9a")
        grid_layout.addWidget(self.consecutive_status_widget, 1, 2)

        monitor_layout.addWidget(grid_widget)

        # 进度条
        progress_widget = QWidget()
        progress_layout = QHBoxLayout(progress_widget)
        progress_layout.addWidget(QLabel("检测进度:"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #cccccc;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)
        monitor_layout.addWidget(progress_widget)

        right_layout.addWidget(monitor_group)

        # 会话统计区域
        stats_group = QGroupBox("会话统计")
        stats_group.setFont(QFont("Arial", 12, QFont.Bold))
        stats_layout = QVBoxLayout(stats_group)

        # 创建2x3统计方格布局
        stats_grid_widget = QWidget()
        stats_grid_layout = QGridLayout(stats_grid_widget)
        stats_grid_layout.setSpacing(10)

        # 会话时间
        self.time_status_widget = self._create_status_card("会话时间", "00:00", "#e3f2fd", "#1976d2")
        stats_grid_layout.addWidget(self.time_status_widget, 0, 0)

        # 缓冲区
        self.buffer_status_widget = self._create_status_card("缓冲区", "0/30", "#f3e5f5", "#7b1fa2")
        stats_grid_layout.addWidget(self.buffer_status_widget, 0, 1)

        # 总检测次数
        self.count_status_widget = self._create_status_card("总检测", "0", "#e8f5e8", "#388e3c")
        stats_grid_layout.addWidget(self.count_status_widget, 1, 0)

        # 打哈欠次数
        self.yawn_status_widget = self._create_status_card("打哈欠", "0", "#fff3e0", "#f57c00")
        stats_grid_layout.addWidget(self.yawn_status_widget, 1, 1)

        # 眨眼次数（跨两列）
        self.blink_status_widget = self._create_status_card("眨眼次数", "0", "#fce4ec", "#c2185b")
        stats_grid_layout.addWidget(self.blink_status_widget, 2, 0, 1, 2)

        stats_layout.addWidget(stats_grid_widget)
        right_layout.addWidget(stats_group)

        # 警报历史
        alert_group = QGroupBox("警报历史")
        alert_group.setFont(QFont("Arial", 12, QFont.Bold))
        alert_layout = QVBoxLayout(alert_group)

        self.alert_text = QTextEdit()
        self.alert_text.setMaximumHeight(150)
        self.alert_text.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                border: 2px solid #cccccc;
                border-radius: 5px;
                font-family: 'Consolas', monospace;
                font-size: 10px;
            }
        """)
        self.alert_text.append("系统启动，等待开始检测...")
        alert_layout.addWidget(self.alert_text)

        right_layout.addWidget(alert_group)

        # 设置滚动区域
        scroll_area.setWidget(right_widget)
        return scroll_area

    def _create_status_card(self, title, value, bg_color, text_color):
        """创建状态卡片"""
        card = QFrame()
        card.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        card.setLineWidth(2)
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color};
                border: 2px solid #cccccc;
                border-radius: 8px;
                padding: 5px;
            }}
        """)

        layout = QVBoxLayout(card)
        layout.setAlignment(Qt.AlignCenter)

        title_label = QLabel(title)
        title_label.setFont(QFont("Arial", 10, QFont.Bold))
        title_label.setStyleSheet(f"color: {text_color}; margin-bottom: 5px;")
        title_label.setAlignment(Qt.AlignCenter)

        value_label = QLabel(value)
        value_label.setFont(QFont("Arial", 12, QFont.Bold))
        value_label.setStyleSheet(f"color: {text_color};")
        value_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(title_label)
        layout.addWidget(value_label)

        # 保存value_label的引用以便更新
        card.value_label = value_label

        return card

    def _set_style(self):
        """设置整体样式"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)

    # 以下方法完全保持原有检测逻辑不变
    def _preprocess_frame(self, frame):
        """预处理帧（保持原有逻辑）"""
        face_img, landmarks = extract_face_landmarks(frame, self.detector, self.predictor)

        if face_img is None or landmarks is None:
            return None, None, None

        face_resized = cv2.resize(face_img, FACE_SIZE)
        landmarks_norm = normalize_landmarks(landmarks, face_img.shape[:2])

        # 获取人脸区域
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        face_rect = faces[0] if len(faces) > 0 else None

        return face_resized, landmarks_norm, face_rect

    def _predict_yawn(self):
        """预测打哈欠（保持原有逻辑）"""
        if len(self.face_buffer) < SEQUENCE_LENGTH:
            return 0.0, 0

        faces = np.array(list(self.face_buffer))
        landmarks = np.array(list(self.landmark_buffer))

        faces_tensor = torch.from_numpy(faces).float().unsqueeze(0)
        landmarks_tensor = torch.from_numpy(landmarks).float().unsqueeze(0)

        faces_tensor = faces_tensor.permute(0, 1, 4, 2, 3)
        landmarks_tensor = landmarks_tensor.reshape(1, SEQUENCE_LENGTH, -1)
        faces_tensor = faces_tensor / 255.0

        faces_tensor = faces_tensor.to(self.device)
        landmarks_tensor = landmarks_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(faces_tensor, landmarks_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            yawn_prob = probabilities[0, 1].item()
            prediction = 1 if yawn_prob > self.yawn_threshold.get() else 0

        return yawn_prob, prediction

    def _calculate_mouth_aspect_ratio(self, landmarks):
        """计算嘴部长宽比(MAR)（保持原有逻辑）"""
        try:
            mouth_points = landmarks[48:68]
            A = np.linalg.norm(mouth_points[13] - mouth_points[19])
            B = np.linalg.norm(mouth_points[14] - mouth_points[18])
            C = np.linalg.norm(mouth_points[15] - mouth_points[17])
            D = np.linalg.norm(mouth_points[0] - mouth_points[6])
            mar = (A + B + C) / (3.0 * D)
            return mar
        except:
            return 0.0

    def _calculate_eye_aspect_ratio(self, landmarks):
        """计算眼部长宽比(EAR)（保持原有逻辑）"""
        try:
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]

            def eye_aspect_ratio(eye_points):
                A = np.linalg.norm(eye_points[1] - eye_points[5])
                B = np.linalg.norm(eye_points[2] - eye_points[4])
                C = np.linalg.norm(eye_points[0] - eye_points[3])
                ear = (A + B) / (2.0 * C)
                return ear

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            return avg_ear
        except:
            return 0.3

    def _detect_blink(self, ear):
        """检测眨眼和长时间闭眼（保持原有逻辑）"""
        ear_threshold = 0.25
        current_time = time.time()

        if ear < ear_threshold:
            if self.eye_closed_frames == 0:
                self.eye_closed_start_time = current_time
            self.eye_closed_frames += 1
        else:
            if self.eye_closed_frames >= self.eye_closed_threshold:
                if current_time - self.last_blink_time > 0.3:
                    self.blink_count += 1
                    self.last_blink_time = current_time
                    self.recent_blinks.append(current_time)
                    return True

            self.eye_closed_frames = 0
            self.eye_closed_start_time = None

        return False

    def _evaluate_fatigue_status(self):
        """评估疲劳状态（保持原有逻辑）"""
        current_time = time.time()

        self.recent_yawns = [t for t in self.recent_yawns if current_time - t <= self.fatigue_window]
        self.recent_blinks = [t for t in self.recent_blinks if current_time - t <= self.fatigue_window]

        yawn_count_1min = len(self.recent_yawns)
        long_eye_closed = self.eye_closed_frames >= self.long_eye_closed_threshold

        if yawn_count_1min >= 3 or long_eye_closed:
            return "重度疲劳"
        elif yawn_count_1min >= 2:
            return "轻度疲劳"
        else:
            return "正常"

    def _draw_face_landmarks(self, frame, face_rect, landmarks):
        """绘制人脸特征点（保持原有逻辑）"""
        if landmarks is not None and face_rect is not None:
            # 绘制人脸框
            cv2.rectangle(frame, (face_rect.left(), face_rect.top()),
                         (face_rect.right(), face_rect.bottom()), (255, 0, 0), 2)

            # 绘制特征点（与原版完全一致）
            for i, (x, y) in enumerate(landmarks):
                # 转换为实际坐标（与原版公式一致）
                actual_x = int(x * (face_rect.right() - face_rect.left()) + face_rect.left())
                actual_y = int(y * (face_rect.bottom() - face_rect.top()) + face_rect.top())

                # 不同区域使用不同颜色（与原版一致）
                if i < 17:  # 脸部轮廓 - 蓝色
                    color = (255, 0, 0)
                elif i < 27:  # 眉毛 - 绿色
                    color = (0, 255, 0)
                elif i < 36:  # 鼻子 - 黄色
                    color = (0, 255, 255)
                elif i < 48:  # 眼部 - 青色
                    color = (255, 255, 0)
                else:  # 嘴部 - 红色
                    color = (0, 0, 255)

                cv2.circle(frame, (actual_x, actual_y), 2, color, -1)

        return frame

    def _apply_preset(self, mode):
        """应用预设模式（保持原有逻辑）"""
        presets = {
            'sensitive': {
                'model_threshold': 0.5,
                'mar_threshold': 0.55,
                'consecutive_threshold': 20,
                'cooldown': 3.0,
                'name': '🔥 敏感模式',
                'color': '#FF5722'
            },
            'balanced': {
                'model_threshold': 0.6,
                'mar_threshold': 0.6,
                'consecutive_threshold': 30,
                'cooldown': 5.0,
                'name': '⚖️ 平衡模式',
                'color': '#4CAF50'
            },
            'conservative': {
                'model_threshold': 0.7,
                'mar_threshold': 0.7,
                'consecutive_threshold': 40,
                'cooldown': 7.0,
                'name': '🛡️ 保守模式',
                'color': '#2196F3'
            }
        }

        if mode in presets:
            preset = presets[mode]
            self.current_mode = mode

            # 更新参数
            self.yawn_threshold.set(preset['model_threshold'])
            self.mar_threshold.set(preset['mar_threshold'])
            self.consecutive_threshold = preset['consecutive_threshold']
            self.alert_cooldown.set(preset['cooldown'])

            # 更新界面显示
            self.current_mode_label.setText(f"当前模式: {preset['name']}")
            self.current_mode_label.setStyleSheet(f"color: {preset['color']};")
            self.current_params_label.setText(
                f"模型阈值: {preset['model_threshold']:.2f} | MAR阈值: {preset['mar_threshold']:.2f} | "
                f"连续阈值: {preset['consecutive_threshold']}帧 | 冷却: {preset['cooldown']:.1f}秒"
            )

            print(f"✅ 已应用{preset['name']}")

    def _start_detection(self):
        """开始检测（保持原有逻辑）"""
        if self.model is None:
            QMessageBox.critical(self, "错误", "模型未加载成功")
            return

        # 初始化摄像头
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "错误", f"无法打开摄像头 {self.camera_index}")
            return

        # 设置摄像头参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # 重置状态（保持原有逻辑）
        self.session_start_time = time.time()
        self.yawn_count = 0
        self.blink_count = 0
        self.total_predictions = 0
        self.consecutive_yawns = 0
        self.last_yawn_time = 0
        self.last_blink_time = 0
        self.eye_closed_frames = 0
        self.recent_yawns.clear()
        self.recent_blinks.clear()
        self.face_buffer.clear()
        self.landmark_buffer.clear()

        # 重置进度条平滑控制变量（保持原有逻辑）
        self.last_detection_time = 0
        self.no_detection_frames = 0

        # 更新按钮状态
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        # 启动视频线程
        self.video_thread = VideoThread(self)
        self.video_thread.frame_ready.connect(self._update_video_display)
        self.video_thread.running = True
        self.video_thread.start()

        # 添加警报记录
        self._add_alert("开始疲劳检测")
        print("🚀 开始疲劳检测")

    def _stop_detection(self):
        """停止检测（保持原有逻辑）"""
        if self.video_thread:
            self.video_thread.running = False
            self.video_thread.wait()
            self.video_thread = None

        if self.cap:
            self.cap.release()
            self.cap = None

        # 更新按钮状态
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        # 重置视频显示
        self.video_label.setText("检测已停止")

        # 添加警报记录
        self._add_alert("停止疲劳检测")
        print("⏹️ 停止疲劳检测")

    @pyqtSlot(np.ndarray, bool, float, int)
    def _update_video_display(self, frame, face_detected, yawn_prob, prediction):
        """更新视频显示"""
        try:
            # 调整图像大小
            display_frame = cv2.resize(frame, (640, 480))

            # 转换为QImage
            height, width, channel = display_frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(display_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

            # 转换为QPixmap并显示
            pixmap = QPixmap.fromImage(q_image)
            self.video_label.setPixmap(pixmap)

        except Exception as e:
            print(f"❌ 视频显示更新错误: {e}")

    def _add_alert(self, message):
        """添加警报记录"""
        current_time = time.strftime("%H:%M:%S")
        alert_message = f"{current_time} - {message}"
        self.alert_text.append(alert_message)

    def _update_gui(self):
        """更新GUI显示（保持原有逻辑）"""
        if not self.video_thread or not self.video_thread.running:
            return

        try:
            # 更新人脸检测状态
            if hasattr(self, '_last_face_detected'):
                if self._last_face_detected:
                    self.face_status_widget.value_label.setText("成功")
                    self.face_status_widget.value_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
                else:
                    self.face_status_widget.value_label.setText("未检测")
                    self.face_status_widget.value_label.setStyleSheet("color: #f44336; font-weight: bold;")

            # 更新状态显示
            if len(self.face_buffer) >= SEQUENCE_LENGTH:
                # 更新打哈欠概率
                if hasattr(self, '_last_yawn_prob'):
                    self.prob_status_widget.value_label.setText(f"{self._last_yawn_prob:.3f}")
                    if self._last_yawn_prob > self.yawn_threshold.get():
                        color = "#f44336"  # 红色
                    elif self._last_yawn_prob > 0.4:
                        color = "#FF9800"  # 橙色
                    else:
                        color = "#2e7d32"  # 绿色
                    self.prob_status_widget.value_label.setStyleSheet(f"color: {color}; font-weight: bold;")

                # 更新嘴部状态
                if hasattr(self, '_last_mar'):
                    if self._last_mar > self.mar_threshold.get():
                        self.mouth_status_widget.value_label.setText("张开")
                        self.mouth_status_widget.value_label.setStyleSheet("color: #FF9800; font-weight: bold;")
                    else:
                        self.mouth_status_widget.value_label.setText("正常")
                        self.mouth_status_widget.value_label.setStyleSheet("color: #4CAF50; font-weight: bold;")

                # 更新眼部状态
                if hasattr(self, '_last_ear'):
                    if self.eye_closed_frames >= self.long_eye_closed_threshold:
                        self.eye_status_widget.value_label.setText("长时间闭眼")
                        self.eye_status_widget.value_label.setStyleSheet("color: #f44336; font-weight: bold;")
                    elif self._last_ear < 0.25:
                        self.eye_status_widget.value_label.setText("闭合")
                        self.eye_status_widget.value_label.setStyleSheet("color: #FF9800; font-weight: bold;")
                    elif self._last_ear < 0.3:
                        self.eye_status_widget.value_label.setText("眯眼")
                        self.eye_status_widget.value_label.setStyleSheet("color: #FFC107; font-weight: bold;")
                    else:
                        self.eye_status_widget.value_label.setText("正常")
                        self.eye_status_widget.value_label.setStyleSheet("color: #4CAF50; font-weight: bold;")

                # 更新疲劳状态
                fatigue_level = self._evaluate_fatigue_status()
                self.fatigue_status_widget.value_label.setText(fatigue_level)
                if fatigue_level == "正常":
                    color = "#4CAF50"
                elif fatigue_level == "轻度疲劳":
                    color = "#FFC107"
                else:  # 重度疲劳
                    color = "#f44336"
                self.fatigue_status_widget.value_label.setStyleSheet(f"color: {color}; font-weight: bold;")

                # 更新连续检测显示
                self.consecutive_status_widget.value_label.setText(f"{self.consecutive_yawns}/{self.consecutive_threshold}")

                # 更新进度条（平滑进度条逻辑）
                progress = (self.consecutive_yawns / self.consecutive_threshold) * 100
                self.progress_bar.setValue(int(progress))

            # 更新统计信息
            if self.session_start_time:
                elapsed = int(time.time() - self.session_start_time)
                minutes, seconds = divmod(elapsed, 60)
                self.time_status_widget.value_label.setText(f"{minutes:02d}:{seconds:02d}")

            self.buffer_status_widget.value_label.setText(f"{len(self.face_buffer)}/{SEQUENCE_LENGTH}")
            self.count_status_widget.value_label.setText(f"{self.total_predictions}")
            self.yawn_status_widget.value_label.setText(f"{self.yawn_count}")
            self.blink_status_widget.value_label.setText(f"{self.blink_count}")

        except Exception as e:
            print(f"❌ GUI更新错误: {e}")

    def run(self):
        """运行GUI"""
        print("🎯 PyQt疲劳检测系统启动")
        print(f"📱 窗口大小: 1400x900")
        print(f"📜 右侧面板支持滚动")
        print(f"🎮 默认模式: 平衡模式")
        print(f"📷 检测到摄像头: 索引 {self.camera_index}")
        self.show()

def main():
    """主函数"""
    app = QApplication(sys.argv)

    # 设置应用程序样式
    app.setStyle('Fusion')

    # 检查模型文件
    model_path = os.path.join(MODEL_SAVE_PATH, "best_model.pth")
    if not os.path.exists(model_path):
        QMessageBox.critical(None, "错误", f"模型文件 {model_path} 不存在")
        return

    # 检查dlib文件
    if not os.path.exists(DLIB_PREDICTOR_PATH):
        QMessageBox.critical(None, "错误", f"dlib文件 {DLIB_PREDICTOR_PATH} 不存在")
        return

    try:
        window = FatigueDetectionGUI(model_path)
        window.run()

        print("🎯 PyQt疲劳检测系统启动")
        print("📱 现代化界面设计")
        print("🎨 支持主题样式")
        print("📜 自动滚动支持")
        print("🔧 保持原有检测逻辑")

        sys.exit(app.exec_())

    except Exception as e:
        print(f"❌ 程序启动失败: {e}")
        QMessageBox.critical(None, "错误", f"程序启动失败: {e}")

if __name__ == "__main__":
    main()
