"""
疲劳检测Web应用 - FFmpeg后端优化版本
使用FFmpeg进行视频压缩和优化，大幅减少网络传输数据量
集成完整的AI疲劳检测功能
"""
import cv2
import torch
import numpy as np
import dlib
from collections import deque
import base64
import json
import subprocess
import tempfile
import os
import sys
import pygame
import asyncio
import time
import threading
from queue import Queue, Empty
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# 导入AI检测相关模块
try:
    from config import *
    from model import create_model
    from utils import extract_face_landmarks, normalize_landmarks
    from database_config import get_db_connection, init_database

    # 定义模型路径
    MODEL_PATH = os.path.join(MODEL_SAVE_PATH, "best_model.pth")
    AI_MODULES_AVAILABLE = True
    print("✅ AI模块导入成功")
except ImportError as e:
    print(f"⚠️  AI模块导入失败: {e}")
    print("系统将在基础模式下运行")
    AI_MODULES_AVAILABLE = False

    # 定义基础配置
    MODEL_PATH = "output/models/best_model.pth"
    DLIB_PREDICTOR_PATH = "output/models/shape_predictor_68_face_landmarks.dat"
    SEQUENCE_LENGTH = 30
    FACE_SIZE = (64, 64)

    # 创建模拟函数
    def create_model():
        return None

    def extract_face_landmarks(frame, detector, predictor):
        return None, None

    def normalize_landmarks(landmarks, shape):
        return None

    def init_database():
        pass

app = FastAPI(title="疲劳检测系统 - FFmpeg优化版")

# 静态文件和模板
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class FFmpegVideoProcessor:
    """FFmpeg视频处理器"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.frame_counter = 0
        self.ffmpeg_path = None  # 将存储FFmpeg的路径

        # FFmpeg压缩参数
        self.compression_settings = {
            'preset': 'veryfast',
            'crf': '28',  # 恒定质量因子 (18-28为合理范围)
            'scale': '320:240',
            'fps': '5',
            'format': 'webm'
        }
        
    def check_ffmpeg(self):
        """检查FFmpeg是否可用"""
        # 首先尝试PATH中的ffmpeg
        try:
            result = subprocess.run(['ffmpeg', '-version'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.ffmpeg_path = 'ffmpeg'  # 使用PATH中的ffmpeg
                return True
        except:
            pass

        # 如果PATH中没有，尝试常见的Windows安装路径
        common_paths = [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
            r"D:\ffmpeg\bin\ffmpeg.exe"
        ]

        for path in common_paths:
            if os.path.exists(path):
                try:
                    result = subprocess.run([path, '-version'],
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        self.ffmpeg_path = path  # 使用找到的完整路径
                        print(f"✅ 找到FFmpeg: {path}")
                        return True
                except:
                    continue

        self.ffmpeg_path = None
        return False
    
    def compress_frame(self, frame_data):
        """使用FFmpeg压缩单帧"""
        try:
            # 解码base64图像
            if frame_data.startswith('data:image'):
                frame_data = frame_data.split(',')[1]
            
            img_data = base64.b64decode(frame_data)
            
            # 创建临时文件
            self.frame_counter += 1
            input_file = os.path.join(self.temp_dir, f'input_{self.frame_counter}.png')
            output_file = os.path.join(self.temp_dir, f'output_{self.frame_counter}.webm')
            
            # 写入输入文件
            with open(input_file, 'wb') as f:
                f.write(img_data)
            
            # FFmpeg压缩命令
            cmd = [
                self.ffmpeg_path or 'ffmpeg',  # 使用找到的FFmpeg路径
                '-y',  # 覆盖输出文件
                '-i', input_file,
                '-c:v', 'libvpx-vp9',  # VP9编码器
                '-preset', self.compression_settings['preset'],
                '-crf', self.compression_settings['crf'],
                '-vf', f"scale={self.compression_settings['scale']},fps={self.compression_settings['fps']}",
                '-f', self.compression_settings['format'],
                '-loglevel', 'quiet',  # 静默模式
                output_file
            ]
            
            # 执行压缩
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            
            if result.returncode == 0 and os.path.exists(output_file):
                # 读取压缩后的文件
                with open(output_file, 'rb') as f:
                    compressed_data = f.read()
                
                # 清理临时文件
                self._cleanup_files([input_file, output_file])
                
                # 计算压缩比
                original_size = len(img_data)
                compressed_size = len(compressed_data)
                compression_ratio = (1 - compressed_size / original_size) * 100
                
                return {
                    'data': base64.b64encode(compressed_data).decode('utf-8'),
                    'original_size': original_size,
                    'compressed_size': compressed_size,
                    'compression_ratio': compression_ratio
                }
            else:
                self._cleanup_files([input_file, output_file])
                return None
                
        except Exception as e:
            print(f"FFmpeg压缩失败: {e}")
            return None
    
    def compress_frame_fast(self, frame_data):
        """快速压缩模式 - 使用更激进的压缩参数"""
        try:
            # 解码并转换为OpenCV格式
            if frame_data.startswith('data:image'):
                frame_data = frame_data.split(',')[1]

            img_data = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return None

            # 获取目标分辨率
            scale_parts = self.compression_settings['scale'].split(':')
            new_width, new_height = int(scale_parts[0]), int(scale_parts[1])

            # 1. 降低分辨率
            frame_resized = cv2.resize(frame, (new_width, new_height))

            # 2. 可选的图像预处理
            if self.compression_settings.get('denoise', False):
                frame_resized = cv2.fastNlMeansDenoisingColored(frame_resized)

            # 3. 根据CRF值调整JPEG质量
            crf = int(self.compression_settings.get('crf', '28'))
            jpeg_quality = max(10, min(95, 100 - crf * 2))  # CRF转JPEG质量

            # 4. 高压缩编码
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            _, buffer = cv2.imencode('.jpg', frame_resized, encode_params)

            compressed_data = buffer.tobytes()
            original_size = len(img_data)
            compressed_size = len(compressed_data)
            compression_ratio = (1 - compressed_size / original_size) * 100

            return {
                'data': base64.b64encode(compressed_data).decode('utf-8'),
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_ratio': compression_ratio,
                'method': 'opencv_fast'
            }

        except Exception as e:
            print(f"快速压缩失败: {e}")
            return None

    def compress_frame_webp(self, frame_data):
        """使用WebP格式进行压缩 - 更好的压缩比"""
        try:
            if frame_data.startswith('data:image'):
                frame_data = frame_data.split(',')[1]

            img_data = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return None

            # 获取目标分辨率
            scale_parts = self.compression_settings['scale'].split(':')
            new_width, new_height = int(scale_parts[0]), int(scale_parts[1])
            frame_resized = cv2.resize(frame, (new_width, new_height))

            # WebP压缩参数
            crf = int(self.compression_settings.get('crf', '28'))
            webp_quality = max(10, min(100, 100 - crf))

            encode_params = [cv2.IMWRITE_WEBP_QUALITY, webp_quality]
            _, buffer = cv2.imencode('.webp', frame_resized, encode_params)

            compressed_data = buffer.tobytes()
            original_size = len(img_data)
            compressed_size = len(compressed_data)
            compression_ratio = (1 - compressed_size / original_size) * 100

            return {
                'data': base64.b64encode(compressed_data).decode('utf-8'),
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_ratio': compression_ratio,
                'method': 'webp'
            }

        except Exception as e:
            print(f"WebP压缩失败: {e}")
            return None

    def compress_frame_adaptive(self, frame_data):
        """自适应压缩 - 根据内容复杂度选择压缩策略"""
        try:
            if frame_data.startswith('data:image'):
                frame_data = frame_data.split(',')[1]

            img_data = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return None

            # 分析图像复杂度
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # 根据复杂度调整压缩参数
            if laplacian_var > 1000:  # 高复杂度图像
                quality_factor = 0.8
            elif laplacian_var > 500:  # 中等复杂度
                quality_factor = 0.6
            else:  # 低复杂度图像
                quality_factor = 0.4

            # 获取目标分辨率
            scale_parts = self.compression_settings['scale'].split(':')
            new_width, new_height = int(scale_parts[0]), int(scale_parts[1])
            frame_resized = cv2.resize(frame, (new_width, new_height))

            # 自适应质量
            base_crf = int(self.compression_settings.get('crf', '28'))
            adaptive_quality = max(10, min(95, int((100 - base_crf) * quality_factor)))

            encode_params = [cv2.IMWRITE_JPEG_QUALITY, adaptive_quality]
            _, buffer = cv2.imencode('.jpg', frame_resized, encode_params)

            compressed_data = buffer.tobytes()
            original_size = len(img_data)
            compressed_size = len(compressed_data)
            compression_ratio = (1 - compressed_size / original_size) * 100

            return {
                'data': base64.b64encode(compressed_data).decode('utf-8'),
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_ratio': compression_ratio,
                'method': 'adaptive',
                'complexity': laplacian_var,
                'quality_used': adaptive_quality
            }

        except Exception as e:
            print(f"自适应压缩失败: {e}")
            return None
    
    def _cleanup_files(self, files):
        """清理临时文件"""
        for file in files:
            try:
                if os.path.exists(file):
                    os.remove(file)
            except:
                pass
    
    def update_settings(self, settings):
        """更新压缩设置"""
        self.compression_settings.update(settings)

class OptimizedFatigueDetectionSystem:
    """优化的疲劳检测系统 - 集成完整AI检测功能"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.ai_available = AI_MODULES_AVAILABLE

        if self.ai_available:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"🔧 使用设备: {self.device}")

            # 加载AI模型
            self.model = self._load_model()

            # 初始化dlib
            self.detector = dlib.get_frontal_face_detector()
            # 检查两个可能的dlib模型路径
            dlib_paths = [
                DLIB_PREDICTOR_PATH,  # config.py中定义的路径
                "output/models/shape_predictor_68_face_landmarks.dat"  # 备用路径
            ]

            self.predictor = None
            for dlib_path in dlib_paths:
                if os.path.exists(dlib_path):
                    self.predictor = dlib.shape_predictor(dlib_path)
                    print(f"✅ dlib模型加载成功: {dlib_path}")
                    break

            if self.predictor is None:
                print(f"⚠️  dlib模型文件不存在，检查路径: {dlib_paths}")
                print("系统将在模拟模式下运行")
                self.ai_available = False
        else:
            print("⚠️  AI模块不可用，系统将在模拟模式下运行")
            self.device = None
            self.model = None
            self.detector = None
            self.predictor = None

        # 检测参数（保持原有逻辑）
        class SimpleVar:
            def __init__(self, value):
                self._value = value
            def get(self):
                return self._value
            def set(self, value):
                self._value = value

        self.yawn_threshold = SimpleVar(0.6)
        self.mar_threshold = SimpleVar(0.5)  # 默认使用平衡模式的MAR阈值
        self.ear_threshold = SimpleVar(0.18)  # 统一EAR阈值，适应所有眼型用户
        self.alert_cooldown = SimpleVar(5.0)

        # 检测状态
        self.is_detecting = False
        self.current_user = None
        self.current_mode = "平衡模式"

        # 疲劳状态评估相关（与PyQt版本保持一致）
        self.recent_yawns = []
        self.recent_blinks = []
        self.fatigue_window = 30  # 30秒窗口
        self.last_fatigue_status = "正常"  # 记录上一次的疲劳状态
        self.last_blink_time = 0
        self.eye_closed_frames = 0
        self.eye_closed_threshold = 10
        self.long_eye_closed_threshold = 60
        self.eye_closed_start_time = None

        # 缓冲区（保持原有逻辑）
        self.face_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.landmark_buffer = deque(maxlen=SEQUENCE_LENGTH)

        # 统计变量（保持原有逻辑）
        self.session_start_time = None
        self.yawn_count = 0
        self.blink_count = 0
        self.total_predictions = 0
        self.consecutive_yawns = 0
        self.consecutive_threshold = 15  # 默认使用平衡模式的连续检测阈值
        self.last_yawn_time = 0
        self.last_detection_time = 0
        self.no_detection_frames = 0
        self.decay_rate = 2.0

        # 眨眼检测变量（保持原有逻辑）
        self.eye_closed_frames = 0
        self.eye_closed_threshold = 3
        self.long_eye_closed_threshold = 30
        self.eye_closed_start_time = None
        self.last_blink_time = 0
        self.recent_blinks = []
        self.recent_yawns = []
        self.fatigue_window = 30.0
        self.last_fatigue_status = "正常"

        # 内部状态变量
        self._last_mar = 0.0
        self._last_ear = 0.3

        # 音频系统
        self.audio_path = "static/warning.mp3"
        self.audio_initialized = False
        self.warning_sound = None
        self._init_audio()

        # FFmpeg处理器
        self.video_processor = FFmpegVideoProcessor()
        self.use_ffmpeg = self.video_processor.check_ffmpeg()

        if self.use_ffmpeg:
            print("✅ FFmpeg可用，将使用FFmpeg进行视频压缩")
        else:
            print("⚠️ FFmpeg不可用，将使用快速压缩模式")

        # 处理队列
        self.frame_queue = Queue(maxsize=5)
        self.processing_thread = None

        # 压缩统计
        self.compression_stats = {
            'total_original_size': 0,
            'total_compressed_size': 0,
            'frames_processed': 0,
            'avg_compression_ratio': 0
        }

        # 最新结果
        self.latest_results = {
            'frame': None,
            'face_detected': False,
            'yawn_prob': 0.0,
            'prediction': 0,
            'mar': 0.0,
            'ear': 0.0,
            'fatigue_status': '正常',
            'consecutive_yawns': 0,
            'session_time': '00:00',
            'buffer_status': '0/30',
            'total_predictions': 0,
            'yawn_count': 0,
            'blink_count': 0,
            'progress': 0,
            'compression_stats': self.compression_stats.copy()
        }

    def _load_model(self):
        """加载AI模型（保持原有逻辑）"""
        if not self.ai_available:
            return None

        try:
            if not os.path.exists(self.model_path):
                print(f"⚠️  模型文件不存在: {self.model_path}")
                print("系统将在模拟模式下运行")
                return None

            model = create_model().to(self.device)
            checkpoint = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print("✅ AI模型加载成功")
            return model
        except Exception as e:
            print(f"❌ AI模型加载失败: {e}")
            print("系统将在模拟模式下运行")
            return None

    def _init_audio(self):
        """初始化音频系统"""
        try:
            pygame.mixer.init()
            if os.path.exists(self.audio_path):
                self.warning_sound = pygame.mixer.Sound(self.audio_path)
                self.audio_initialized = True
                print("✅ 音频系统初始化成功")
            else:
                print(f"❌ 警告音频文件不存在: {self.audio_path}")
                self.audio_initialized = False
        except Exception as e:
            print(f"❌ 音频系统初始化失败: {e}")
            self.audio_initialized = False

    def _play_warning_sound(self):
        """播放警告音频"""
        if self.audio_initialized and self.warning_sound:
            try:
                self.warning_sound.play()
                print("🔊 播放警告音频")
            except Exception as e:
                print(f"❌ 播放音频失败: {e}")

    def _preprocess_frame(self, frame):
        """预处理帧（保持原有逻辑）"""
        if self.predictor is None:
            return None, None, None

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
        if self.model is None or len(self.face_buffer) < SEQUENCE_LENGTH:
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
            self._last_mar = mar
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
            self._last_ear = avg_ear

            return avg_ear
        except:
            return 0.3

    def _detect_blink(self, ear):
        """检测眨眼和长时间闭眼（使用可配置的EAR阈值）"""
        ear_threshold = self.ear_threshold.get()
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
        """评估疲劳状态（与PyQt版本保持一致）"""
        current_time = time.time()

        # 清理过期的记录（30秒窗口）
        self.recent_yawns = [t for t in self.recent_yawns if current_time - t <= self.fatigue_window]
        self.recent_blinks = [t for t in self.recent_blinks if current_time - t <= self.fatigue_window]

        yawn_count_30s = len(self.recent_yawns)  # 30秒窗口内的打哈欠次数
        long_eye_closed = self.eye_closed_frames >= self.long_eye_closed_threshold

        # 疲劳状态判断逻辑（与PyQt版本完全一致）
        if yawn_count_30s >= 3 or long_eye_closed:
            return "重度疲劳"
        elif yawn_count_30s >= 2:
            return "中度疲劳"
        elif yawn_count_30s >= 1:
            return "轻度疲劳"
        else:
            return "正常"

    def _draw_face_landmarks(self, frame, face_rect, landmarks_norm):
        """在人脸上绘制特征点和人脸框（增强显示效果）"""
        if face_rect is None:
            print("⚠️  face_rect为None，跳过绘制")
            return frame

        # 绘制人脸框 - 使用更粗的线条和更亮的颜色
        x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
        print(f"🎨 开始绘制人脸框: ({x}, {y}, {w}, {h})")

        # 绘制多层人脸框以确保可见性 - 使用更粗的线条
        cv2.rectangle(frame, (x-3, y-3), (x + w + 3, y + h + 3), (0, 255, 0), 6)  # 外层绿框（更粗）
        cv2.rectangle(frame, (x-1, y-1), (x + w + 1, y + h + 1), (255, 255, 255), 4)  # 中层白框
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 内层绿框
        print(f"✅ 人脸框绘制完成")

        # 如果有归一化的landmarks，需要转换回原始坐标
        if landmarks_norm is not None and self.ai_available:
            print(f"🎯 开始绘制特征点，AI可用: {self.ai_available}")
            # 重新获取原始landmarks来绘制
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            if len(faces) > 0:
                landmarks = self.predictor(gray, faces[0])
                print(f"🎯 获取到landmarks，开始绘制68个特征点")

                # 绘制68个特征点 - 使用更大更明显的点
                for i in range(68):
                    x_point = landmarks.part(i).x
                    y_point = landmarks.part(i).y

                    # 根据不同区域使用不同颜色和大小
                    if i < 17:  # 下巴轮廓
                        color = (255, 255, 0)  # 青色
                        radius = 3
                    elif i < 22:  # 右眉毛
                        color = (0, 255, 255)  # 黄色
                        radius = 3
                    elif i < 27:  # 左眉毛
                        color = (0, 255, 255)  # 黄色
                        radius = 3
                    elif i < 36:  # 鼻子
                        color = (255, 0, 255)  # 紫色
                        radius = 3
                    elif i < 42:  # 右眼
                        color = (255, 0, 0)    # 蓝色
                        radius = 4  # 眼部特征点稍大
                    elif i < 48:  # 左眼
                        color = (255, 0, 0)    # 蓝色
                        radius = 4  # 眼部特征点稍大
                    else:  # 嘴部
                        color = (0, 0, 255)    # 红色
                        radius = 4  # 嘴部特征点稍大

                    # 绘制更大的特征点，带黑色边框增强对比度
                    cv2.circle(frame, (x_point, y_point), radius + 2, (0, 0, 0), -1)  # 黑色底
                    cv2.circle(frame, (x_point, y_point), radius + 1, (255, 255, 255), -1)  # 白色中层
                    cv2.circle(frame, (x_point, y_point), radius, color, -1)  # 彩色点

                # 绘制关键区域的连线 - 使用更粗的线条和更好的对比度
                # 眼部轮廓
                for eye_start, eye_end in [(36, 42), (42, 48)]:
                    eye_points = []
                    for i in range(eye_start, eye_end):
                        eye_points.append((landmarks.part(i).x, landmarks.part(i).y))
                    eye_points = np.array(eye_points, np.int32)
                    cv2.polylines(frame, [eye_points], True, (0, 0, 0), 5)  # 黑色底线
                    cv2.polylines(frame, [eye_points], True, (255, 255, 255), 3)  # 白色中线
                    cv2.polylines(frame, [eye_points], True, (255, 0, 0), 2)  # 蓝色线

                # 嘴部轮廓
                mouth_points = []
                for i in range(48, 68):
                    mouth_points.append((landmarks.part(i).x, landmarks.part(i).y))
                mouth_points = np.array(mouth_points, np.int32)
                cv2.polylines(frame, [mouth_points], True, (0, 0, 0), 5)  # 黑色底线
                cv2.polylines(frame, [mouth_points], True, (255, 255, 255), 3)  # 白色中线
                cv2.polylines(frame, [mouth_points], True, (0, 0, 255), 2)  # 红色线
        else:
            # 如果AI不可用，只绘制人脸框，不添加文字标识
            print(f"⚠️  AI不可用或landmarks为None，只绘制人脸框")

        print(f"✅ 人脸框和特征点绘制完成")

        return frame

    def _record_fatigue_status(self, fatigue_status):
        """记录疲劳状态到数据库（与PyQt版本保持一致）"""
        # 只在疲劳状态发生变化时记录（与PyQt版本逻辑一致）
        if fatigue_status != self.last_fatigue_status:
            if fatigue_status == "轻度疲劳" and self.last_fatigue_status == "正常":
                self._save_fatigue_record("轻度疲劳")
                print("⚠️ 轻度疲劳警告")
            elif fatigue_status == "中度疲劳" and self.last_fatigue_status in ["正常", "轻度疲劳"]:
                self._save_fatigue_record("中度疲劳")
                print("⚠️⚠️ 中度疲劳警告")
            elif fatigue_status == "重度疲劳" and self.last_fatigue_status in ["正常", "轻度疲劳", "中度疲劳"]:
                self._save_fatigue_record("重度疲劳")
                print("🚨 重度疲劳警告")
            elif fatigue_status == "正常":
                print("✅ 疲劳状态恢复正常")

            # 播放警告音频（如果状态变化且非正常）
            if fatigue_status != "正常":
                self._play_warning_sound()

            # 更新上一次疲劳状态
            self.last_fatigue_status = fatigue_status

    def _save_fatigue_record(self, fatigue_level):
        """保存疲劳记录到数据库（与PyQt版本保持一致）"""
        if not self.current_user:
            return

        try:
            if AI_MODULES_AVAILABLE:
                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO fatigue_records
                        (username, timestamp, fatigue_level)
                        VALUES (%s, %s, %s)
                    ''', (
                        self.current_user['username'],
                        datetime.now(),
                        fatigue_level
                    ))
                    conn.commit()

                print(f"💾 疲劳记录已保存: {self.current_user['username']} - {fatigue_level}")
            else:
                print(f"💾 疲劳记录（模拟）: {self.current_user['username']} - {fatigue_level}")

        except Exception as e:
            print(f"❌ 保存疲劳记录失败: {e}")

    def apply_preset(self, mode):
        """应用预设模式（EAR阈值保持不变）"""
        if mode == 'sensitive':
            self.yawn_threshold.set(0.6)  # 保持模型阈值不变
            self.mar_threshold.set(0.45)  # MAR阈值调整为0.45
            # EAR阈值保持不变，统一为0.18
            self.consecutive_threshold = 10  # 连续检测阈值10帧
            self.alert_cooldown.set(3.0)
            self.current_mode = "敏感模式"
        elif mode == 'balanced':
            self.yawn_threshold.set(0.6)  # 保持模型阈值不变
            self.mar_threshold.set(0.5)   # MAR阈值调整为0.5
            # EAR阈值保持不变，统一为0.18
            self.consecutive_threshold = 15  # 连续检测阈值15帧
            self.alert_cooldown.set(5.0)
            self.current_mode = "平衡模式"
        elif mode == 'conservative':
            self.yawn_threshold.set(0.6)  # 保持模型阈值不变
            self.mar_threshold.set(0.55)  # MAR阈值调整为0.55
            # EAR阈值保持不变，统一为0.18
            self.consecutive_threshold = 20  # 连续检测阈值20帧
            self.alert_cooldown.set(8.0)
            self.current_mode = "保守模式"

    def start_detection(self):
        """开始检测"""
        if self.is_detecting:
            return False

        self.is_detecting = True
        self.session_start_time = time.time()

        # 重置AI检测统计数据
        self.yawn_count = 0
        self.blink_count = 0
        self.total_predictions = 0
        self.consecutive_yawns = 0
        self.last_yawn_time = 0
        self.last_detection_time = 0
        self.no_detection_frames = 0

        # 重置缓冲区
        self.face_buffer.clear()
        self.landmark_buffer.clear()

        # 重置眨眼检测
        self.eye_closed_frames = 0
        self.eye_closed_start_time = None
        self.last_blink_time = 0
        self.recent_blinks = []
        self.recent_yawns = []

        # 重置压缩统计
        self.compression_stats = {
            'total_original_size': 0,
            'total_compressed_size': 0,
            'frames_processed': 0,
            'avg_compression_ratio': 0
        }

        # 启动处理线程
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()

        print("✅ 检测已开始 - AI + FFmpeg优化模式")
        return True

    def stop_detection(self):
        """停止检测"""
        self.is_detecting = False
        
        # 清空队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
                
        print("✅ 检测已停止")

    def add_frame(self, frame_data):
        """添加帧到处理队列"""
        if not self.is_detecting:
            return False
            
        try:
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()  # 移除最老的帧
                except Empty:
                    pass
            
            self.frame_queue.put_nowait(frame_data)
            return True
        except:
            return False

    def _processing_loop(self):
        """处理循环 - 集成完整AI检测"""
        compression_methods = ['fast', 'webp', 'adaptive']
        current_method_index = 0

        while self.is_detecting:
            try:
                # 获取帧数据
                frame_data = None
                try:
                    frame_data = self.frame_queue.get(timeout=0.5)
                except Empty:
                    continue

                # 执行完整的AI检测流程
                detection_result = self.process_frame_with_ai(frame_data)

                if detection_result:
                    # 更新最新结果
                    self.latest_results.update(detection_result)
                    self.latest_results['compression_stats'] = self.compression_stats.copy()

                    # 动态调整压缩方法（基于性能）
                    if self.compression_stats['frames_processed'] % 50 == 0:
                        self._optimize_compression_method()

            except Exception as e:
                print(f"处理循环错误: {e}")
                time.sleep(0.1)

    def process_frame_with_ai(self, frame_data):
        """处理视频帧并进行AI检测（完整检测逻辑）"""
        try:
            # 将base64数据转换为OpenCV图像
            if frame_data.startswith('data:image'):
                frame_data = frame_data.split(',')[1]

            # 解码base64数据
            img_data = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return None

            # 先进行FFmpeg压缩
            compressed_result = self.video_processor.compress_frame_fast(frame_data)
            if compressed_result:
                self._update_compression_stats(compressed_result)

            # 如果AI不可用，使用模拟检测
            if not self.ai_available:
                return self._simulate_detection(frame)

            # 执行AI检测逻辑（保持原有逻辑）
            face_img, landmarks_norm, face_rect = self._preprocess_frame(frame)
            face_detected = face_img is not None

            # 获取原始landmarks用于MAR/EAR计算
            original_landmarks = None
            if face_detected:
                original_face_img, original_landmarks = extract_face_landmarks(frame, self.detector, self.predictor)
                # 立即绘制人脸框和特征点
                print(f"🎨 绘制人脸框和特征点，人脸区域: {face_rect.left()}, {face_rect.top()}, {face_rect.width()}, {face_rect.height()}")
                frame = self._draw_face_landmarks(frame, face_rect, landmarks_norm)
                print(f"✅ 人脸框和特征点绘制完成")

            yawn_prob = 0.0
            prediction = 0

            if face_detected:
                self.face_buffer.append(face_img)
                self.landmark_buffer.append(landmarks_norm)

                # 如果缓冲区满了，进行预测（保持原有逻辑）
                if len(self.face_buffer) >= SEQUENCE_LENGTH:
                    yawn_prob, model_prediction = self._predict_yawn()
                    self.total_predictions += 1

                    # 计算当前帧的嘴部长宽比和眼部长宽比（使用原始landmarks）
                    current_mar = self._calculate_mouth_aspect_ratio(original_landmarks)
                    current_ear = self._calculate_eye_aspect_ratio(original_landmarks)

                    # 检测眨眼（保持原有逻辑）
                    blink_detected = self._detect_blink(current_ear)

                    # 新的检测逻辑：模型预测 + MAR阈值的组合判断（保持原有逻辑）
                    model_says_yawn = yawn_prob > self.yawn_threshold.get()
                    mar_says_yawn = current_mar > self.mar_threshold.get()

                    # 最终判断：两个条件都满足才认为是打哈欠（保持原有逻辑）
                    final_prediction = 1 if (model_says_yawn and mar_says_yawn) else 0

                    # 更新连续检测计数 - 使用平滑衰减机制（保持原有逻辑）
                    current_time = time.time()
                    if final_prediction == 1:
                        # 检测到打哈欠：增加计数，更新最后检测时间
                        self.consecutive_yawns += 1
                        self.last_detection_time = current_time
                        self.no_detection_frames = 0  # 重置未检测帧数
                        print(f"🔍 打哈欠检测: 模型={yawn_prob:.3f}({'✓' if model_says_yawn else '✗'}), MAR={current_mar:.3f}({'✓' if mar_says_yawn else '✗'}), 连续={self.consecutive_yawns}")
                    else:
                        # 未检测到打哈欠：使用平滑衰减
                        self.no_detection_frames += 1

                        # 如果有之前的检测记录，则开始衰减
                        if self.consecutive_yawns > 0:
                            # 计算衰减量：基于时间的衰减
                            if self.last_detection_time > 0:
                                time_since_last = current_time - self.last_detection_time
                                # 每秒衰减decay_rate帧，但至少保持1秒不衰减
                                if time_since_last > 1.0:  # 1秒后开始衰减
                                    decay_amount = int((time_since_last - 1.0) * self.decay_rate)
                                    self.consecutive_yawns = max(0, self.consecutive_yawns - decay_amount)

                                    if self.consecutive_yawns == 0:
                                        print(f"📉 进度条衰减至零（未检测{self.no_detection_frames}帧，时间间隔{time_since_last:.1f}秒）")
                                    else:
                                        print(f"📉 进度条衰减: {self.consecutive_yawns}（未检测{self.no_detection_frames}帧）")
                            else:
                                # 如果没有时间记录，立即开始衰减
                                if self.no_detection_frames > 30:  # 30帧后开始衰减（约1秒）
                                    self.consecutive_yawns = max(0, self.consecutive_yawns - 1)
                        else:
                            # 如果consecutive_yawns已经是0，保持为0
                            self.consecutive_yawns = 0

                    # 检查是否触发警报（保持原有逻辑）
                    if (self.consecutive_yawns >= self.consecutive_threshold and
                        (current_time - self.last_yawn_time) > self.alert_cooldown.get()):
                        self.yawn_count += 1
                        self.last_yawn_time = current_time
                        self.recent_yawns.append(current_time)
                        print(f"🚨 触发警报！连续{self.consecutive_yawns}帧检测到打哈欠")
                        self._play_warning_sound()

                    # 更新prediction变量用于GUI显示
                    prediction = final_prediction

                # 人脸框和特征点已在前面绘制
            else:
                # 未检测到人脸时的衰减逻辑（保持原有逻辑）
                if self.consecutive_yawns > 0:
                    current_time = time.time()
                    self.no_detection_frames += 1

                    # 如果有之前的检测记录，则开始衰减
                    if self.last_detection_time > 0:
                        time_since_last = current_time - self.last_detection_time
                        # 未检测到人脸时，衰减更快一些
                        if time_since_last > 0.5:  # 0.5秒后开始衰减
                            decay_amount = int((time_since_last - 0.5) * self.decay_rate * 1.5)  # 衰减速度1.5倍
                            old_consecutive = self.consecutive_yawns
                            self.consecutive_yawns = max(0, self.consecutive_yawns - decay_amount)

                            if old_consecutive != self.consecutive_yawns:
                                if self.consecutive_yawns == 0:
                                    print(f"📉 未检测到人脸，进度条衰减至零（未检测{self.no_detection_frames}帧）")
                                else:
                                    print(f"📉 未检测到人脸，进度条衰减: {self.consecutive_yawns}")
                    else:
                        # 如果没有时间记录，较快衰减
                        if self.no_detection_frames > 15:  # 15帧后开始衰减（约0.5秒）
                            self.consecutive_yawns = max(0, self.consecutive_yawns - 1)

            # 评估疲劳状态
            fatigue_status = self._evaluate_fatigue_status()

            # 记录疲劳状态到数据库
            self._record_fatigue_status(fatigue_status)

            # 更新最新结果用于Web显示
            return self._update_latest_results(frame, face_detected, yawn_prob, prediction, fatigue_status)

        except Exception as e:
            print(f"❌ AI检测处理错误: {e}")
            return None

    def _update_latest_results(self, frame, face_detected, yawn_prob, prediction, fatigue_status):
        """更新最新的检测结果用于Web显示"""
        # 注意：frame已经包含了人脸框和特征点（在_draw_face_landmarks中绘制）
        # 不再添加额外的文本信息，只保留人脸框和特征点

        # 将frame转换为base64用于Web显示，使用更高质量
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        # 计算会话时间
        session_time = "00:00"
        if self.session_start_time:
            elapsed = int(time.time() - self.session_start_time)
            minutes = elapsed // 60
            seconds = elapsed % 60
            session_time = f"{minutes:02d}:{seconds:02d}"

        return {
            'frame': frame_base64,
            'face_detected': face_detected,
            'yawn_prob': round(yawn_prob, 3),
            'prediction': prediction,
            'mar': round(self._last_mar, 3),
            'ear': round(self._last_ear, 3),
            'fatigue_status': fatigue_status,
            'consecutive_yawns': self.consecutive_yawns,
            'session_time': session_time,
            'buffer_status': f"{min(SEQUENCE_LENGTH, len(self.face_buffer))}/{SEQUENCE_LENGTH}",
            'total_predictions': self.total_predictions,
            'yawn_count': self.yawn_count,
            'blink_count': self.blink_count,
            'progress': min(100, int((self.consecutive_yawns / self.consecutive_threshold) * 100))
        }

    def _simulate_detection(self, frame):
        """模拟AI检测（当AI模块不可用时）"""
        import random

        # 模拟检测结果
        face_detected = random.choice([True, False, True, True])  # 75%概率检测到人脸
        yawn_prob = random.uniform(0.0, 1.0)
        prediction = 1 if yawn_prob > 0.7 else 0

        # 更新统计
        self.total_predictions += 1
        if prediction == 1:
            self.yawn_count += 1

        # 模拟眨眼检测
        if random.random() < 0.1:  # 10%概率检测到眨眼
            self.blink_count += 1

        # 模拟疲劳状态
        if self.yawn_count >= 3:
            fatigue_status = "中度疲劳"
        elif self.yawn_count >= 1:
            fatigue_status = "轻度疲劳"
        else:
            fatigue_status = "正常"

        # 在frame上绘制模拟检测框和特征点
        if face_detected:
            h, w = frame.shape[:2]
            # 绘制人脸框
            face_x, face_y = w//4, h//4
            face_w, face_h = w//2, h//2
            cv2.rectangle(frame, (face_x, face_y), (face_x + face_w, face_y + face_h), (0, 255, 0), 3)

            # 绘制模拟特征点
            # 眼部区域
            eye_y = face_y + face_h//3
            left_eye_x = face_x + face_w//4
            right_eye_x = face_x + 3*face_w//4

            # 左眼
            for i in range(6):
                angle = i * 60 * np.pi / 180
                x = int(left_eye_x + 15 * np.cos(angle))
                y = int(eye_y + 8 * np.sin(angle))
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

            # 右眼
            for i in range(6):
                angle = i * 60 * np.pi / 180
                x = int(right_eye_x + 15 * np.cos(angle))
                y = int(eye_y + 8 * np.sin(angle))
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

            # 嘴部区域
            mouth_y = face_y + 2*face_h//3
            mouth_x = face_x + face_w//2

            # 嘴部轮廓
            for i in range(8):
                angle = i * 45 * np.pi / 180
                x = int(mouth_x + 20 * np.cos(angle))
                y = int(mouth_y + 10 * np.sin(angle))
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

            # 鼻子
            nose_y = face_y + face_h//2
            nose_x = face_x + face_w//2
            cv2.circle(frame, (nose_x, nose_y), 3, (255, 0, 255), -1)
            cv2.circle(frame, (nose_x-5, nose_y+5), 2, (255, 0, 255), -1)
            cv2.circle(frame, (nose_x+5, nose_y+5), 2, (255, 0, 255), -1)

        # 记录疲劳状态到数据库
        self._record_fatigue_status(fatigue_status)

        return self._update_latest_results(frame, face_detected, yawn_prob, prediction, fatigue_status)

    def _optimize_compression_method(self):
        """根据性能动态优化压缩方法"""
        try:
            avg_ratio = self.compression_stats['avg_compression_ratio']

            # 如果压缩比不够好，切换到更激进的压缩
            if avg_ratio < 70:  # 压缩比低于70%
                new_crf = min(35, int(self.video_processor.compression_settings['crf']) + 2)
                self.video_processor.compression_settings['crf'] = str(new_crf)
                print(f"🔧 自动优化: 提高压缩比，CRF调整为 {new_crf}")

            # 如果压缩比太高可能影响质量，适当降低
            elif avg_ratio > 90:  # 压缩比高于90%
                new_crf = max(18, int(self.video_processor.compression_settings['crf']) - 1)
                self.video_processor.compression_settings['crf'] = str(new_crf)
                print(f"🔧 自动优化: 保持质量，CRF调整为 {new_crf}")

        except Exception as e:
            print(f"优化压缩方法失败: {e}")

    def _update_compression_stats(self, result):
        """更新压缩统计"""
        self.compression_stats['total_original_size'] += result['original_size']
        self.compression_stats['total_compressed_size'] += result['compressed_size']
        self.compression_stats['frames_processed'] += 1
        
        if self.compression_stats['total_original_size'] > 0:
            self.compression_stats['avg_compression_ratio'] = (
                (1 - self.compression_stats['total_compressed_size'] / 
                 self.compression_stats['total_original_size']) * 100
            )



    def get_latest_results(self):
        """获取最新结果"""
        return self.latest_results.copy()

    def update_compression_settings(self, settings):
        """更新压缩设置"""
        self.video_processor.update_settings(settings)

# 全局检测系统实例
detection_system = OptimizedFatigueDetectionSystem(MODEL_PATH)

# 在应用启动时初始化数据库
@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    if AI_MODULES_AVAILABLE:
        try:
            init_database()
            print("✅ 数据库初始化成功")
        except Exception as e:
            print(f"⚠️  数据库初始化失败: {e}")
    else:
        print("⚠️  跳过数据库初始化（AI模块不可用）")

# 路由定义
@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    """用户登录 - 使用数据库认证"""
    try:
        if AI_MODULES_AVAILABLE:
            # 使用数据库认证
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT username, password FROM users WHERE username = %s", (username,))
                user = cursor.fetchone()

            if user and user[1] == password:
                detection_system.current_user = {
                    'username': user[0],
                    'full_name': user[0]
                }
                return RedirectResponse(url="/dashboard", status_code=302)
            else:
                return JSONResponse({"success": False, "message": "用户名或密码错误"})
        else:
            # 如果数据库不可用，使用简化认证
            simple_users = {"test": "123456", "admin": "admin"}
            if username in simple_users and simple_users[username] == password:
                detection_system.current_user = {
                    'username': username,
                    'full_name': username
                }
                return RedirectResponse(url="/dashboard", status_code=302)
            else:
                return JSONResponse({"success": False, "message": "用户名或密码错误"})

    except Exception as e:
        print(f"❌ 登录失败: {e}")
        return JSONResponse({"success": False, "message": f"登录失败: {e}"})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """注册页面"""
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register")
async def register(username: str = Form(...), password: str = Form(...), confirm_password: str = Form(...)):
    """用户注册 - 使用数据库存储"""
    if len(password) < 6:
        return JSONResponse({"success": False, "message": "密码长度至少6位"})

    if password != confirm_password:
        return JSONResponse({"success": False, "message": "两次输入的密码不一致"})

    try:
        if AI_MODULES_AVAILABLE:
            # 使用数据库存储
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
                conn.commit()

            return JSONResponse({"success": True, "message": f"用户 {username} 注册成功！"})
        else:
            # 如果数据库不可用，返回提示
            return JSONResponse({"success": False, "message": "数据库不可用，无法注册新用户"})

    except Exception as e:
        if "Duplicate entry" in str(e):
            return JSONResponse({"success": False, "message": "用户名已存在，请选择其他用户名"})
        else:
            return JSONResponse({"success": False, "message": f"注册失败: {e}"})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    if not detection_system.current_user:
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse("dashboard_backend_ffmpeg.html", {
        "request": request,
        "user": detection_system.current_user
    })

@app.post("/api/start_detection")
async def start_detection():
    if detection_system.start_detection():
        return JSONResponse({"success": True, "message": "检测已开始"})
    else:
        return JSONResponse({"success": False, "message": "启动检测失败"})

@app.post("/api/stop_detection")
async def stop_detection():
    detection_system.stop_detection()
    return JSONResponse({"success": True, "message": "检测已停止"})

@app.post("/api/update_compression")
async def update_compression(
    preset: str = Form(...),
    crf: str = Form(...),
    scale: str = Form(...),
    fps: str = Form(...)
):
    """更新压缩设置"""
    settings = {
        'preset': preset,
        'crf': crf,
        'scale': scale,
        'fps': fps
    }
    detection_system.update_compression_settings(settings)
    return JSONResponse({"success": True, "message": "压缩设置已更新"})

@app.get("/api/compression_stats")
async def get_compression_stats():
    """获取压缩统计"""
    return JSONResponse(detection_system.compression_stats)

@app.post("/api/reset_stats")
async def reset_stats():
    """重置统计数据"""
    detection_system.compression_stats = {
        'total_original_size': 0,
        'total_compressed_size': 0,
        'frames_processed': 0,
        'avg_compression_ratio': 0
    }
    return JSONResponse({"success": True, "message": "统计数据已重置"})

@app.get("/api/system_info")
async def get_system_info():
    """获取系统信息"""
    import psutil
    import platform

    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        system_info = {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "cpu_usage": cpu_percent,
            "memory_total": memory.total,
            "memory_used": memory.used,
            "memory_percent": memory.percent,
            "disk_total": disk.total,
            "disk_used": disk.used,
            "disk_percent": (disk.used / disk.total) * 100,
            "ffmpeg_available": detection_system.use_ffmpeg
        }

        return JSONResponse(system_info)
    except Exception as e:
        return JSONResponse({"error": str(e)})

@app.post("/api/benchmark")
async def run_benchmark():
    """运行压缩性能基准测试"""
    try:
        # 创建测试图像
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.png', test_image)
        test_data = base64.b64encode(buffer).decode('utf-8')

        # 测试不同压缩方法
        methods = ['fast', 'webp', 'adaptive']
        results = {}

        for method in methods:
            start_time = time.time()

            if method == 'fast':
                result = detection_system.video_processor.compress_frame_fast(f"data:image/png;base64,{test_data}")
            elif method == 'webp':
                result = detection_system.video_processor.compress_frame_webp(f"data:image/png;base64,{test_data}")
            elif method == 'adaptive':
                result = detection_system.video_processor.compress_frame_adaptive(f"data:image/png;base64,{test_data}")

            end_time = time.time()

            if result:
                results[method] = {
                    "compression_ratio": result['compression_ratio'],
                    "processing_time": (end_time - start_time) * 1000,  # ms
                    "original_size": result['original_size'],
                    "compressed_size": result['compressed_size']
                }
            else:
                results[method] = {"error": "压缩失败"}

        return JSONResponse({
            "success": True,
            "benchmark_results": results,
            "test_image_size": f"{test_image.shape[1]}x{test_image.shape[0]}"
        })

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})

@app.get("/api/performance_tips")
async def get_performance_tips():
    """获取性能优化建议"""
    tips = []

    # 基于当前统计给出建议
    if detection_system.compression_stats['frames_processed'] > 0:
        avg_ratio = detection_system.compression_stats['avg_compression_ratio']

        if avg_ratio < 50:
            tips.append("压缩比较低，建议提高CRF值或降低分辨率")
        elif avg_ratio > 85:
            tips.append("压缩比很高，可能影响检测精度，建议适当降低压缩")

        if detection_system.compression_stats['frames_processed'] > 100:
            tips.append("系统运行稳定，可以尝试提高帧率")

    # 系统相关建议
    if not detection_system.use_ffmpeg:
        tips.append("建议安装FFmpeg以获得更好的压缩效果")

    tips.extend([
        "在网络带宽有限时，选择保守模式",
        "在本地网络环境下，可以选择敏感模式获得更好的检测精度",
        "定期重置统计数据以获得准确的性能指标"
    ])

    return JSONResponse({"tips": tips})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """优化的WebSocket处理"""
    await websocket.accept()
    print("WebSocket连接已建立")
    
    last_send_time = 0
    send_interval = 0.2  # 200ms发送间隔
    
    try:
        while True:
            try:
                # 接收消息
                data = await asyncio.wait_for(websocket.receive_json(), timeout=0.1)
                
                if data.get("type") == "video_frame" and detection_system.is_detecting:
                    # 后端直接处理原始视频帧并进行压缩
                    frame_data = data.get("frame")
                    if frame_data:
                        detection_system.add_frame(frame_data)
                        
            except asyncio.TimeoutError:
                # 定期发送结果
                current_time = time.time()
                if current_time - last_send_time >= send_interval:
                    if detection_system.is_detecting:
                        results = detection_system.get_latest_results()
                        await websocket.send_json({
                            "type": "detection_result",
                            "data": results
                        })
                    last_send_time = current_time
                continue
                
    except WebSocketDisconnect:
        print("WebSocket连接断开")
    except Exception as e:
        print(f"WebSocket错误: {e}")



@app.get("/api/status")
async def get_status():
    """获取系统状态"""
    return JSONResponse({
        "is_detecting": detection_system.is_detecting,
        "current_mode": detection_system.current_mode,
        "ffmpeg_available": detection_system.use_ffmpeg,
        "results": detection_system.get_latest_results()
    })

@app.post("/api/apply_preset")
async def apply_preset(mode: str = Form(...)):
    """应用预设模式"""
    # 应用检测参数预设
    detection_system.apply_preset(mode)

    # 同时设置FFmpeg压缩参数
    if mode == 'sensitive':
        # 敏感模式：高质量低延迟设置
        settings = {
            'preset': 'ultrafast',
            'crf': '23',
            'scale': '480:360',
            'fps': '10'
        }
    elif mode == 'balanced':
        # 平衡模式：平衡设置
        settings = {
            'preset': 'veryfast',
            'crf': '28',
            'scale': '320:240',
            'fps': '5'
        }
    elif mode == 'conservative':
        # 保守模式：高压缩低带宽设置
        settings = {
            'preset': 'fast',
            'crf': '32',
            'scale': '240:180',
            'fps': '3'
        }

    detection_system.update_compression_settings(settings)

    # 返回详细的参数信息
    return JSONResponse({
        "success": True,
        "message": f"已切换到{detection_system.current_mode}",
        "parameters": {
            "mar_threshold": detection_system.mar_threshold.get(),
            "ear_threshold": detection_system.ear_threshold.get(),
            "consecutive_threshold": detection_system.consecutive_threshold,
            "alert_cooldown": detection_system.alert_cooldown.get()
        }
    })

@app.get("/logout")
async def logout():
    """用户退出"""
    detection_system.stop_detection()
    detection_system.current_user = None
    return RedirectResponse(url="/", status_code=302)

if __name__ == "__main__":
    print("🚀 启动疲劳检测Web应用 - FFmpeg优化版")
    print("📱 请在浏览器中访问: http://localhost:8000")
    if AI_MODULES_AVAILABLE:
        print("🔑 数据库认证已启用")
        print("📝 可以注册新用户或使用现有账户")
    else:
        print("🔑 测试账户: test/123456, admin/admin")
    print("⚡ FFmpeg优化特性:")
    print("   - VP9视频编码")
    print("   - 可调节压缩质量")
    print("   - 智能分辨率缩放")
    print("   - 帧率优化")
    print("   - 实时压缩统计")

    uvicorn.run(app, host="localhost", port=8000)
