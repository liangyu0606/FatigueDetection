"""
疲劳检测GUI界面 - 按照用户要求设计
布局：
- 左上角：摄像头显示区域（显示人脸框和特征点）
- 右上角：预测信息面板
- 下部：阈值调节控制面板
"""
import cv2
import torch
import numpy as np
import dlib
from collections import deque
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os

from config import *
from model import create_model
from utils import extract_face_landmarks, normalize_landmarks

class FatigueDetectionGUI:
    def __init__(self, model_path: str):
        """疲劳检测GUI"""
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化主窗口
        self.root = tk.Tk()
        self.root.title("疲劳驾驶检测系统")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        self.root.resizable(True, True)
        
        # 加载模型
        self.model = self._load_model()
        
        # 初始化dlib
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(DLIB_PREDICTOR_PATH)
        
        # 摄像头
        self.cap = None
        self.running = False
        
        # 检测参数（可调节）
        self.yawn_threshold = tk.DoubleVar(value=0.6)
        self.consecutive_threshold = tk.IntVar(value=3)
        self.alert_cooldown = tk.DoubleVar(value=5.0)
        
        # 缓存队列
        self.face_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.landmark_buffer = deque(maxlen=SEQUENCE_LENGTH)
        
        # 状态变量
        self.yawn_count = 0
        self.total_predictions = 0
        self.consecutive_yawns = 0
        self.session_start_time = None
        self.last_yawn_time = 0
        
        # 创建GUI
        self._create_gui()
        
    def _load_model(self):
        """加载模型"""
        try:
            model = create_model().to(self.device)
            checkpoint = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            return model
        except Exception as e:
            messagebox.showerror("错误", f"模型加载失败: {e}")
            return None
    
    def _create_gui(self):
        """创建GUI界面"""
        # 主容器
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 上部区域：摄像头 + 预测信息
        top_frame = tk.Frame(main_container, bg='#f0f0f0')
        top_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # 左上角：摄像头显示区域
        self._create_camera_area(top_frame)

        # 右上角：预测信息面板
        self._create_prediction_panel(top_frame)

        # 下部：阈值调节控制面板
        self._create_control_panel(main_container)
    
    def _create_camera_area(self, parent):
        """创建摄像头显示区域"""
        camera_frame = tk.LabelFrame(parent, text="摄像头视频", font=("Arial", 12, "bold"),
                                   bg='#f0f0f0', fg='#333333', padx=10, pady=10)
        camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # 视频显示标签 - 使用合适的尺寸
        self.video_label = tk.Label(camera_frame, bg='black')
        self.video_label.pack(pady=5)

        # 设置视频显示的实际尺寸
        self.video_width = 480  # 稍微小一点但仍然清晰
        self.video_height = 360
        
        # 摄像头控制按钮
        btn_frame = tk.Frame(camera_frame, bg='#f0f0f0')
        btn_frame.pack(fill=tk.X, pady=5)
        
        self.start_btn = tk.Button(btn_frame, text="开始检测", command=self.start_detection,
                                 bg='#4CAF50', fg='white', font=("Arial", 10, "bold"),
                                 padx=20, pady=5)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_btn = tk.Button(btn_frame, text="停止检测", command=self.stop_detection,
                                bg='#f44336', fg='white', font=("Arial", 10, "bold"),
                                padx=20, pady=5, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)
        
        # 摄像头选择
        tk.Label(btn_frame, text="摄像头:", bg='#f0f0f0', font=("Arial", 10)).pack(side=tk.LEFT, padx=(20, 5))
        self.camera_var = tk.StringVar(value="0")
        camera_combo = ttk.Combobox(btn_frame, textvariable=self.camera_var, 
                                  values=["0", "1", "2"], width=5, state="readonly")
        camera_combo.pack(side=tk.LEFT)
    
    def _create_prediction_panel(self, parent):
        """创建预测信息面板"""
        pred_frame = tk.LabelFrame(parent, text="检测信息", font=("Arial", 12, "bold"),
                                 bg='#f0f0f0', fg='#333333', padx=15, pady=15)
        pred_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        # 当前状态显示
        status_frame = tk.Frame(pred_frame, bg='#f0f0f0')
        status_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.status_label = tk.Label(status_frame, text="状态: 未开始", 
                                   font=("Arial", 14, "bold"), bg='#f0f0f0', fg='#666666')
        self.status_label.pack()
        
        # 预测结果区域
        result_frame = tk.LabelFrame(pred_frame, text="实时预测", font=("Arial", 11, "bold"),
                                   bg='#f0f0f0', fg='#333333', padx=10, pady=10)
        result_frame.pack(fill=tk.X, pady=(0, 15))
        
        # 人脸检测状态
        self.face_status = tk.Label(result_frame, text="人脸检测: 无", 
                                  font=("Arial", 11), bg='#f0f0f0', fg='#666666')
        self.face_status.pack(anchor=tk.W, pady=2)
        
        # 打哈欠概率
        self.prob_label = tk.Label(result_frame, text="打哈欠概率: 0.000", 
                                 font=("Arial", 11), bg='#f0f0f0', fg='#666666')
        self.prob_label.pack(anchor=tk.W, pady=2)
        
        # 预测结果
        self.prediction_label = tk.Label(result_frame, text="预测结果: 正常", 
                                       font=("Arial", 11, "bold"), bg='#f0f0f0', fg='#4CAF50')
        self.prediction_label.pack(anchor=tk.W, pady=2)
        
        # 连续检测次数
        self.consecutive_label = tk.Label(result_frame, text="连续检测: 0", 
                                        font=("Arial", 11), bg='#f0f0f0', fg='#666666')
        self.consecutive_label.pack(anchor=tk.W, pady=2)
        
        # 统计信息区域
        stats_frame = tk.LabelFrame(pred_frame, text="会话统计", font=("Arial", 11, "bold"),
                                  bg='#f0f0f0', fg='#333333', padx=10, pady=10)
        stats_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.time_label = tk.Label(stats_frame, text="会话时间: 00:00", 
                                 font=("Arial", 10), bg='#f0f0f0', fg='#666666')
        self.time_label.pack(anchor=tk.W, pady=1)
        
        self.buffer_label = tk.Label(stats_frame, text="缓冲区: 0/30", 
                                   font=("Arial", 10), bg='#f0f0f0', fg='#666666')
        self.buffer_label.pack(anchor=tk.W, pady=1)
        
        self.count_label = tk.Label(stats_frame, text="总检测次数: 0", 
                                  font=("Arial", 10), bg='#f0f0f0', fg='#666666')
        self.count_label.pack(anchor=tk.W, pady=1)
        
        self.yawn_count_label = tk.Label(stats_frame, text="打哈欠次数: 0", 
                                       font=("Arial", 10), bg='#f0f0f0', fg='#666666')
        self.yawn_count_label.pack(anchor=tk.W, pady=1)
        
        # 警报状态
        self.alert_frame = tk.Frame(pred_frame, bg='#f0f0f0')
        self.alert_frame.pack(fill=tk.X)
        
        self.alert_label = tk.Label(self.alert_frame, text="", 
                                  font=("Arial", 12, "bold"), bg='#f0f0f0')
        self.alert_label.pack()
    
    def _create_control_panel(self, parent):
        """创建阈值调节控制面板"""
        control_frame = tk.LabelFrame(parent, text="参数调节", font=("Arial", 12, "bold"),
                                    bg='#f0f0f0', fg='#333333', padx=15, pady=15)
        control_frame.pack(fill=tk.X, pady=(10, 0))
        
        # 创建三列布局
        col1 = tk.Frame(control_frame, bg='#f0f0f0')
        col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20))
        
        col2 = tk.Frame(control_frame, bg='#f0f0f0')
        col2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20))
        
        col3 = tk.Frame(control_frame, bg='#f0f0f0')
        col3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 第一列：打哈欠检测阈值
        tk.Label(col1, text="打哈欠检测阈值", font=("Arial", 11, "bold"), 
               bg='#f0f0f0', fg='#333333').pack(anchor=tk.W)
        
        threshold_frame = tk.Frame(col1, bg='#f0f0f0')
        threshold_frame.pack(fill=tk.X, pady=5)
        
        self.threshold_scale = tk.Scale(threshold_frame, from_=0.1, to=0.9, resolution=0.01,
                                      variable=self.yawn_threshold, orient=tk.HORIZONTAL,
                                      bg='#f0f0f0', fg='#333333', font=("Arial", 10),
                                      length=200)
        self.threshold_scale.pack(side=tk.LEFT)
        
        self.threshold_value = tk.Label(threshold_frame, text="0.60", 
                                      font=("Arial", 11, "bold"), bg='#f0f0f0', fg='#2196F3')
        self.threshold_value.pack(side=tk.LEFT, padx=(10, 0))
        
        # 第二列：连续检测阈值
        tk.Label(col2, text="连续检测阈值", font=("Arial", 11, "bold"), 
               bg='#f0f0f0', fg='#333333').pack(anchor=tk.W)
        
        consecutive_frame = tk.Frame(col2, bg='#f0f0f0')
        consecutive_frame.pack(fill=tk.X, pady=5)
        
        self.consecutive_scale = tk.Scale(consecutive_frame, from_=1, to=10, resolution=1,
                                        variable=self.consecutive_threshold, orient=tk.HORIZONTAL,
                                        bg='#f0f0f0', fg='#333333', font=("Arial", 10),
                                        length=200)
        self.consecutive_scale.pack(side=tk.LEFT)
        
        self.consecutive_value = tk.Label(consecutive_frame, text="3", 
                                        font=("Arial", 11, "bold"), bg='#f0f0f0', fg='#FF9800')
        self.consecutive_value.pack(side=tk.LEFT, padx=(10, 0))
        
        # 第三列：警报冷却时间
        tk.Label(col3, text="警报冷却时间(秒)", font=("Arial", 11, "bold"), 
               bg='#f0f0f0', fg='#333333').pack(anchor=tk.W)
        
        cooldown_frame = tk.Frame(col3, bg='#f0f0f0')
        cooldown_frame.pack(fill=tk.X, pady=5)
        
        self.cooldown_scale = tk.Scale(cooldown_frame, from_=1.0, to=10.0, resolution=0.5,
                                     variable=self.alert_cooldown, orient=tk.HORIZONTAL,
                                     bg='#f0f0f0', fg='#333333', font=("Arial", 10),
                                     length=200)
        self.cooldown_scale.pack(side=tk.LEFT)
        
        self.cooldown_value = tk.Label(cooldown_frame, text="5.0", 
                                     font=("Arial", 11, "bold"), bg='#f0f0f0', fg='#9C27B0')
        self.cooldown_value.pack(side=tk.LEFT, padx=(10, 0))
        
        # 绑定阈值更新事件
        self.yawn_threshold.trace('w', self._update_threshold_display)
        self.consecutive_threshold.trace('w', self._update_consecutive_display)
        self.alert_cooldown.trace('w', self._update_cooldown_display)
    
    def _update_threshold_display(self, *args):
        """更新阈值显示"""
        value = self.yawn_threshold.get()
        self.threshold_value.config(text=f"{value:.2f}")
    
    def _update_consecutive_display(self, *args):
        """更新连续检测阈值显示"""
        value = self.consecutive_threshold.get()
        self.consecutive_value.config(text=str(value))
    
    def _update_cooldown_display(self, *args):
        """更新冷却时间显示"""
        value = self.alert_cooldown.get()
        self.cooldown_value.config(text=f"{value:.1f}")
    
    def _draw_face_landmarks(self, frame, face_rect, landmarks):
        """在人脸上绘制特征点和人脸框"""
        if face_rect is not None:
            # 绘制人脸框 - 绿色
            cv2.rectangle(frame, (face_rect.left(), face_rect.top()), 
                         (face_rect.right(), face_rect.bottom()), (0, 255, 0), 2)
        
        if landmarks is not None and face_rect is not None:
            # 绘制68个特征点
            for i, (x, y) in enumerate(landmarks):
                # 转换为实际坐标
                actual_x = int(x * (face_rect.right() - face_rect.left()) + face_rect.left())
                actual_y = int(y * (face_rect.bottom() - face_rect.top()) + face_rect.top())
                
                # 不同区域使用不同颜色
                if i < 17:  # 脸部轮廓 - 蓝色
                    color = (255, 0, 0)
                elif i < 27:  # 眉毛 - 绿色
                    color = (0, 255, 0)
                elif i < 36:  # 鼻子 - 红色
                    color = (0, 0, 255)
                elif i < 48:  # 眼睛 - 青色
                    color = (255, 255, 0)
                else:  # 嘴巴 - 紫色
                    color = (255, 0, 255)
                
                cv2.circle(frame, (actual_x, actual_y), 2, color, -1)
        
        return frame

    def _preprocess_frame(self, frame):
        """预处理帧"""
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
        """预测打哈欠"""
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

    def start_detection(self):
        """开始检测"""
        if self.model is None:
            messagebox.showerror("错误", "模型未加载")
            return

        # 初始化摄像头
        camera_id = int(self.camera_var.get())
        self.cap = cv2.VideoCapture(camera_id)

        if not self.cap.isOpened():
            messagebox.showerror("错误", f"无法打开摄像头 {camera_id}")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # 重置状态
        self.running = True
        self.session_start_time = time.time()
        self.yawn_count = 0
        self.total_predictions = 0
        self.consecutive_yawns = 0
        self.last_yawn_time = 0
        self.face_buffer.clear()
        self.landmark_buffer.clear()

        # 更新按钮状态
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        # 更新状态显示
        self.status_label.config(text="状态: 运行中", fg='#4CAF50')

        # 启动检测线程
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()

    def stop_detection(self):
        """停止检测"""
        self.running = False

        if self.cap:
            self.cap.release()

        # 更新按钮状态
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

        # 更新状态显示
        self.status_label.config(text="状态: 已停止", fg='#f44336')
        self.video_label.config(image='')

        # 清空警报
        self.alert_label.config(text="", bg='#f0f0f0')

    def _detection_loop(self):
        """检测循环"""
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # 翻转图像（镜像效果）
                frame = cv2.flip(frame, 1)

                # 预处理
                face_img, landmarks, face_rect = self._preprocess_frame(frame)
                face_detected = face_img is not None

                yawn_prob = 0.0
                prediction = 0

                if face_detected:
                    self.face_buffer.append(face_img)
                    self.landmark_buffer.append(landmarks)

                    if len(self.face_buffer) >= SEQUENCE_LENGTH:
                        yawn_prob, prediction = self._predict_yawn()
                        self.total_predictions += 1

                        # 更新连续检测计数
                        if prediction == 1:
                            self.consecutive_yawns += 1
                        else:
                            self.consecutive_yawns = 0

                        # 检查是否触发警报
                        current_time = time.time()
                        if (self.consecutive_yawns >= self.consecutive_threshold.get() and
                            (current_time - self.last_yawn_time) > self.alert_cooldown.get()):
                            self.yawn_count += 1
                            self.last_yawn_time = current_time
                            self.root.after(0, self._trigger_alert)

                    # 在人脸上绘制特征点和人脸框
                    frame = self._draw_face_landmarks(frame, face_rect, landmarks)

                # 更新GUI
                self.root.after(0, self._update_gui, frame, face_detected, yawn_prob, prediction)

                time.sleep(0.03)  # 控制帧率

            except Exception as e:
                print(f"检测循环错误: {e}")
                break

    def _update_gui(self, frame, face_detected, yawn_prob, prediction):
        """更新GUI显示"""
        try:
            # 更新视频显示
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (self.video_width, self.video_height))
            frame_pil = Image.fromarray(frame_resized)
            frame_tk = ImageTk.PhotoImage(frame_pil)
            self.video_label.config(image=frame_tk)
            self.video_label.image = frame_tk

            # 更新预测信息
            if face_detected:
                self.face_status.config(text="人脸检测: 成功", fg='#4CAF50')
            else:
                self.face_status.config(text="人脸检测: 失败", fg='#f44336')

            if len(self.face_buffer) >= SEQUENCE_LENGTH:
                self.prob_label.config(text=f"打哈欠概率: {yawn_prob:.3f}")

                if prediction == 1:
                    self.prediction_label.config(text="预测结果: 打哈欠", fg='#f44336')
                else:
                    self.prediction_label.config(text="预测结果: 正常", fg='#4CAF50')

                self.consecutive_label.config(text=f"连续检测: {self.consecutive_yawns}")
            else:
                self.prob_label.config(text="打哈欠概率: 等待中...")
                self.prediction_label.config(text="预测结果: 缓冲中", fg='#FF9800')
                self.consecutive_label.config(text="连续检测: 0")

            # 更新统计信息
            if self.session_start_time:
                session_time = time.time() - self.session_start_time
                minutes, seconds = divmod(int(session_time), 60)
                self.time_label.config(text=f"会话时间: {minutes:02d}:{seconds:02d}")

            self.buffer_label.config(text=f"缓冲区: {len(self.face_buffer)}/{SEQUENCE_LENGTH}")
            self.count_label.config(text=f"总检测次数: {self.total_predictions}")
            self.yawn_count_label.config(text=f"打哈欠次数: {self.yawn_count}")

        except Exception as e:
            print(f"GUI更新错误: {e}")

    def _trigger_alert(self):
        """触发警报"""
        self.alert_label.config(text="⚠️ 疲劳警报！", fg='white', bg='#f44336')

        # 3秒后清除警报显示
        self.root.after(3000, lambda: self.alert_label.config(text="", bg='#f0f0f0'))

    def run(self):
        """运行GUI"""
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.root.mainloop()

    def _on_closing(self):
        """关闭处理"""
        if self.running:
            self.stop_detection()
        self.root.destroy()

def main():
    """主函数"""
    model_path = os.path.join(MODEL_SAVE_PATH, 'best_model.pth')
    if not os.path.exists(model_path):
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("错误", f"模型文件不存在: {model_path}")
        return

    if not os.path.exists(DLIB_PREDICTOR_PATH):
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("错误", f"dlib文件不存在: {DLIB_PREDICTOR_PATH}")
        return

    try:
        app = FatigueDetectionGUI(model_path)
        app.run()
    except Exception as e:
        messagebox.showerror("错误", f"程序运行失败: {e}")

if __name__ == "__main__":
    main()
