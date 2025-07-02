import cv2
import dlib
import numpy as np
import torch
import torch.nn as nn
import time
import pygame  # For playing alarm sounds


class CNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(CNNLSTM, self).__init__()

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

        # Fully connected layers - Output classification results (consistent with training, Sigmoid removed)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_size)  # 移除Sigmoid
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


class RealTimeFatigueDetector:
    def __init__(self, model_path, predictor_path,
                 seq_length=30, ear_threshold=0.25, mar_threshold=0.7,
                 consecutive_frames=15):
        # 加载模型和面部检测器
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化模型（与训练时的结构保持一致）
        self.model = CNNLSTM(
            input_size=138,  # 2 (EAR, MAR) + 68*2 (landmark coordinates)
            hidden_size=64,  # 与训练时一致
            num_layers=1,    # 与训练时一致
            output_size=1
        ).to(self.device)

        # 加载模型权重
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

        # 模型参数
        self.seq_length = seq_length

        # 疲劳检测阈值
        self.ear_threshold = ear_threshold  # 眼睛纵横比阈值
        self.mar_threshold = mar_threshold  # 嘴巴纵横比阈值
        self.consecutive_frames = consecutive_frames  # 判定疲劳的连续帧数

        # 状态变量
        self.features_buffer = []  # 特征缓冲区
        self.frame_count = 0
        self.fatigue_frames = 0
        self.is_fatigued = False

        # 定义眼部和嘴部关键点索引
        self.left_eye_indices = list(range(36, 42))
        self.right_eye_indices = list(range(42, 48))
        self.mouth_indices = list(range(48, 68))

        # 初始化音频警报
        pygame.mixer.init()

    def calculate_eye_aspect_ratio(self, landmarks):
        """Calculate Eye Aspect Ratio (EAR)"""
        left_eye = landmarks[self.left_eye_indices]
        right_eye = landmarks[self.right_eye_indices]

        # 计算左眼EAR
        A = np.linalg.norm(left_eye[1] - left_eye[5])
        B = np.linalg.norm(left_eye[2] - left_eye[4])
        C = np.linalg.norm(left_eye[0] - left_eye[3])
        left_ear = (A + B) / (2.0 * C)

        # 计算右眼EAR
        A = np.linalg.norm(right_eye[1] - right_eye[5])
        B = np.linalg.norm(right_eye[2] - right_eye[4])
        C = np.linalg.norm(right_eye[0] - right_eye[3])
        right_ear = (A + B) / (2.0 * C)

        # 返回双眼平均EAR
        return (left_ear + right_ear) / 2.0

    def calculate_mouth_aspect_ratio(self, landmarks):
        """Calculate Mouth Aspect Ratio (MAR)"""
        mouth = landmarks[self.mouth_indices]

        # 计算垂直距离
        A = np.linalg.norm(mouth[2] - mouth[10])
        B = np.linalg.norm(mouth[4] - mouth[8])

        # 计算水平距离
        C = np.linalg.norm(mouth[0] - mouth[6])

        # 计算嘴巴纵横比
        mar = (A + B) / (2.0 * C)
        return mar

    def extract_features(self, frame):
        """Extract features from single frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if not faces:
            # If no face detected, return None
            return None

        # Take the largest face
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        shape = self.predictor(gray, face)
        landmarks = np.array([[part.x, part.y] for part in shape.parts()])

        # Calculate eye and mouth aspect ratios
        ear = self.calculate_eye_aspect_ratio(landmarks)
        mar = self.calculate_mouth_aspect_ratio(landmarks)

        # Normalize landmark coordinates
        nose = landmarks[30]  # Nose tip landmark
        normalized_landmarks = (landmarks - nose).flatten() / frame.shape[0]

        # Combine features
        features = np.concatenate([[ear, mar], normalized_landmarks])
        return features, ear, mar, landmarks

    def update_buffer(self, features):
        """Update feature buffer"""
        self.features_buffer.append(features)
        if len(self.features_buffer) > self.seq_length:
            self.features_buffer.pop(0)

    def detect_fatigue(self):
        """Detect fatigue based on current feature buffer"""
        if len(self.features_buffer) < self.seq_length:
            return False, 0.0

        # Prepare input sequence
        input_seq = np.array([self.features_buffer])
        input_tensor = torch.FloatTensor(input_seq).to(self.device)

        # Model prediction
        with torch.no_grad():
            logits = self.model(input_tensor).item()
            prediction = torch.sigmoid(torch.tensor(logits)).item()  # Convert logits to probability

        # Update fatigue state
        self.frame_count += 1
        if prediction >= 0.5:  # Assume model output >0.5 indicates fatigue
            self.fatigue_frames += 1
        else:
            self.fatigue_frames = 0

        # Determine if fatigued
        self.is_fatigued = self.fatigue_frames >= self.consecutive_frames

        return self.is_fatigued, prediction

    def draw_landmarks(self, frame, landmarks):
        """Draw facial landmarks on image"""
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        return frame

    def play_alarm(self):
        """Play alarm sound"""
        try:
            pygame.mixer.music.load("alarm.wav")  # Need to prepare an alarm sound file
            pygame.mixer.music.play()
        except:
            # If unable to play sound, print warning
            print("WARNING: Unable to play alarm sound")

    def draw_enhanced_info(self, frame, ear, mar, fatigue_score, is_fatigued):
        """Draw enhanced information overlay on frame"""
        height, width = frame.shape[:2]

        # Create semi-transparent background for text
        overlay = frame.copy()

        # Main info panel (top-left)
        cv2.rectangle(overlay, (10, 10), (450, 180), (0, 0, 0), -1)

        # Status panel (top-right)
        status_width = 300
        cv2.rectangle(overlay, (width - status_width - 10, 10), (width - 10, 120), (0, 0, 0), -1)

        # Apply transparency
        alpha = 0.7
        frame = cv2.addWeighted(frame, alpha, overlay, 1 - alpha, 0)

        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale_large = 0.8
        font_scale_medium = 0.7
        font_scale_small = 0.6
        thickness_bold = 3
        thickness_normal = 2

        # Color scheme
        color_normal = (0, 255, 0)  # Green
        color_warning = (0, 165, 255)  # Orange
        color_danger = (0, 0, 255)  # Red
        color_white = (255, 255, 255)
        color_cyan = (255, 255, 0)  # Cyan

        # Main metrics display
        y_offset = 40
        line_spacing = 35

        # EAR display with threshold indication
        ear_color = color_danger if ear < 0.25 else color_normal
        cv2.putText(frame, f"EAR: {ear:.3f}", (20, y_offset),
                   font, font_scale_large, ear_color, thickness_bold)
        cv2.putText(frame, "(Eye Aspect Ratio)", (20, y_offset + 20),
                   font, font_scale_small, color_white, 1)

        # MAR display with threshold indication
        y_offset += line_spacing + 20
        mar_color = color_warning if mar > 0.7 else color_normal
        cv2.putText(frame, f"MAR: {mar:.3f}", (20, y_offset),
                   font, font_scale_large, mar_color, thickness_bold)
        cv2.putText(frame, "(Mouth Aspect Ratio)", (20, y_offset + 20),
                   font, font_scale_small, color_white, 1)

        # Fatigue score with color coding
        y_offset += line_spacing + 20
        score_color = color_danger if fatigue_score > 0.7 else (color_warning if fatigue_score > 0.4 else color_normal)
        cv2.putText(frame, f"Fatigue Score: {fatigue_score:.3f}", (20, y_offset),
                   font, font_scale_large, score_color, thickness_bold)

        # Buffer status
        y_offset += line_spacing
        buffer_status = f"Buffer: {len(self.features_buffer)}/{self.seq_length}"
        cv2.putText(frame, buffer_status, (20, y_offset),
                   font, font_scale_medium, color_cyan, thickness_normal)

        # Status panel (top-right)
        status_x = width - status_width + 10
        status_y = 45

        # Current status
        if is_fatigued:
            status_text = "FATIGUE DETECTED!"
            status_color = color_danger
            # Add blinking effect
            if int(time.time() * 3) % 2:  # Blink every ~0.33 seconds
                cv2.putText(frame, status_text, (status_x, status_y),
                           font, font_scale_large, status_color, thickness_bold)
        else:
            status_text = "NORMAL STATE"
            status_color = color_normal
            cv2.putText(frame, status_text, (status_x, status_y),
                       font, font_scale_large, status_color, thickness_bold)

        # Frame counter
        cv2.putText(frame, f"Frame: {self.frame_count}", (status_x, status_y + 35),
                   font, font_scale_medium, color_white, thickness_normal)

        # Fatigue frame counter
        if self.fatigue_frames > 0:
            cv2.putText(frame, f"Fatigue Frames: {self.fatigue_frames}/{self.consecutive_frames}",
                       (status_x, status_y + 65), font, font_scale_small, color_warning, thickness_normal)

        # Warning border for fatigue
        if is_fatigued:
            # Thick red border
            border_thickness = 8
            cv2.rectangle(frame, (0, 0), (width - 1, height - 1), color_danger, border_thickness)

            # Additional warning text at bottom
            warning_text = ">>> DRIVER FATIGUE ALERT <<<"
            text_size = cv2.getTextSize(warning_text, font, font_scale_large, thickness_bold)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height - 30

            # Background for warning text
            cv2.rectangle(frame, (text_x - 10, text_y - 35), (text_x + text_size[0] + 10, text_y + 10),
                         color_danger, -1)
            cv2.putText(frame, warning_text, (text_x, text_y),
                       font, font_scale_large, color_white, thickness_bold)

        # Instructions at bottom-left
        instructions = [
            "Press ESC or 'q' to quit",
            "Ensure good lighting",
            "Keep face visible"
        ]

        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (20, height - 80 + i * 25),
                       font, font_scale_small, color_white, 1)

        return frame

    def run(self, camera_index=0, output_path=None):
        """Run real-time fatigue detection"""
        cap = cv2.VideoCapture(camera_index)

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Set output video if path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Display window settings
        window_name = "Real-time Fatigue Detection System"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1200, 800)

        # Start processing video stream
        print("🎬 Camera started - processing video stream...")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Failed to read frame from camera")
                break

            # Flip image horizontally for mirror effect (more intuitive)
            frame = cv2.flip(frame, 1)

            # Extract features
            features = self.extract_features(frame)
            if features is not None:
                features_vector, ear, mar, landmarks = features

                # Update feature buffer
                self.update_buffer(features_vector)

                # Detect fatigue
                is_fatigued, fatigue_score = self.detect_fatigue()

                # Draw facial landmarks on image
                frame = self.draw_landmarks(frame, landmarks)

                # Draw enhanced information overlay
                frame = self.draw_enhanced_info(frame, ear, mar, fatigue_score, is_fatigued)

                # Play alarm if fatigue detected
                if is_fatigued:
                    self.play_alarm()
            else:
                # No face detected - show clear message
                height, width = frame.shape[:2]

                # Semi-transparent background
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, height//2 - 60), (width, height//2 + 60), (0, 0, 0), -1)
                frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

                # No face message
                no_face_text = "NO FACE DETECTED"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.2
                thickness = 3
                text_size = cv2.getTextSize(no_face_text, font, font_scale, thickness)[0]
                text_x = (width - text_size[0]) // 2
                text_y = height // 2

                cv2.putText(frame, no_face_text, (text_x, text_y),
                           font, font_scale, (0, 0, 255), thickness)

                # Instructions
                instruction_text = "Please position your face in front of the camera"
                font_scale_small = 0.7
                text_size_small = cv2.getTextSize(instruction_text, font, font_scale_small, 2)[0]
                text_x_small = (width - text_size_small[0]) // 2
                cv2.putText(frame, instruction_text, (text_x_small, text_y + 40),
                           font, font_scale_small, (255, 255, 255), 2)

            # Display processed frame
            cv2.imshow(window_name, frame)

            # Save output video
            if output_path:
                out.write(frame)

            # Check for exit keys (ESC or 'q')
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or 'q'
                break

        # Release resources
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()


# Usage example
if __name__ == "__main__":
    # Model and dlib predictor paths
    model_path = "models/best_fatigue_model.pth"  # Use best model
    predictor_path = "shape_predictor_68_face_landmarks.dat"

    print("=" * 60)
    print("🚗 REAL-TIME FATIGUE DETECTION SYSTEM")
    print("=" * 60)
    print("📹 Initializing camera and loading models...")
    print("⚠️  Make sure your face is clearly visible")
    print("🔊 Audio alerts will play when fatigue is detected")
    print("⌨️  Press ESC or 'q' to exit")
    print("=" * 60)

    try:
        # Initialize detector
        detector = RealTimeFatigueDetector(
            model_path=model_path,
            predictor_path=predictor_path,
            seq_length=30,
            ear_threshold=0.25,
            mar_threshold=0.7,
            consecutive_frames=15
        )

        print("✅ System initialized successfully!")
        print("🎥 Starting real-time detection...")

        # Run real-time detection (using default camera)
        detector.run(camera_index=0, output_path=None)

    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Please check:")
        print("   - Camera is connected and accessible")
        print("   - Model files exist in the models/ directory")
        print("   - shape_predictor_68_face_landmarks.dat is present")

    print("\n👋 Fatigue detection system stopped.")
    print("🛡️ Drive safely!")