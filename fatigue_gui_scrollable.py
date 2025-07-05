"""
疲劳检测GUI界面 - 带滚动功能的版本
改进：
- 右侧面板添加滚动条
- 调整布局以适应更多内容
- 方格更大，信息更清晰
- 确保所有按钮都可见
"""
import tkinter as tk
from tkinter import ttk, messagebox
import time

class FatigueDetectionScrollableGUI:
    def __init__(self):
        """疲劳检测GUI - 带滚动功能的演示版本"""
        
        # 初始化主窗口
        self.root = tk.Tk()
        self.root.title("疲劳驾驶检测系统 - 滚动版本")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        self.root.resizable(True, True)
        
        # 模拟状态变量
        self.current_mode = "balanced"
        self.yawn_count = 0
        self.blink_count = 0
        self.total_predictions = 0
        self.consecutive_yawns = 0
        self.consecutive_threshold = 30
        self.session_start_time = time.time()
        
        # 创建GUI
        self._create_gui()
    
    def _create_gui(self):
        """创建GUI界面 - 带滚动功能的版本"""
        # 创建主框架
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 左侧视频区域
        left_frame = tk.Frame(main_frame, bg='#f0f0f0')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20))
        
        # 视频显示区域
        video_frame = tk.LabelFrame(left_frame, text="视频预览", font=("Arial", 12, "bold"),
                                  bg='#f0f0f0', fg='#333333', padx=10, pady=10)
        video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        self.video_label = tk.Label(video_frame, text="摄像头画面\n640x480\n(演示模式)", 
                                   font=("Arial", 16), bg='#333333', fg='white')
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 控制按钮区域
        btn_frame = tk.Frame(left_frame, bg='#f0f0f0')
        btn_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.start_btn = tk.Button(btn_frame, text="开始检测", command=self._start_detection,
                                 bg='#4CAF50', fg='white', font=("Arial", 12, "bold"),
                                 padx=20, pady=10)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_btn = tk.Button(btn_frame, text="停止检测", command=self._stop_detection,
                                bg='#f44336', fg='white', font=("Arial", 12, "bold"),
                                padx=20, pady=10, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)
        
        # 当前设置显示
        current_settings_frame = tk.LabelFrame(left_frame, text="当前设置", font=("Arial", 10, "bold"),
                                             bg='#f0f0f0', fg='#333333', padx=10, pady=5)
        current_settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.current_mode_label = tk.Label(current_settings_frame, text="当前模式: ⚖️ 平衡模式", 
                                         font=("Arial", 11, "bold"), bg='#f0f0f0', fg='#4CAF50')
        self.current_mode_label.pack(anchor=tk.W, pady=2)
        
        self.current_params_label = tk.Label(current_settings_frame, 
                                           text="模型阈值: 0.60 | MAR阈值: 0.60 | 连续阈值: 30帧 | 冷却: 5.0秒", 
                                           font=("Arial", 9), bg='#f0f0f0', fg='#666666')
        self.current_params_label.pack(anchor=tk.W)
        
        # 快速预设按钮
        preset_frame = tk.LabelFrame(left_frame, text="快速预设", font=("Arial", 10, "bold"),
                                   bg='#f0f0f0', fg='#333333', padx=10, pady=5)
        preset_frame.pack(fill=tk.X)
        
        button_frame = tk.Frame(preset_frame, bg='#f0f0f0')
        button_frame.pack(fill=tk.X, pady=5)
        
        # 预设按钮
        sensitive_btn = tk.Button(button_frame, text="🔥 敏感模式", 
                                command=lambda: self._apply_preset('sensitive'),
                                bg='#FF5722', fg='white', font=("Arial", 10, "bold"),
                                relief=tk.RAISED, bd=2, padx=15, pady=5)
        sensitive_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        balanced_btn = tk.Button(button_frame, text="⚖️ 平衡模式", 
                               command=lambda: self._apply_preset('balanced'),
                               bg='#4CAF50', fg='white', font=("Arial", 10, "bold"),
                               relief=tk.RAISED, bd=2, padx=15, pady=5)
        balanced_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        conservative_btn = tk.Button(button_frame, text="🛡️ 保守模式", 
                                   command=lambda: self._apply_preset('conservative'),
                                   bg='#2196F3', fg='white', font=("Arial", 10, "bold"),
                                   relief=tk.RAISED, bd=2, padx=15, pady=5)
        conservative_btn.pack(side=tk.LEFT)
        
        # 右侧滚动区域
        self._create_scrollable_right_panel(main_frame)

    def _create_scrollable_right_panel(self, parent):
        """创建带滚动条的右侧面板"""
        # 右侧主框架
        right_main_frame = tk.Frame(parent, bg='#f0f0f0', width=450)
        right_main_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        right_main_frame.pack_propagate(False)
        
        # 创建Canvas和Scrollbar
        canvas = tk.Canvas(right_main_frame, bg='#f0f0f0', highlightthickness=0)
        scrollbar = ttk.Scrollbar(right_main_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas, bg='#f0f0f0')
        
        # 配置滚动
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 布局Canvas和Scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 鼠标滚轮绑定
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # 在滚动框架中创建内容
        self._create_right_panel_content(self.scrollable_frame)

    def _create_right_panel_content(self, parent):
        """创建右侧面板的内容"""
        # 实时监测区域（3x2方格布局）
        result_frame = tk.LabelFrame(parent, text="实时监测", font=("Arial", 12, "bold"),
                                   bg='#f0f0f0', fg='#333333', padx=15, pady=15)
        result_frame.pack(fill=tk.X, pady=(0, 15), padx=10)
        
        # 创建3x2的方格布局
        grid_frame = tk.Frame(result_frame, bg='#f0f0f0')
        grid_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 配置网格权重 - 更大的方格
        for i in range(3):
            grid_frame.columnconfigure(i, weight=1, minsize=120)
        for i in range(2):
            grid_frame.rowconfigure(i, weight=1, minsize=80)
        
        # 人脸检测状态方格
        face_frame = tk.Frame(grid_frame, bg='#ffffff', relief=tk.RAISED, bd=2)
        face_frame.grid(row=0, column=0, padx=4, pady=4, sticky='nsew')
        tk.Label(face_frame, text="人脸检测", font=("Arial", 10, "bold"), 
               bg='#ffffff', fg='#333333').pack(pady=(10, 3))
        self.face_status = tk.Label(face_frame, text="成功", 
                                  font=("Arial", 12, "bold"), bg='#ffffff', fg='#4CAF50')
        self.face_status.pack(pady=(0, 10))
        
        # 打哈欠概率方格
        prob_frame = tk.Frame(grid_frame, bg='#e8f5e8', relief=tk.RAISED, bd=2)
        prob_frame.grid(row=0, column=1, padx=4, pady=4, sticky='nsew')
        tk.Label(prob_frame, text="打哈欠概率", font=("Arial", 10, "bold"), 
               bg='#e8f5e8', fg='#2e7d32').pack(pady=(10, 3))
        self.prob_status = tk.Label(prob_frame, text="0.750", 
                                  font=("Arial", 12, "bold"), bg='#e8f5e8', fg='#f44336')
        self.prob_status.pack(pady=(0, 10))
        
        # 嘴部状态方格
        mouth_frame = tk.Frame(grid_frame, bg='#fff3e0', relief=tk.RAISED, bd=2)
        mouth_frame.grid(row=0, column=2, padx=4, pady=4, sticky='nsew')
        tk.Label(mouth_frame, text="嘴部状态", font=("Arial", 10, "bold"), 
               bg='#fff3e0', fg='#e65100').pack(pady=(10, 3))
        self.mouth_status = tk.Label(mouth_frame, text="张开", 
                                   font=("Arial", 12, "bold"), bg='#fff3e0', fg='#FF9800')
        self.mouth_status.pack(pady=(0, 10))
        
        # 眼部状态方格
        eye_frame = tk.Frame(grid_frame, bg='#e3f2fd', relief=tk.RAISED, bd=2)
        eye_frame.grid(row=1, column=0, padx=4, pady=4, sticky='nsew')
        tk.Label(eye_frame, text="眼部状态", font=("Arial", 10, "bold"), 
               bg='#e3f2fd', fg='#1565c0').pack(pady=(10, 3))
        self.eye_status = tk.Label(eye_frame, text="正常", 
                                 font=("Arial", 12, "bold"), bg='#e3f2fd', fg='#4CAF50')
        self.eye_status.pack(pady=(0, 10))
        
        # 疲劳状态方格
        fatigue_frame = tk.Frame(grid_frame, bg='#fce4ec', relief=tk.RAISED, bd=2)
        fatigue_frame.grid(row=1, column=1, padx=4, pady=4, sticky='nsew')
        tk.Label(fatigue_frame, text="疲劳状态", font=("Arial", 10, "bold"), 
               bg='#fce4ec', fg='#ad1457').pack(pady=(10, 3))
        self.fatigue_status = tk.Label(fatigue_frame, text="轻度疲劳", 
                                     font=("Arial", 12, "bold"), bg='#fce4ec', fg='#FFC107')
        self.fatigue_status.pack(pady=(0, 10))
        
        # 连续检测方格
        consecutive_frame = tk.Frame(grid_frame, bg='#f3e5f5', relief=tk.RAISED, bd=2)
        consecutive_frame.grid(row=1, column=2, padx=4, pady=4, sticky='nsew')
        tk.Label(consecutive_frame, text="连续检测", font=("Arial", 10, "bold"), 
               bg='#f3e5f5', fg='#6a1b9a').pack(pady=(10, 3))
        self.consecutive_label = tk.Label(consecutive_frame, text="18/30",
                                        font=("Arial", 12, "bold"), bg='#f3e5f5', fg='#666666')
        self.consecutive_label.pack(pady=(0, 10))
        
        # 进度条
        progress_frame = tk.Frame(result_frame, bg='#f0f0f0')
        progress_frame.pack(fill=tk.X, padx=5, pady=(0, 10))
        
        tk.Label(progress_frame, text="检测进度:", font=("Arial", 10), 
               bg='#f0f0f0', fg='#666666').pack(side=tk.LEFT)
        
        self.progress_bar = ttk.Progressbar(progress_frame, length=300, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)
        self.progress_bar['value'] = 60  # 演示进度
