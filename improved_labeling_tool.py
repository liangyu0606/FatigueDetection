#!/usr/bin/env python3
"""
改进的视频标注工具 - 增强进度条功能
专注于精确的哈欠标记和进度条拖动
"""

import cv2
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time
import json


class ImprovedLabelingTool:
    def __init__(self, root):
        self.root = root
        self.root.title("改进的视频标注工具 - 精确进度控制")
        self.root.geometry("1200x800")
        
        # Video properties
        self.cap = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        self.is_playing = False
        self.play_speed = 1.0
        
        # Labeling data
        self.label_type = 0  # 0=Normal, 1=Talking, 2=Yawning
        self.yawn_ranges = []
        self.temp_yawn_start = -1  # 临时哈欠开始帧
        self.temp_yawn_end = -1    # 临时哈欠结束帧
        self.is_marking_mode = False  # 是否在标记模式
        
        # Current video and folder management
        self.current_video_path = ""
        self.video_name = ""
        self.labels_file = ""
        self.video_folder = ""
        self.video_files = []
        self.current_video_index = 0
        
        # Progress bar enhancement
        self.is_dragging = False
        self.last_progress_update = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        """设置用户界面"""
        # 顶部控制面板
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 文件操作
        file_frame = ttk.Frame(top_frame)
        file_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(file_frame, text="📁 打开视频", command=self.open_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="📂 打开文件夹", command=self.open_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="💾 保存标注", command=self.save_labels).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="📋 加载标注", command=self.load_labels).pack(side=tk.LEFT, padx=5)
        
        # 视频信息和导航
        video_nav_frame = ttk.Frame(file_frame)
        video_nav_frame.pack(side=tk.LEFT, padx=20)

        self.video_info_var = tk.StringVar(value="未加载视频")
        ttk.Label(video_nav_frame, textvariable=self.video_info_var, font=('Arial', 10, 'bold')).pack()

        # 视频导航按钮
        nav_buttons_frame = ttk.Frame(video_nav_frame)
        nav_buttons_frame.pack(pady=(5, 0))

        ttk.Button(nav_buttons_frame, text="⬆️ 上一个", command=self.previous_video).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_buttons_frame, text="⬇️ 下一个", command=self.next_video).pack(side=tk.LEFT, padx=2)

        # 文件夹信息
        self.folder_info_var = tk.StringVar(value="")
        ttk.Label(video_nav_frame, textvariable=self.folder_info_var, font=('Arial', 9), foreground='blue').pack()
        
        # 主要内容区域
        content_frame = ttk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 左侧视频显示
        video_frame = ttk.LabelFrame(content_frame, text="视频显示", padding=10)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.video_canvas = tk.Canvas(video_frame, bg='black', width=640, height=480)
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 右侧控制面板
        control_panel = ttk.Frame(content_frame)
        control_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # 标签选择
        label_frame = ttk.LabelFrame(control_panel, text="标签类型", padding=10)
        label_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.label_var = tk.IntVar(value=0)
        ttk.Radiobutton(label_frame, text="🟢 正常 (0)", variable=self.label_var, value=0,
                       command=self.update_label_type).pack(anchor=tk.W)
        ttk.Radiobutton(label_frame, text="🟡 说话 (1)", variable=self.label_var, value=1,
                       command=self.update_label_type).pack(anchor=tk.W)
        ttk.Radiobutton(label_frame, text="🔴 打哈欠 (2)", variable=self.label_var, value=2,
                       command=self.update_label_type).pack(anchor=tk.W)
        
        # 哈欠标记控制
        yawn_frame = ttk.LabelFrame(control_panel, text="哈欠标记", padding=10)
        yawn_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 标记模式切换
        self.mark_mode_var = tk.BooleanVar()
        self.mark_mode_check = ttk.Checkbutton(yawn_frame, text="🎯 标记模式", 
                                              variable=self.mark_mode_var,
                                              command=self.toggle_marking_mode)
        self.mark_mode_check.pack(anchor=tk.W, pady=(0, 5))
        
        # 快速标记按钮
        mark_buttons_frame = ttk.Frame(yawn_frame)
        mark_buttons_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(mark_buttons_frame, text="📍 设为开始", 
                  command=self.set_yawn_start).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(mark_buttons_frame, text="🏁 设为结束", 
                  command=self.set_yawn_end).pack(side=tk.LEFT)
        
        # 当前标记状态
        self.mark_status_var = tk.StringVar(value="未标记")
        ttk.Label(yawn_frame, textvariable=self.mark_status_var, 
                 font=('Arial', 9), foreground='blue').pack(anchor=tk.W, pady=(5, 0))
        
        # 添加/清除按钮
        action_frame = ttk.Frame(yawn_frame)
        action_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(action_frame, text="✅ 添加范围", 
                  command=self.add_yawn_range).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(action_frame, text="🗑️ 清除全部", 
                  command=self.clear_all_yawns).pack(side=tk.LEFT)
        
        # 哈欠范围列表
        list_frame = ttk.LabelFrame(yawn_frame, text="哈欠范围", padding=5)
        list_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.yawn_listbox = tk.Listbox(list_frame, height=6, font=('Arial', 9))
        self.yawn_listbox.pack(fill=tk.X)
        self.yawn_listbox.bind('<Double-Button-1>', self.jump_to_yawn)
        self.yawn_listbox.bind('<Button-3>', self.delete_yawn_range)  # 右键删除
        
        # 帧信息
        info_frame = ttk.LabelFrame(control_panel, text="帧信息", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.frame_info_var = tk.StringVar(value="帧: 0/0")
        ttk.Label(info_frame, textvariable=self.frame_info_var).pack()
        
        self.time_info_var = tk.StringVar(value="时间: 00:00/00:00")
        ttk.Label(info_frame, textvariable=self.time_info_var).pack()
        
        # 底部播放控制
        self.setup_playback_controls()
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪 - 请打开视频文件")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, pady=(5, 0))
        
    def setup_playback_controls(self):
        """设置播放控制"""
        bottom_frame = ttk.LabelFrame(self.root, text="播放控制", padding=10)
        bottom_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # 播放按钮
        button_frame = ttk.Frame(bottom_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(button_frame, text="⏪", command=lambda: self.seek_frame(-30)).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="⏮️", command=lambda: self.seek_frame(-10)).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="◀️", command=lambda: self.seek_frame(-1)).pack(side=tk.LEFT, padx=2)
        
        self.play_button = ttk.Button(button_frame, text="▶️", command=self.toggle_playback)
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="▶️", command=lambda: self.seek_frame(1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="⏭️", command=lambda: self.seek_frame(10)).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="⏩", command=lambda: self.seek_frame(30)).pack(side=tk.LEFT, padx=2)
        
        # 速度控制
        speed_frame = ttk.Frame(button_frame)
        speed_frame.pack(side=tk.RIGHT)
        
        ttk.Label(speed_frame, text="速度:").pack(side=tk.LEFT, padx=(10, 5))
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_combo = ttk.Combobox(speed_frame, textvariable=self.speed_var, 
                                  values=[0.1, 0.25, 0.5, 1.0, 1.5, 2.0], width=8)
        speed_combo.pack(side=tk.LEFT)
        speed_combo.bind('<<ComboboxSelected>>', self.update_play_speed)
        
        # 增强的进度条
        progress_frame = ttk.LabelFrame(bottom_frame, text="视频进度 (拖动定位帧)", padding=5)
        progress_frame.pack(fill=tk.X)
        
        # 进度信息
        progress_info_frame = ttk.Frame(progress_frame)
        progress_info_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.progress_info_var = tk.StringVar(value="拖动进度条精确定位哈欠开始和结束帧")
        ttk.Label(progress_info_frame, textvariable=self.progress_info_var, 
                 font=('Arial', 9), foreground='green').pack()
        
        # 进度条容器
        progress_container = ttk.Frame(progress_frame)
        progress_container.pack(fill=tk.X, pady=(0, 5))

        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Scale(progress_container, from_=0, to=100,
                                     variable=self.progress_var, orient=tk.HORIZONTAL,
                                     command=self.on_progress_change)
        self.progress_bar.pack(fill=tk.X)

        # 哈欠范围标记画布（覆盖在进度条上方）
        self.range_canvas = tk.Canvas(progress_container, height=8, bg='white', highlightthickness=0)
        self.range_canvas.pack(fill=tk.X, pady=(2, 0))

        # 绑定进度条事件
        self.progress_bar.bind('<Button-1>', self.on_progress_click)
        self.progress_bar.bind('<B1-Motion>', self.on_progress_drag)
        self.progress_bar.bind('<ButtonRelease-1>', self.on_progress_release)

        # 绑定画布事件（让画布也能控制进度）
        self.range_canvas.bind('<Button-1>', self.on_canvas_click)
        self.range_canvas.bind('<B1-Motion>', self.on_canvas_drag)
        self.range_canvas.bind('<ButtonRelease-1>', self.on_canvas_release)
        
        # 进度条下方的帧数显示
        frame_display_frame = ttk.Frame(progress_frame)
        frame_display_frame.pack(fill=tk.X)
        
        ttk.Label(frame_display_frame, text="0", font=('Arial', 8)).pack(side=tk.LEFT)
        self.current_frame_label = ttk.Label(frame_display_frame, text="当前帧: 0", 
                                           font=('Arial', 9, 'bold'), foreground='red')
        self.current_frame_label.pack(side=tk.LEFT, expand=True)
        
        self.total_frame_label = ttk.Label(frame_display_frame, text="0", font=('Arial', 8))
        self.total_frame_label.pack(side=tk.RIGHT)

    # 文件操作方法
    def open_video(self):
        """打开单个视频文件"""
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("所有文件", "*.*")
            ]
        )

        if not file_path:
            return

        # 清空文件夹模式
        self.video_folder = ""
        self.video_files = []
        self.current_video_index = 0
        self.folder_info_var.set("")

        self.load_video(file_path)

    def open_folder(self):
        """打开视频文件夹"""
        folder_path = filedialog.askdirectory(title="选择包含视频的文件夹")

        if not folder_path:
            return

        # 查找文件夹中的所有视频文件
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.MP4', '.AVI', '.MOV', '.MKV', '.WMV')
        self.video_files = []

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(video_extensions):
                    self.video_files.append(os.path.join(root, file))

        if not self.video_files:
            messagebox.showwarning("无视频文件", "在选择的文件夹中未找到视频文件")
            return

        # 按文件名排序
        self.video_files.sort()
        self.video_folder = folder_path
        self.current_video_index = 0

        # 设置标注文件路径
        self.labels_file = os.path.join(folder_path, "labels.txt")

        # 加载第一个视频
        self.load_video(self.video_files[0])

        # 更新文件夹信息
        self.update_folder_info()

        self.status_var.set(f"已加载文件夹: {len(self.video_files)} 个视频文件")

    def previous_video(self):
        """切换到上一个视频"""
        if not self.video_files:
            messagebox.showinfo("提示", "请先打开视频文件夹")
            return

        if self.current_video_index <= 0:
            messagebox.showinfo("提示", "已经是第一个视频")
            return

        # 保存当前视频的标注
        self.save_current_video_labels()

        # 切换到上一个视频
        self.current_video_index -= 1
        self.load_video(self.video_files[self.current_video_index])
        self.update_folder_info()

    def next_video(self):
        """切换到下一个视频"""
        if not self.video_files:
            messagebox.showinfo("提示", "请先打开视频文件夹")
            return

        if self.current_video_index >= len(self.video_files) - 1:
            messagebox.showinfo("提示", "已经是最后一个视频")
            return

        # 保存当前视频的标注
        self.save_current_video_labels()

        # 切换到下一个视频
        self.current_video_index += 1
        self.load_video(self.video_files[self.current_video_index])
        self.update_folder_info()

    def update_folder_info(self):
        """更新文件夹信息显示"""
        if self.video_files:
            info_text = f"文件夹: {self.current_video_index + 1}/{len(self.video_files)}"
            self.folder_info_var.set(info_text)
        else:
            self.folder_info_var.set("")

    def load_video(self, video_path):
        """加载视频文件"""
        if self.cap:
            self.cap.release()

        self.current_video_path = video_path
        self.video_name = os.path.basename(video_path)
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            messagebox.showerror("错误", f"无法打开视频: {video_path}")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.current_frame = 0

        # 重置标注数据
        self.label_type = 0
        self.yawn_ranges = []
        self.temp_yawn_start = -1
        self.temp_yawn_end = -1
        self.is_marking_mode = False

        # 自动检测标签
        self.auto_detect_label()

        # 如果是文件夹模式，尝试加载现有标注
        if self.video_folder and self.labels_file:
            self.load_labels_for_current_video()

        # 更新UI
        self.label_var.set(self.label_type)
        self.mark_mode_var.set(False)
        self.update_yawn_listbox()
        self.update_display()
        self.update_progress_labels()

        # 更新视频信息显示
        if self.video_files:
            self.video_info_var.set(f"视频: {self.video_name}")
        else:
            self.video_info_var.set(f"视频: {self.video_name}")

        self.status_var.set(f"已加载: {self.video_name} ({self.total_frames} 帧, {self.fps:.1f} FPS)")

    def auto_detect_label(self):
        """从文件名自动检测标签"""
        filename_lower = self.video_name.lower()
        if "yawn" in filename_lower or "哈欠" in filename_lower:
            self.label_type = 2
        elif "talk" in filename_lower or "说话" in filename_lower:
            self.label_type = 1
        else:
            self.label_type = 0

    def save_labels(self):
        """保存标注到文件"""
        if not self.current_video_path:
            messagebox.showwarning("无视频", "请先加载视频文件")
            return

        # 如果是文件夹模式，直接保存到文件夹的labels.txt
        if self.video_folder:
            self.save_current_video_labels()
            messagebox.showinfo("成功", f"标注已保存到 {self.labels_file}")
            return

        # 单文件模式，询问保存位置
        if not self.labels_file:
            self.labels_file = filedialog.asksaveasfilename(
                title="保存标注文件",
                defaultextension=".txt",
                filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")],
                initialfile="labels.txt"
            )

        if not self.labels_file:
            return

        self.save_current_video_labels()
        messagebox.showinfo("成功", f"标注已保存到 {self.labels_file}")

    def save_current_video_labels(self):
        """保存当前视频的标注数据"""
        if not self.current_video_path or not self.labels_file:
            return

        try:
            # 获取当前标签类型
            self.label_type = self.label_var.get()

            # 格式化哈欠范围
            yawn_str = "-1,-1"
            if self.yawn_ranges:
                yawn_str = ",".join([f"{start}-{end}" for start, end in self.yawn_ranges])

            # 读取现有标注（如果有）
            existing_labels = {}
            if os.path.exists(self.labels_file):
                with open(self.labels_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            existing_labels[parts[0]] = line.strip()

            # 更新当前视频的标注
            new_line = f"{self.video_name} {self.label_type} {yawn_str}"
            existing_labels[self.video_name] = new_line

            # 写入文件
            with open(self.labels_file, 'w', encoding='utf-8') as f:
                for video_name in sorted(existing_labels.keys()):
                    f.write(existing_labels[video_name] + '\n')

            self.status_var.set(f"已保存: {self.video_name} {self.label_type} {yawn_str}")

        except Exception as e:
            messagebox.showerror("错误", f"保存标注失败: {e}")

    def load_labels(self):
        """加载标注文件"""
        file_path = filedialog.askopenfilename(
            title="选择标注文件",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )

        if not file_path:
            return

        self.labels_file = file_path

        if not self.current_video_path:
            messagebox.showinfo("提示", "请先加载视频文件，然后标注数据将自动应用")
            return

        self.load_labels_for_current_video()

    def load_labels_for_current_video(self):
        """为当前视频加载标注数据"""
        if not self.labels_file or not os.path.exists(self.labels_file):
            return

        try:
            with open(self.labels_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3 and parts[0] == self.video_name:
                        # 加载标签类型
                        self.label_type = int(parts[1])
                        self.label_var.set(self.label_type)

                        # 加载哈欠范围
                        yawn_str = parts[2]
                        self.yawn_ranges = []
                        if yawn_str != "-1,-1":
                            yawn_pairs = yawn_str.split(',')
                            for pair in yawn_pairs:
                                if '-' in pair:
                                    start, end = map(int, pair.split('-'))
                                    self.yawn_ranges.append((start, end))

                        self.update_yawn_listbox()
                        self.update_display()
                        self.status_var.set(f"已加载标注: {self.video_name}")
                        break
        except Exception as e:
            messagebox.showerror("错误", f"加载标注失败: {e}")

    # 进度条控制方法
    def on_progress_click(self, event):
        """进度条点击事件"""
        self.is_dragging = True
        self.is_playing = False  # 停止播放
        self.play_button.config(text="▶️")

    def on_progress_drag(self, event):
        """进度条拖动事件"""
        if self.is_dragging and self.cap:
            current_time = time.time()
            # 限制更新频率，避免过于频繁
            if current_time - self.last_progress_update > 0.05:  # 50ms
                progress = self.progress_var.get()
                new_frame = int((progress / 100) * self.total_frames)
                self.current_frame = max(0, min(self.total_frames - 1, new_frame))
                self.update_display()
                self.last_progress_update = current_time

    def on_progress_release(self, event):
        """进度条释放事件"""
        self.is_dragging = False
        if self.cap:
            progress = self.progress_var.get()
            new_frame = int((progress / 100) * self.total_frames)
            self.current_frame = max(0, min(self.total_frames - 1, new_frame))
            self.update_display()

    def on_progress_change(self, value):
        """进度条值变化事件"""
        if not self.is_dragging and self.cap:
            progress = float(value)
            new_frame = int((progress / 100) * self.total_frames)
            self.current_frame = max(0, min(self.total_frames - 1, new_frame))
            self.update_display()

    # 画布事件处理（让画布也能控制进度）
    def on_canvas_click(self, event):
        """画布点击事件"""
        self.on_progress_click_from_canvas(event)

    def on_canvas_drag(self, event):
        """画布拖动事件"""
        self.on_progress_drag_from_canvas(event)

    def on_canvas_release(self, event):
        """画布释放事件"""
        self.on_progress_release_from_canvas(event)

    def on_progress_click_from_canvas(self, event):
        """从画布点击控制进度"""
        if not self.cap:
            return
        self.is_dragging = True
        self.is_playing = False
        self.play_button.config(text="▶️")

        # 计算点击位置对应的帧
        canvas_width = self.range_canvas.winfo_width()
        if canvas_width > 0:
            progress = (event.x / canvas_width) * 100
            self.progress_var.set(progress)
            new_frame = int((progress / 100) * self.total_frames)
            self.current_frame = max(0, min(self.total_frames - 1, new_frame))
            self.update_display()

    def on_progress_drag_from_canvas(self, event):
        """从画布拖动控制进度"""
        if not self.is_dragging or not self.cap:
            return

        canvas_width = self.range_canvas.winfo_width()
        if canvas_width > 0:
            progress = max(0, min(100, (event.x / canvas_width) * 100))
            self.progress_var.set(progress)
            new_frame = int((progress / 100) * self.total_frames)
            self.current_frame = max(0, min(self.total_frames - 1, new_frame))
            self.update_display()

    def on_progress_release_from_canvas(self, event):
        """从画布释放控制进度"""
        self.is_dragging = False

    # 哈欠标记方法
    def toggle_marking_mode(self):
        """切换标记模式"""
        self.is_marking_mode = self.mark_mode_var.get()
        if self.is_marking_mode:
            self.progress_info_var.set("标记模式：拖动进度条到哈欠开始位置，点击'设为开始'")
            self.status_var.set("标记模式已启用 - 使用进度条精确定位哈欠帧")
        else:
            self.progress_info_var.set("拖动进度条精确定位哈欠开始和结束帧")
            self.temp_yawn_start = -1
            self.temp_yawn_end = -1
            self.update_mark_status()

    def set_yawn_start(self):
        """设置哈欠开始帧"""
        if not self.cap:
            messagebox.showwarning("无视频", "请先加载视频文件")
            return

        self.temp_yawn_start = self.current_frame
        self.temp_yawn_end = -1  # 重置结束帧
        self.update_mark_status()
        self.update_display()
        self.progress_info_var.set(f"已设置开始帧: {self.temp_yawn_start}，现在拖动到结束位置")
        self.status_var.set(f"哈欠开始帧: {self.temp_yawn_start}")

    def set_yawn_end(self):
        """设置哈欠结束帧"""
        if not self.cap:
            messagebox.showwarning("无视频", "请先加载视频文件")
            return

        if self.temp_yawn_start == -1:
            messagebox.showwarning("未设置开始", "请先设置哈欠开始帧")
            return

        if self.current_frame <= self.temp_yawn_start:
            messagebox.showwarning("无效范围", "结束帧必须大于开始帧")
            return

        self.temp_yawn_end = self.current_frame
        self.update_mark_status()
        self.update_display()
        self.progress_info_var.set(f"已设置范围: {self.temp_yawn_start}-{self.temp_yawn_end}，点击'添加范围'保存")
        self.status_var.set(f"哈欠范围: {self.temp_yawn_start}-{self.temp_yawn_end}")

    def add_yawn_range(self):
        """添加哈欠范围"""
        if self.temp_yawn_start == -1 or self.temp_yawn_end == -1:
            messagebox.showwarning("未完成标记", "请先设置哈欠的开始和结束帧")
            return

        # 检查重叠
        for start, end in self.yawn_ranges:
            if not (self.temp_yawn_end < start or self.temp_yawn_start > end):
                if messagebox.askyesno("范围重叠", "新范围与现有范围重叠，是否继续添加？"):
                    break
                else:
                    return

        # 添加范围
        self.yawn_ranges.append((self.temp_yawn_start, self.temp_yawn_end))
        self.yawn_ranges.sort()  # 按开始帧排序

        # 自动设置为打哈欠标签
        self.label_type = 2
        self.label_var.set(2)

        # 重置临时标记
        self.temp_yawn_start = -1
        self.temp_yawn_end = -1

        # 更新UI
        self.update_yawn_listbox()  # 这会自动调用 update_range_markers()
        self.update_mark_status()
        self.update_display()

        duration = (self.yawn_ranges[-1][1] - self.yawn_ranges[-1][0]) / self.fps
        self.status_var.set(f"已添加哈欠范围 {len(self.yawn_ranges)}: {self.yawn_ranges[-1][0]}-{self.yawn_ranges[-1][1]} ({duration:.1f}秒)")
        self.progress_info_var.set("哈欠范围已添加，可继续标记或保存标注")

    def clear_all_yawns(self):
        """清除所有哈欠范围"""
        if not self.yawn_ranges and self.temp_yawn_start == -1:
            messagebox.showinfo("无标记", "没有哈欠标记需要清除")
            return

        if messagebox.askyesno("确认清除", "确定要清除所有哈欠标记吗？"):
            self.yawn_ranges = []
            self.temp_yawn_start = -1
            self.temp_yawn_end = -1
            self.update_yawn_listbox()  # 这会自动调用 update_range_markers()
            self.update_mark_status()
            self.update_display()
            self.status_var.set("已清除所有哈欠标记")
            self.progress_info_var.set("已清除标记，可重新开始标注")

    def delete_yawn_range(self, event):
        """删除选中的哈欠范围（右键）"""
        selection = self.yawn_listbox.curselection()
        if selection:
            index = selection[0]
            if messagebox.askyesno("确认删除", f"确定要删除哈欠范围 {index + 1} 吗？"):
                del self.yawn_ranges[index]
                self.update_yawn_listbox()
                self.update_display()
                self.status_var.set(f"已删除哈欠范围 {index + 1}")

    def jump_to_yawn(self, event):
        """跳转到选中的哈欠范围"""
        selection = self.yawn_listbox.curselection()
        if selection and self.yawn_ranges:
            index = selection[0]
            start_frame, _ = self.yawn_ranges[index]
            self.current_frame = start_frame
            self.update_display()
            self.status_var.set(f"已跳转到哈欠范围 {index + 1} 的开始帧")

    def update_mark_status(self):
        """更新标记状态显示"""
        if self.temp_yawn_start != -1 and self.temp_yawn_end != -1:
            duration = (self.temp_yawn_end - self.temp_yawn_start) / self.fps
            self.mark_status_var.set(f"范围: {self.temp_yawn_start}-{self.temp_yawn_end} ({duration:.1f}秒)")
        elif self.temp_yawn_start != -1:
            self.mark_status_var.set(f"开始: {self.temp_yawn_start} (等待结束帧)")
        else:
            self.mark_status_var.set("未标记")

    def update_yawn_listbox(self):
        """更新哈欠范围列表"""
        self.yawn_listbox.delete(0, tk.END)
        for i, (start, end) in enumerate(self.yawn_ranges):
            duration = (end - start) / self.fps
            self.yawn_listbox.insert(tk.END, f"哈欠 {i+1}: {start}-{end} ({duration:.1f}秒)")

        # 更新进度条上的范围标记
        self.update_range_markers()

    def update_range_markers(self):
        """更新进度条上的哈欠范围标记"""
        if not hasattr(self, 'range_canvas') or not self.total_frames:
            return

        # 清除现有标记
        self.range_canvas.delete("all")

        # 获取画布尺寸
        canvas_width = self.range_canvas.winfo_width()
        canvas_height = self.range_canvas.winfo_height()

        if canvas_width <= 1:
            # 如果画布还没有正确初始化，延迟更新
            self.root.after(100, self.update_range_markers)
            return

        # 绘制背景
        self.range_canvas.create_rectangle(0, 0, canvas_width, canvas_height,
                                         fill='#f0f0f0', outline='#d0d0d0', width=1)

        # 绘制哈欠范围
        for i, (start, end) in enumerate(self.yawn_ranges):
            # 计算范围在画布上的位置
            start_x = (start / self.total_frames) * canvas_width
            end_x = (end / self.total_frames) * canvas_width

            # 绘制范围矩形
            self.range_canvas.create_rectangle(start_x, 1, end_x, canvas_height-1,
                                             fill='#ff6b6b', outline='#ff4757', width=1)

            # 添加范围标签（如果空间足够）
            range_width = end_x - start_x
            if range_width > 30:  # 只有当范围足够宽时才显示标签
                center_x = (start_x + end_x) / 2
                self.range_canvas.create_text(center_x, canvas_height/2,
                                            text=f"Y{i+1}", fill='white',
                                            font=('Arial', 7, 'bold'))

        # 绘制临时范围（如果存在）
        if self.temp_yawn_start != -1 and self.temp_yawn_end != -1:
            temp_start_x = (self.temp_yawn_start / self.total_frames) * canvas_width
            temp_end_x = (self.temp_yawn_end / self.total_frames) * canvas_width

            self.range_canvas.create_rectangle(temp_start_x, 1, temp_end_x, canvas_height-1,
                                             fill='#ff9ff3', outline='#f368e0', width=2)

            # 临时范围标签
            temp_range_width = temp_end_x - temp_start_x
            if temp_range_width > 25:
                temp_center_x = (temp_start_x + temp_end_x) / 2
                self.range_canvas.create_text(temp_center_x, canvas_height/2,
                                            text="TEMP", fill='white',
                                            font=('Arial', 6, 'bold'))

        # 绘制当前帧位置指示器
        current_x = (self.current_frame / self.total_frames) * canvas_width
        self.range_canvas.create_line(current_x, 0, current_x, canvas_height,
                                    fill='#2ed573', width=2)

        # 添加刻度线（每10%一个）
        for i in range(11):
            tick_x = (i / 10) * canvas_width
            tick_height = 3 if i % 5 == 0 else 2  # 每50%的刻度线更长
            self.range_canvas.create_line(tick_x, 0, tick_x, tick_height,
                                        fill='#666666', width=1)
            self.range_canvas.create_line(tick_x, canvas_height-tick_height, tick_x, canvas_height,
                                        fill='#666666', width=1)

    def update_label_type(self):
        """更新标签类型"""
        self.label_type = self.label_var.get()
        self.update_display()

    # 视频显示和播放控制
    def update_display(self):
        """更新视频帧显示"""
        if not self.cap:
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()

        if ret:
            # 添加覆盖信息
            self.add_frame_overlay(frame)

            # 调整大小适应画布
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()

            if canvas_width > 1 and canvas_height > 1:
                h, w = frame.shape[:2]
                aspect_ratio = w / h

                if canvas_width / canvas_height > aspect_ratio:
                    new_height = canvas_height
                    new_width = int(new_height * aspect_ratio)
                else:
                    new_width = canvas_width
                    new_height = int(new_width / aspect_ratio)

                frame = cv2.resize(frame, (new_width, new_height))

                # 转换为PhotoImage
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(image)

                # 更新画布
                self.video_canvas.delete("all")
                x = (canvas_width - new_width) // 2
                y = (canvas_height - new_height) // 2
                self.video_canvas.create_image(x, y, anchor=tk.NW, image=photo)
                self.video_canvas.image = photo  # 保持引用

        self.update_info_display()

    def add_frame_overlay(self, frame):
        """添加帧覆盖信息"""
        h, w = frame.shape[:2]

        # 当前帧信息
        cv2.putText(frame, f"Frame: {self.current_frame}/{self.total_frames}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 标签类型
        label_names = ["Normal", "Talking", "Yawning"]
        label_colors = [(0, 255, 0), (0, 255, 255), (0, 0, 255)]
        cv2.putText(frame, f"Label: {label_names[self.label_type]}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_colors[self.label_type], 2)

        # 临时标记状态
        if self.temp_yawn_start != -1:
            if self.temp_yawn_end != -1:
                cv2.putText(frame, f"Temp Range: {self.temp_yawn_start}-{self.temp_yawn_end}",
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            else:
                cv2.putText(frame, f"Temp Start: {self.temp_yawn_start}",
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

        # 高亮显示哈欠范围
        in_yawn_range = False
        for start, end in self.yawn_ranges:
            if start <= self.current_frame <= end:
                cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 5)
                cv2.putText(frame, "YAWNING RANGE", (10, h-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                in_yawn_range = True
                break

        # 高亮临时范围
        if (self.temp_yawn_start != -1 and self.temp_yawn_end != -1 and
            self.temp_yawn_start <= self.current_frame <= self.temp_yawn_end):
            cv2.rectangle(frame, (5, 5), (w-6, h-6), (255, 0, 255), 3)
            cv2.putText(frame, "TEMP YAWN RANGE", (10, h-50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    def update_info_display(self):
        """更新信息显示"""
        # 帧信息
        self.frame_info_var.set(f"帧: {self.current_frame}/{self.total_frames}")

        # 时间信息
        current_time = self.current_frame / self.fps
        total_time = self.total_frames / self.fps
        self.time_info_var.set(f"时间: {self.format_time(current_time)}/{self.format_time(total_time)}")

        # 更新进度条（避免循环更新）
        if not self.is_dragging and self.total_frames > 0:
            progress = (self.current_frame / self.total_frames) * 100
            self.progress_var.set(progress)

        # 更新进度标签
        self.update_progress_labels()

        # 更新范围标记（只更新当前帧位置）
        self.update_current_frame_marker()

    def update_current_frame_marker(self):
        """只更新当前帧位置标记，避免重绘整个画布"""
        if not hasattr(self, 'range_canvas') or not self.total_frames:
            return

        canvas_width = self.range_canvas.winfo_width()
        canvas_height = self.range_canvas.winfo_height()

        if canvas_width <= 1:
            return

        # 删除之前的当前帧标记
        self.range_canvas.delete("current_frame")

        # 绘制新的当前帧位置
        current_x = (self.current_frame / self.total_frames) * canvas_width
        self.range_canvas.create_line(current_x, 0, current_x, canvas_height,
                                    fill='#2ed573', width=2, tags="current_frame")

        # 添加当前帧数字标签
        if canvas_height > 15:  # 只有当画布足够高时才显示
            self.range_canvas.create_text(current_x, canvas_height/2,
                                        text=str(self.current_frame), fill='#2ed573',
                                        font=('Arial', 6, 'bold'), tags="current_frame")

    def update_progress_labels(self):
        """更新进度条标签"""
        if self.total_frames > 0:
            self.current_frame_label.config(text=f"当前帧: {self.current_frame}")
            self.total_frame_label.config(text=str(self.total_frames))

    def format_time(self, seconds):
        """格式化时间显示"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    # 播放控制
    def toggle_playback(self):
        """切换播放状态"""
        if not self.cap:
            return

        self.is_playing = not self.is_playing
        self.play_button.config(text="⏸️" if self.is_playing else "▶️")

        if self.is_playing:
            threading.Thread(target=self.play_video, daemon=True).start()

    def play_video(self):
        """播放视频循环"""
        while self.is_playing and self.current_frame < self.total_frames - 1:
            start_time = time.time()

            self.current_frame += 1
            self.root.after(0, self.update_display)

            # 计算延迟
            frame_delay = (1.0 / self.fps) / self.play_speed
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_delay - elapsed)
            time.sleep(sleep_time)

        # 播放结束
        self.is_playing = False
        self.root.after(0, lambda: self.play_button.config(text="▶️"))

    def seek_frame(self, delta):
        """跳转帧"""
        if not self.cap:
            return

        new_frame = max(0, min(self.total_frames - 1, self.current_frame + delta))
        self.current_frame = new_frame
        self.update_display()

    def update_play_speed(self, event=None):
        """更新播放速度"""
        self.play_speed = self.speed_var.get()


def main():
    """主函数"""
    root = tk.Tk()
    app = ImprovedLabelingTool(root)

    # 键盘快捷键
    def on_key_press(event):
        if event.keysym == 'space':
            app.toggle_playback()
        elif event.keysym == 'Left':
            app.seek_frame(-1)
        elif event.keysym == 'Right':
            app.seek_frame(1)
        elif event.keysym == 'Up':
            # 如果有文件夹，切换到上一个视频；否则跳跃10帧
            if app.video_files:
                app.previous_video()
            else:
                app.seek_frame(10)
        elif event.keysym == 'Down':
            # 如果有文件夹，切换到下一个视频；否则跳跃-10帧
            if app.video_files:
                app.next_video()
            else:
                app.seek_frame(-10)
        elif event.keysym in ['0', '1', '2']:
            app.label_var.set(int(event.keysym))
            app.update_label_type()
        elif event.keysym == 's' and event.state & 0x4:  # Ctrl+S
            app.save_labels()
        elif event.keysym == 'o' and event.state & 0x4:  # Ctrl+O
            app.open_video()
        elif event.keysym == 'f' and event.state & 0x4:  # Ctrl+F
            app.open_folder()

    root.bind('<Key>', on_key_press)
    root.focus_set()

    print("=" * 60)
    print("🎬 改进的视频标注工具 - 文件夹批量标注版")
    print("=" * 60)
    print("🎯 特色功能：")
    print("   📊 精确进度条拖动定位")
    print("   🎯 可视化哈欠范围标记")
    print("   📂 文件夹批量标注")
    print("   💾 智能标注保存")
    print("⌨️  快捷键：")
    print("   空格: 播放/暂停")
    print("   ←/→: 上一帧/下一帧")
    print("   ↑/↓: 上一个/下一个视频 (文件夹模式) 或 跳跃帧")
    print("   0/1/2: 设置标签类型")
    print("   Ctrl+S: 保存标注")
    print("   Ctrl+O: 打开视频")
    print("   Ctrl+F: 打开文件夹")
    print("=" * 60)

    root.mainloop()


if __name__ == "__main__":
    main()
