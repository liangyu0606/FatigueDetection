
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
现代化主界面UI - 重新设计的美观界面
"""

from PySide6.QtCore import QCoreApplication, QMetaObject, Qt
from PySide6.QtWidgets import (QCheckBox, QGridLayout, QGroupBox,
    QLabel, QPlainTextEdit, QPushButton,
    QSpinBox, QVBoxLayout, QHBoxLayout, QFrame)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(1400, 900)  # 进一步增加默认窗口大小以容纳更大的视频区域
        Form.setMinimumSize(1200, 800)  # 增加最小窗口大小

        # 应用现代化样式
        Form.setStyleSheet(self.get_modern_style())

        # 主布局
        main_layout = QVBoxLayout(Form)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # 内容区域
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # 左侧：视频和统计区域
        left_layout = QVBoxLayout()
        left_layout.setSpacing(15)
        self.create_video_section(left_layout)
        self.create_stats_section(left_layout)

        # 创建左侧容器并设置拉伸策略
        left_widget = QFrame()
        left_widget.setLayout(left_layout)
        from PySide6.QtWidgets import QSizePolicy
        left_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content_layout.addWidget(left_widget, 3)  # 增加拉伸因子

        # 右侧：控制面板
        self.create_control_panel(content_layout)

        main_layout.addLayout(content_layout)

        # 底部：日志和提示区域
        self.create_bottom_section(main_layout)

        self.retranslateUi(Form)
        QMetaObject.connectSlotsByName(Form)

    def get_modern_style(self):
        """获取现代化样式"""
        return """
        QWidget {
            font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #f8f9fa, stop:1 #e9ecef);
        }

        QGroupBox {
            font-weight: bold;
            font-size: 14pt;
            color: #2e7d32;
            border: 2px solid #e0e0e0;
            border-radius: 15px;
            margin-top: 15px;
            padding-top: 15px;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #ffffff, stop:1 #f8f9fa);
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 20px;
            padding: 0 15px 0 15px;
            background: white;
            border-radius: 5px;
        }

        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #4CAF50, stop:1 #45a049);
            border: none;
            border-radius: 10px;
            color: white;
            font-weight: bold;
            padding: 12px 25px;
            font-size: 12pt;
            min-height: 25px;
        }

        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #5CBF60, stop:1 #4CAF50);
        }

        QPushButton#pushButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #FF6B6B, stop:1 #EE5A52);
            font-size: 16pt;
            padding: 15px 30px;
            min-height: 40px;
        }

        QPushButton#pushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #FF8E8E, stop:1 #FF6B6B);
        }

        QSpinBox {
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 8px 12px;
            background: white;
            font-size: 11pt;
            min-height: 20px;
        }

        QSpinBox:focus {
            border-color: #4CAF50;
            background: #f8fff8;
        }

        QCheckBox {
            font-size: 11pt;
            color: #333333;
            spacing: 8px;
        }

        QPlainTextEdit {
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            background: white;
            padding: 10px;
            font-size: 10pt;
        }

        QLabel {
            color: #333333;
            font-size: 11pt;
        }

        QFrame[class="card"] {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 15px;
        }
        """



    def create_video_section(self, layout):
        """创建视频显示区域"""
        video_frame = QFrame()
        video_frame.setProperty("class", "card")
        # 增加视频区域的最小尺寸，确保有足够空间显示完整人脸
        video_frame.setMinimumSize(700, 550)  # 增加尺寸
        video_layout = QVBoxLayout(video_frame)
        video_layout.setSpacing(10)
        video_layout.setContentsMargins(15, 15, 15, 15)

        video_title = QLabel("📹 实时视频监控")
        video_title.setFixedHeight(40)  # 固定标题高度
        video_title.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #2e7d32;
                padding: 10px;
                background: #f8f9fa;
                border-radius: 8px;
                margin-bottom: 5px;
            }
        """)
        video_layout.addWidget(video_title)

        # 实际的视频显示标签 - 这是main.py期望的label_img
        self.label_img = QLabel("等待视频流...")
        self.label_img.setObjectName("label_img")
        self.label_img.setAlignment(Qt.AlignCenter)
        # 启用缩放内容以确保图像能正确显示并保持宽高比
        self.label_img.setScaledContents(True)  # 改为True以确保图像能够缩放显示
        # 增加最小尺寸，确保有足够空间显示完整人脸
        self.label_img.setMinimumSize(640, 480)  # 增加最小尺寸
        # 设置大小策略，允许扩展
        from PySide6.QtWidgets import QSizePolicy
        self.label_img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_img.setStyleSheet("""
            QLabel {
                background: #f0f0f0;
                border: 2px solid #ddd;
                border-radius: 10px;
                font-size: 14px;
                color: #666;
            }
        """)
        video_layout.addWidget(self.label_img, 1)  # 添加拉伸因子
        layout.addWidget(video_frame)

    def create_stats_section(self, layout):
        """创建统计显示区域"""
        self.groupBox_5 = QGroupBox("📊 实时统计")
        self.groupBox_5.setObjectName(u"groupBox_5")
        # 设置固定高度，防止在窗口放大时过度拉伸
        self.groupBox_5.setMaximumHeight(220)
        self.groupBox_5.setMinimumHeight(200)

        stats_layout = QGridLayout(self.groupBox_5)
        stats_layout.setSpacing(8)
        stats_layout.setContentsMargins(15, 25, 15, 15)
        # 设置行间距
        stats_layout.setVerticalSpacing(5)
        stats_layout.setHorizontalSpacing(12)

        # 创建统计标签
        stats_data = [
            ("人脸识别", "label_11", "未识别", 0, 0),
            ("疲劳状态", "label_12", "正常", 0, 1),
            ("帧率", "label_13", "0", 0, 2),
            ("头部姿态", "label_21", "正", 1, 0),
            ("嘴部状态", "label_22", "闭嘴", 1, 1),
            ("眼部状态", "label_23", "睁眼", 1, 2),
            ("眨眼次数", "label_31", "0", 2, 0),
            ("哈欠次数", "label_32", "0", 2, 1),
            ("点头次数", "label_33", "0", 2, 2),
        ]

        for title, obj_name, default_val, row, col in stats_data:
            # 标题标签
            title_label = QLabel(title)
            title_label.setStyleSheet("""
                QLabel {
                    font-weight: bold;
                    color: #666;
                    font-size: 9pt;
                    margin-bottom: 3px;
                    padding: 2px;
                }
            """)
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setMinimumHeight(20)
            title_label.setMaximumHeight(25)
            title_label.setWordWrap(True)  # 允许文字换行
            stats_layout.addWidget(title_label, row * 2, col)

            # 数值标签
            value_label = QLabel(default_val)
            value_label.setObjectName(obj_name)
            value_label.setAlignment(Qt.AlignCenter)
            value_label.setMinimumHeight(30)
            value_label.setMaximumHeight(40)
            value_label.setMinimumWidth(80)
            value_label.setStyleSheet("""
                QLabel {
                    background: #f8f9fa;
                    border: 1px solid #dee2e6;
                    border-radius: 5px;
                    padding: 6px 8px;
                    font-weight: bold;
                    color: #495057;
                    font-size: 12pt;
                }
            """)
            value_label.setWordWrap(True)  # 允许文字换行
            stats_layout.addWidget(value_label, row * 2 + 1, col)
            setattr(self, obj_name, value_label)

        # 设置列的拉伸策略，使其均匀分布
        for col in range(3):
            stats_layout.setColumnStretch(col, 1)
            stats_layout.setColumnMinimumWidth(col, 100)  # 设置最小列宽

        # 设置行的拉伸策略和最小高度
        for row in range(6):  # 3行 * 2 (标题+数值) = 6行
            stats_layout.setRowStretch(row, 0)
            if row % 2 == 0:  # 标题行
                stats_layout.setRowMinimumHeight(row, 25)
            else:  # 数值行
                stats_layout.setRowMinimumHeight(row, 40)

        layout.addWidget(self.groupBox_5)

    def create_control_panel(self, content_layout):
        """创建控制面板"""
        self.groupBox_2 = QGroupBox("🎛️ 控制面板")
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setMinimumWidth(320)
        self.groupBox_2.setMaximumWidth(400)  # 设置最大宽度，防止过度拉伸
        # 设置大小策略，固定宽度
        from PySide6.QtWidgets import QSizePolicy
        self.groupBox_2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        self.gridLayout_2 = QGridLayout(self.groupBox_2)
        self.gridLayout_2.setSpacing(15)
        self.gridLayout_2.setContentsMargins(20, 25, 20, 20)

        # 主控制按钮（切换式：开始检测/停止检测）
        self.pushButton = QPushButton("🚀 开始检测")
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setMinimumHeight(50)
        self.pushButton.setStyleSheet("""
            QPushButton#pushButton {
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
            QPushButton#pushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5CBF60, stop:1 #4CAF50);
            }
        """)
        self.gridLayout_2.addWidget(self.pushButton, 0, 0, 1, 2)

        # 参数设置
        params = [
            ("眨眼阈值", "spinBox_3", 30, 1),
            ("敏感度", "spinBox_4", 5, 2),
            ("疲劳阈值", "spinBox_5", 25, 3),
            ("检测间隔", "spinBox_1", 100, 4),
        ]

        for label_text, obj_name, default_val, row in params:
            label = QLabel(f"⚙️ {label_text}:")
            label.setStyleSheet("font-weight: bold; color: #333;")
            self.gridLayout_2.addWidget(label, row, 0)

            spinbox = QSpinBox(self.groupBox_2)
            spinbox.setObjectName(obj_name)
            spinbox.setRange(1, 1000)
            spinbox.setValue(default_val)
            self.gridLayout_2.addWidget(spinbox, row, 1)
            setattr(self, obj_name, spinbox)

        # 功能选项
        options = [
            ("🔊 声音警报", "checkBox", True, 5),
            ("📊 实时统计", "checkBox_2", True, 6),
            ("💾 自动保存", "checkBox_3", False, 7),
            ("🌙 夜间模式", "checkBox_4", False, 8),
        ]

        for label_text, obj_name, checked, row in options:
            checkbox = QCheckBox(label_text)
            checkbox.setObjectName(obj_name)
            checkbox.setChecked(checked)
            self.gridLayout_2.addWidget(checkbox, row, 0, 1, 2)
            setattr(self, obj_name, checkbox)

        content_layout.addWidget(self.groupBox_2)

    def create_bottom_section(self, main_layout):
        """创建底部日志和提示区域"""
        self.groupBox_4 = QGroupBox("📝 系统日志")
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.groupBox_4.setMaximumHeight(150)
        self.groupBox_4.setMinimumHeight(120)
        # 设置大小策略，固定高度
        from PySide6.QtWidgets import QSizePolicy
        self.groupBox_4.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        bottom_layout = QVBoxLayout(self.groupBox_4)
        bottom_layout.setContentsMargins(15, 20, 15, 15)

        self.plainTextEdit_tip = QPlainTextEdit()
        self.plainTextEdit_tip.setObjectName(u"plainTextEdit_tip")
        self.plainTextEdit_tip.setMaximumHeight(100)
        self.plainTextEdit_tip.setMinimumHeight(80)
        self.plainTextEdit_tip.setPlainText("系统就绪，等待开始检测...")

        bottom_layout.addWidget(self.plainTextEdit_tip)
        main_layout.addWidget(self.groupBox_4)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"疲劳检测系统", None))
        # 只设置实际存在的组件
        if hasattr(self, 'groupBox_5'):
            self.groupBox_5.setTitle(QCoreApplication.translate("Form", u"📊 实时统计", None))
        if hasattr(self, 'groupBox_2'):
            self.groupBox_2.setTitle(QCoreApplication.translate("Form", u"🎛️ 控制面板", None))
        if hasattr(self, 'groupBox_4'):
            self.groupBox_4.setTitle(QCoreApplication.translate("Form", u"📝 系统日志", None))
        if hasattr(self, 'pushButton'):
            self.pushButton.setText(QCoreApplication.translate("Form", u"🚀 开始检测", None))
        if hasattr(self, 'label_img'):
            self.label_img.setText("")
    # retranslateUi

