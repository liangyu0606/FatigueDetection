#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用户管理界面模块
提供用户登录、用户管理、权限控制等界面功能
"""

import sys
from typing import Optional
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QComboBox, QMessageBox, QDialog, QFormLayout, QCheckBox,
    QTabWidget, QTextEdit, QDateTimeEdit, QSpinBox, QGroupBox,
    QGridLayout, QHeaderView, QFrame, QSplitter, QGraphicsDropShadowEffect
)
from PySide6.QtCore import Qt, QDateTime, QTimer, Signal, QPropertyAnimation, QEasingCurve, QRect
from PySide6.QtGui import QFont, QIcon, QPalette, QColor, QPixmap, QPainter, QLinearGradient

from user_management import UserManager, UserRole, Permission, User
from system_logger import SystemLogger, LogLevel, LogCategory


class ModernStyleSheet:
    """现代化样式表"""

    @staticmethod
    def get_main_style():
        return """
        /* 主窗口样式 */
        QMainWindow {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #f8f9fa, stop:1 #e9ecef);
        }

        /* 通用组件样式 */
        QWidget {
            font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif;
            font-size: 10pt;
        }

        /* 按钮样式 */
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #4CAF50, stop:1 #45a049);
            border: none;
            border-radius: 8px;
            color: white;
            font-weight: bold;
            padding: 10px 20px;
            min-height: 20px;
        }

        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #5CBF60, stop:1 #4CAF50);
            transform: translateY(-1px);
        }

        QPushButton:pressed {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #3d8b40, stop:1 #2e7d32);
        }

        QPushButton:disabled {
            background: #cccccc;
            color: #666666;
        }

        /* 危险按钮样式 */
        QPushButton[class="danger"] {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #f44336, stop:1 #d32f2f);
        }

        QPushButton[class="danger"]:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #f66356, stop:1 #f44336);
        }

        /* 次要按钮样式 */
        QPushButton[class="secondary"] {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #6c757d, stop:1 #5a6268);
        }

        QPushButton[class="secondary"]:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #7c858d, stop:1 #6c757d);
        }

        /* 输入框样式 */
        QLineEdit {
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 10px 15px;
            background: white;
            font-size: 11pt;
        }

        QLineEdit:focus {
            border-color: #4CAF50;
            background: #f8fff8;
        }

        /* 组合框样式 */
        QComboBox {
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 8px 15px;
            background: white;
            min-width: 120px;
        }

        QComboBox:focus {
            border-color: #4CAF50;
        }

        QComboBox::drop-down {
            border: none;
            width: 30px;
        }

        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid #666;
            margin-right: 10px;
        }

        /* 表格样式 */
        QTableWidget {
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            background: white;
            gridline-color: #f0f0f0;
            selection-background-color: #e8f5e8;
        }

        QTableWidget::item {
            padding: 12px 8px;
            border-bottom: 1px solid #f0f0f0;
        }

        QTableWidget::item:selected {
            background: #e8f5e8;
            color: #2e7d32;
        }

        QHeaderView::section {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #f8f9fa, stop:1 #e9ecef);
            border: none;
            border-bottom: 2px solid #4CAF50;
            padding: 12px 8px;
            font-weight: bold;
            color: #2e7d32;
        }

        /* 分组框样式 */
        QGroupBox {
            font-weight: bold;
            font-size: 12pt;
            color: #2e7d32;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            margin-top: 10px;
            padding-top: 10px;
            background: white;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 15px;
            padding: 0 10px 0 10px;
            background: white;
        }

        /* 标签样式 */
        QLabel {
            color: #333333;
        }

        QLabel[class="title"] {
            font-size: 18pt;
            font-weight: bold;
            color: #2e7d32;
        }

        QLabel[class="subtitle"] {
            font-size: 12pt;
            color: #666666;
        }

        /* 对话框样式 */
        QDialog {
            background: white;
            border-radius: 15px;
        }

        /* 滚动条样式 */
        QScrollBar:vertical {
            border: none;
            background: #f0f0f0;
            width: 12px;
            border-radius: 6px;
        }

        QScrollBar::handle:vertical {
            background: #c0c0c0;
            border-radius: 6px;
            min-height: 20px;
        }

        QScrollBar::handle:vertical:hover {
            background: #a0a0a0;
        }
        """


class LoginDialog(QDialog):
    """现代化登录对话框"""

    login_successful = Signal(str)  # 登录成功信号，传递用户名

    def __init__(self, user_manager: UserManager, logger: SystemLogger):
        super().__init__()
        self.user_manager = user_manager
        self.logger = logger
        self.setup_ui()
        self.apply_modern_style()

    def setup_ui(self):
        """设置界面"""
        self.setWindowTitle("疲劳驾驶检测系统 - 用户登录")
        self.setFixedSize(450, 550)  # 增加高度以确保标题完全显示
        self.setModal(True)

        # 应用样式
        self.setStyleSheet(ModernStyleSheet.get_main_style())

        # 主布局
        main_layout = QVBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 创建主容器
        container = QFrame()
        container.setObjectName("loginContainer")
        container.setStyleSheet("""
            QFrame#loginContainer {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 15px;
            }
        """)

        layout = QVBoxLayout(container)
        layout.setSpacing(15)  # 减少间距，让文本框向上移动
        layout.setContentsMargins(30, 25, 30, 30)  # 减少顶部边距

        # 标题区域
        title_container = QFrame()
        title_layout = QVBoxLayout(title_container)
        title_layout.setSpacing(12)  # 减少标题区域内部间距
        title_layout.setAlignment(Qt.AlignCenter)
        title_layout.setContentsMargins(20, 5, 20, 5)  # 减少内边距

        # 图标
        icon_label = QLabel("🚗")
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet("""
            QLabel {
                font-size: 52px;
                color: white;
                background: rgba(255, 255, 255, 0.15);
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-radius: 40px;
                padding: 15px;
                min-width: 80px;
                max-width: 80px;
                min-height: 80px;
                max-height: 80px;
                margin: 0 auto;
            }
        """)

        # 标题
        title_label = QLabel("疲劳驾驶检测系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                font-family: 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS', sans-serif;
                font-size: 24px;
                font-weight: bold;
                color: red;
                margin: 10px 0;
                padding: 8px 0;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
                background: transparent;
                border: none;
            }
        """)

        # 副标题
        subtitle_label = QLabel("安全驾驶，智能监控")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("""
            QLabel {
                font-family: 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS', sans-serif;
                font-size: 15px;
                color: rgba(255, 255, 255, 0.8);
                margin-bottom: 10px;
                padding-bottom: 10px;
            }
        """)

        title_layout.addWidget(icon_label)
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        layout.addWidget(title_container)

        # 表单区域 - 减少顶部间距，将输入框往上移
        form_container = QFrame()
        form_container.setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                padding: 10px 20px;
            }
        """)

        form_layout = QVBoxLayout(form_container)
        form_layout.setSpacing(8)  # 进一步减少表单内部间距
        form_layout.setContentsMargins(0, 0, 0, 0)  # 移除表单容器边距


        self.username_edit = QLineEdit()
        self.username_edit.setPlaceholderText("请输入用户名")
        self.username_edit.setText("admin")  # 默认填入admin
        self.username_edit.setStyleSheet("""
            QLineEdit {
                background: rgba(255, 255, 255, 0.95);
                border: 2px solid rgba(255, 255, 255, 0.5);
                border-radius: 8px;
                padding: 12px 15px;
                font-size: 12pt;
                color: #333;
                min-height: 20px;
            }
            QLineEdit:focus {
                border-color: white;
                background: white;
                box-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
            }
        """)
        form_layout.addWidget(self.username_edit)


        self.password_edit = QLineEdit()
        self.password_edit.setPlaceholderText("请输入密码")
        self.password_edit.setText("admin123")  # 默认填入密码
        self.password_edit.setEchoMode(QLineEdit.Password)
        self.password_edit.setStyleSheet("""
            QLineEdit {
                background: rgba(255, 255, 255, 0.95);
                border: 2px solid rgba(255, 255, 255, 0.5);
                border-radius: 8px;
                padding: 12px 15px;
                font-size: 12pt;
                color: #333;
                min-height: 20px;
            }
            QLineEdit:focus {
                border-color: white;
                background: white;
                box-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
            }
        """)
        form_layout.addWidget(self.password_edit)

        layout.addWidget(form_container)

        # 按钮区域
        button_container = QHBoxLayout()
        button_container.setSpacing(15)

        self.register_button = QPushButton("注册")
        self.register_button.clicked.connect(self.register)
        self.register_button.setStyleSheet("""
            QPushButton {
                background: rgba(255, 255, 255, 0.2);
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-radius: 8px;
                color: white;
                font-weight: bold;
                padding: 12px 25px;
                font-size: 11pt;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.3);
                border-color: rgba(255, 255, 255, 0.5);
            }
        """)

        self.login_button = QPushButton("登录")
        self.login_button.clicked.connect(self.login)
        self.login_button.setDefault(True)
        self.login_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4CAF50, stop:1 #45a049);
                border: none;
                border-radius: 8px;
                color: white;
                font-weight: bold;
                padding: 12px 25px;
                font-size: 11pt;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5CBF60, stop:1 #4CAF50);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3d8b40, stop:1 #2e7d32);
            }
        """)

        button_container.addWidget(self.register_button)
        button_container.addWidget(self.login_button)
        layout.addLayout(button_container)

        # 提示信息
        info_label = QLabel("默认管理员账户: admin / admin123")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 0.7);
                font-size: 10pt;
                margin-top: 15px;
                padding: 8px;
                background: rgba(0, 0, 0, 0.1);
                border-radius: 5px;
            }
        """)
        layout.addWidget(info_label)

        main_layout.addWidget(container)
        self.setLayout(main_layout)

        # 设置焦点和回车键登录
        self.username_edit.setFocus()
        self.password_edit.returnPressed.connect(self.login)

    def apply_modern_style(self):
        """应用现代化样式效果"""
        # 添加阴影效果
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 10)
        self.setGraphicsEffect(shadow)
        
    def login(self):
        """执行登录"""
        username = self.username_edit.text().strip()
        password = self.password_edit.text()
        
        if not username or not password:
            QMessageBox.warning(self, "警告", "请输入用户名和密码")
            return
        
        success, message = self.user_manager.login(username, password)
        
        if success:
            # 记录登录日志
            self.logger.log_user_action(
                action="login",
                description="用户登录成功",
                user_id=self.user_manager.current_user.user_id,
                username=username,
                session_id=message  # message是session_token
            )
            
            self.login_successful.emit(username)
            self.accept()
        else:
            # 记录登录失败日志
            self.logger.log_security_event(
                action="login_failed",
                description=f"用户登录失败: {message}",
                username=username,
                level=LogLevel.WARNING
            )
            
            QMessageBox.critical(self, "登录失败", message)

    def register(self):
        """执行注册"""
        username = self.username_edit.text().strip()
        password = self.password_edit.text()

        if not username or not password:
            QMessageBox.warning(self, "警告", "请输入用户名和密码")
            return

        # 检查用户名长度
        if len(username) < 3:
            QMessageBox.warning(self, "警告", "用户名至少需要3个字符")
            return

        # 检查密码长度
        if len(password) < 6:
            QMessageBox.warning(self, "警告", "密码至少需要6个字符")
            return

        # 检查用户名是否已存在
        if self.user_manager.get_user_by_username(username):
            QMessageBox.warning(self, "警告", f"用户名 '{username}' 已存在，请选择其他用户名")
            return

        # 注册新用户，角色固定为driver
        success = self.user_manager.register_user(
            username=username,
            password=password,
            role=UserRole.DRIVER,  # 固定为驾驶员角色
            email="",  # 可以为空
            full_name=""  # 可以为空
        )

        if success:
            # 记录注册日志
            self.logger.log_system_event(
                action="user_register",
                description=f"新用户注册成功: {username}",
                level=LogLevel.INFO,
                details={"username": username, "role": "driver"}
            )

            QMessageBox.information(
                self, "注册成功",
                f"用户 '{username}' 注册成功！\n"
                f"您的角色是：驾驶员\n"
                f"请使用新账户登录。"
            )

            # 清空密码框，保留用户名方便登录
            self.password_edit.clear()
            self.password_edit.setFocus()
        else:
            # 记录注册失败日志
            self.logger.log_system_event(
                action="user_register_failed",
                description=f"用户注册失败: {username}",
                level=LogLevel.WARNING,
                details={"username": username, "reason": "registration_failed"}
            )

            QMessageBox.critical(self, "注册失败", "用户注册失败，请稍后重试")


class UserManagementWidget(QWidget):
    """现代化用户管理组件"""

    def __init__(self, user_manager: UserManager, logger: SystemLogger):
        super().__init__()
        self.user_manager = user_manager
        self.logger = logger
        self.setup_ui()
        self.apply_modern_style()
        self.load_users()

    def setup_ui(self):
        """设置现代化界面"""
        # 应用样式
        self.setStyleSheet(ModernStyleSheet.get_main_style())

        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # 标题区域
        title_container = QFrame()
        title_container.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 10px;
                padding: 20px;
            }
        """)

        title_layout = QHBoxLayout(title_container)

        # 标题图标和文字
        title_icon = QLabel("👥")
        title_icon.setStyleSheet("""
            QLabel {
                font-size: 24px;
                color: white;
                margin-right: 10px;
            }
        """)

        title_label = QLabel("用户管理")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: bold;
                color: white;
            }
        """)

        subtitle_label = QLabel("管理系统用户和权限")
        subtitle_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: rgba(255, 255, 255, 0.8);
                margin-left: 20px;
            }
        """)

        title_layout.addWidget(title_icon)
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        title_layout.addStretch()

        main_layout.addWidget(title_container)

        # 工具栏区域
        toolbar_container = QFrame()
        toolbar_container.setStyleSheet("""
            QFrame {
                background: white;
                border-radius: 10px;
                padding: 15px;
                border: 1px solid #e0e0e0;
            }
        """)

        toolbar_layout = QHBoxLayout(toolbar_container)
        toolbar_layout.setSpacing(15)

        # 添加用户按钮
        self.add_user_button = QPushButton("➕ 添加用户")
        self.add_user_button.clicked.connect(self.add_user)
        self.add_user_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4CAF50, stop:1 #45a049);
                border: none;
                border-radius: 8px;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                font-size: 11pt;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5CBF60, stop:1 #4CAF50);
            }
        """)

        # 编辑用户按钮
        self.edit_user_button = QPushButton("✏️ 编辑用户")
        self.edit_user_button.clicked.connect(self.edit_user)
        self.edit_user_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2196F3, stop:1 #1976D2);
                border: none;
                border-radius: 8px;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                font-size: 11pt;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #42A5F5, stop:1 #2196F3);
            }
        """)

        # 删除用户按钮
        self.delete_user_button = QPushButton("🗑️ 删除用户")
        self.delete_user_button.clicked.connect(self.delete_user)
        self.delete_user_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f44336, stop:1 #d32f2f);
                border: none;
                border-radius: 8px;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                font-size: 11pt;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f66356, stop:1 #f44336);
            }
        """)

        # 刷新按钮
        self.refresh_button = QPushButton("🔄 刷新")
        self.refresh_button.clicked.connect(self.load_users)
        self.refresh_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #6c757d, stop:1 #5a6268);
                border: none;
                border-radius: 8px;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                font-size: 11pt;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #7c858d, stop:1 #6c757d);
            }
        """)

        toolbar_layout.addWidget(self.add_user_button)
        toolbar_layout.addWidget(self.edit_user_button)
        toolbar_layout.addWidget(self.delete_user_button)
        toolbar_layout.addWidget(self.refresh_button)
        toolbar_layout.addStretch()

        main_layout.addWidget(toolbar_container)

        # 用户表格容器
        table_container = QFrame()
        table_container.setStyleSheet("""
            QFrame {
                background: white;
                border-radius: 10px;
                border: 1px solid #e0e0e0;
            }
        """)

        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)

        # 用户表格
        self.user_table = QTableWidget()
        self.user_table.setColumnCount(7)
        self.user_table.setHorizontalHeaderLabels([
            "用户名", "角色", "全名", "邮箱", "创建时间", "最后登录", "状态"
        ])

        # 设置表格样式
        self.user_table.setStyleSheet("""
            QTableWidget {
                border: none;
                border-radius: 10px;
                background: white;
                gridline-color: #f0f0f0;
                selection-background-color: #e8f5e8;
                font-size: 11pt;
            }
            QTableWidget::item {
                padding: 15px 10px;
                border-bottom: 1px solid #f0f0f0;
            }
            QTableWidget::item:selected {
                background: #e8f5e8;
                color: #2e7d32;
            }
            QHeaderView::section {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8f9fa, stop:1 #e9ecef);
                border: none;
                border-bottom: 3px solid #4CAF50;
                padding: 15px 10px;
                font-weight: bold;
                color: #2e7d32;
                font-size: 12pt;
            }
        """)

        # 设置表格属性
        header = self.user_table.horizontalHeader()
        
        # 设置列宽度策略
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # 用户名
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # 角色
        header.setSectionResizeMode(2, QHeaderView.Stretch)           # 全名
        header.setSectionResizeMode(3, QHeaderView.Stretch)           # 邮箱
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # 创建时间
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # 最后登录
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)  # 状态

        # 设置最小列宽，确保时间列有足够空间
        self.user_table.setColumnWidth(4, 150)  # 创建时间列最小宽度
        self.user_table.setColumnWidth(5, 150)  # 最后登录列最小宽度

        self.user_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.user_table.setAlternatingRowColors(True)
        self.user_table.verticalHeader().setVisible(False)

        table_layout.addWidget(self.user_table)
        main_layout.addWidget(table_container)

        self.setLayout(main_layout)

    def apply_modern_style(self):
        """应用现代化样式效果"""
        # 添加阴影效果到主容器
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 30))
        shadow.setOffset(0, 5)
        self.setGraphicsEffect(shadow)
        
    def load_users(self):
        """加载用户列表"""
        try:
            users = self.user_manager.get_all_users()
            self.user_table.setRowCount(len(users))

            for row, user in enumerate(users):
                # 列顺序：["用户名", "角色", "全名", "邮箱", "创建时间", "最后登录", "状态"]
                self.user_table.setItem(row, 0, QTableWidgetItem(user.username or ""))
                self.user_table.setItem(row, 1, QTableWidgetItem(user.role.value if user.role else ""))
                self.user_table.setItem(row, 2, QTableWidgetItem(user.full_name or ""))
                # 邮箱显示，如果为空则显示占位符
                email_display = user.email if user.email else "未设置邮箱"
                self.user_table.setItem(row, 3, QTableWidgetItem(email_display))

                # 格式化创建时间
                created_at = ""
                if user.created_at:
                    try:
                        created_at = user.created_at.strftime("%m-%d %H:%M")
                    except:
                        created_at = str(user.created_at)
                self.user_table.setItem(row, 4, QTableWidgetItem(created_at))

                # 格式化最后登录时间
                last_login = "从未登录"
                if user.last_login:
                    try:
                        last_login = user.last_login.strftime("%m-%d %H:%M")
                    except:
                        last_login = str(user.last_login)
                self.user_table.setItem(row, 5, QTableWidgetItem(last_login))

                # 用户状态
                status = "活跃" if user.is_active else "禁用"
                self.user_table.setItem(row, 6, QTableWidgetItem(status))

                # 存储用户ID用于后续操作
                self.user_table.item(row, 0).setData(Qt.UserRole, user.user_id)

        except Exception as e:
            print(f"加载用户列表失败: {e}")
            QMessageBox.critical(self, "错误", f"加载用户列表失败: {e}")
    
    def add_user(self):
        """添加用户"""
        if not self.user_manager.has_permission(Permission.USER_MANAGE):
            QMessageBox.warning(self, "权限不足", "您没有用户管理权限")
            return
            
        dialog = UserEditDialog(self.user_manager, self.logger)
        if dialog.exec() == QDialog.Accepted:
            self.load_users()
    
    def edit_user(self):
        """编辑用户"""
        if not self.user_manager.has_permission(Permission.USER_MANAGE):
            QMessageBox.warning(self, "权限不足", "您没有用户管理权限")
            return
            
        current_row = self.user_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "警告", "请选择要编辑的用户")
            return
        
        user_id = self.user_table.item(current_row, 0).data(Qt.UserRole)
        user = self.user_manager.get_user_by_id(user_id)
        
        if user:
            dialog = UserEditDialog(self.user_manager, self.logger, user)
            if dialog.exec() == QDialog.Accepted:
                self.load_users()
    
    def delete_user(self):
        """删除用户"""
        if not self.user_manager.has_permission(Permission.USER_MANAGE):
            QMessageBox.warning(self, "权限不足", "您没有用户管理权限")
            return
            
        current_row = self.user_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "警告", "请选择要删除的用户")
            return
        
        username = self.user_table.item(current_row, 0).text()
        user_id = self.user_table.item(current_row, 0).data(Qt.UserRole)
        
        # 防止删除当前用户
        if user_id == self.user_manager.current_user.user_id:
            QMessageBox.warning(self, "警告", "不能删除当前登录用户")
            return
        
        reply = QMessageBox.question(
            self, "确认删除", 
            f"确定要删除用户 '{username}' 吗？\n此操作不可撤销。",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            if self.user_manager.delete_user(user_id):
                # 记录删除日志
                self.logger.log_user_action(
                    action="delete_user",
                    description=f"删除用户: {username}",
                    user_id=self.user_manager.current_user.user_id,
                    username=self.user_manager.current_user.username,
                    details={"deleted_user": username, "deleted_user_id": user_id}
                )
                
                QMessageBox.information(self, "成功", "用户删除成功")
                self.load_users()
            else:
                QMessageBox.critical(self, "错误", "用户删除失败")


class UserEditDialog(QDialog):
    """用户编辑对话框"""
    
    def __init__(self, user_manager: UserManager, logger: SystemLogger, user: User = None):
        super().__init__()
        self.user_manager = user_manager
        self.logger = logger
        self.user = user
        self.is_edit_mode = user is not None
        self.setup_ui()
        
        if self.is_edit_mode:
            self.load_user_data()
    
    def setup_ui(self):
        """设置界面"""
        title = "编辑用户" if self.is_edit_mode else "添加用户"
        self.setWindowTitle(title)
        self.setFixedSize(400, 350)
        self.setModal(True)
        
        layout = QFormLayout()
        
        # 用户名
        self.username_edit = QLineEdit()
        if self.is_edit_mode:
            self.username_edit.setReadOnly(True)
        layout.addRow("用户名:", self.username_edit)
        
        # 密码
        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.Password)
        if self.is_edit_mode:
            self.password_edit.setPlaceholderText("留空表示不修改密码")
        layout.addRow("密码:", self.password_edit)
        
        # 角色
        self.role_combo = QComboBox()
        for role in UserRole:
            self.role_combo.addItem(role.value, role)
        layout.addRow("角色:", self.role_combo)
        
        # 全名
        self.full_name_edit = QLineEdit()
        layout.addRow("全名:", self.full_name_edit)
        
        # 邮箱
        self.email_edit = QLineEdit()
        layout.addRow("邮箱:", self.email_edit)
        
        # 状态
        self.active_checkbox = QCheckBox("账户激活")
        self.active_checkbox.setChecked(True)
        layout.addRow("状态:", self.active_checkbox)
        
        # 按钮
        button_layout = QHBoxLayout()
        
        self.save_button = QPushButton("保存")
        self.save_button.clicked.connect(self.save_user)
        
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addRow(button_layout)
        self.setLayout(layout)
    
    def load_user_data(self):
        """加载用户数据"""
        if self.user:
            self.username_edit.setText(self.user.username)
            self.full_name_edit.setText(self.user.full_name)
            self.email_edit.setText(self.user.email)
            self.active_checkbox.setChecked(self.user.is_active)
            
            # 设置角色
            for i in range(self.role_combo.count()):
                if self.role_combo.itemData(i) == self.user.role:
                    self.role_combo.setCurrentIndex(i)
                    break
    
    def save_user(self):
        """保存用户"""
        username = self.username_edit.text().strip()
        password = self.password_edit.text()
        role = self.role_combo.currentData()
        full_name = self.full_name_edit.text().strip()
        email = self.email_edit.text().strip()
        is_active = self.active_checkbox.isChecked()
        
        if not username:
            QMessageBox.warning(self, "警告", "请输入用户名")
            return
        
        if not self.is_edit_mode and not password:
            QMessageBox.warning(self, "警告", "请输入密码")
            return
        
        try:
            if self.is_edit_mode:
                # 更新用户
                update_data = {
                    'role': role,
                    'full_name': full_name,
                    'email': email,
                    'is_active': is_active
                }
                
                if password:  # 如果输入了新密码
                    update_data['password'] = password
                
                if self.user_manager.update_user(self.user.user_id, **update_data):
                    # 记录更新日志
                    self.logger.log_user_action(
                        action="update_user",
                        description=f"更新用户信息: {username}",
                        user_id=self.user_manager.current_user.user_id,
                        username=self.user_manager.current_user.username,
                        details={"updated_user": username, "changes": update_data}
                    )
                    
                    QMessageBox.information(self, "成功", "用户信息更新成功")
                    self.accept()
                else:
                    QMessageBox.critical(self, "错误", "用户信息更新失败")
            else:
                # 创建新用户
                if self.user_manager.register_user(username, password, role, email, full_name):
                    # 记录创建日志
                    self.logger.log_user_action(
                        action="create_user",
                        description=f"创建新用户: {username}",
                        user_id=self.user_manager.current_user.user_id,
                        username=self.user_manager.current_user.username,
                        details={"new_user": username, "role": role.value}
                    )
                    
                    QMessageBox.information(self, "成功", "用户创建成功")
                    self.accept()
                else:
                    QMessageBox.critical(self, "错误", "用户创建失败")
                    
        except Exception as e:
            QMessageBox.critical(self, "错误", f"操作失败: {e}")
