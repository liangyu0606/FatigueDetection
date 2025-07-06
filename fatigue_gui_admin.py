#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
疲劳检测系统管理员界面
提供疲劳记录查询和统计功能
"""

import sys
import datetime
import csv
from typing import List, Dict, Optional, Tuple
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QPushButton, QLineEdit, QLabel,
    QComboBox, QDateTimeEdit, QGroupBox, QMessageBox, QHeaderView,
    QTabWidget, QTextEdit, QFileDialog, QDialog, QFormLayout, QGridLayout
)
from PyQt5.QtCore import Qt, QDateTime, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon

# 导入数据库配置
from database_config import get_db_connection, init_database


class AdminLoginDialog(QDialog):
    """管理员登录对话框"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("管理员登录")
        self.setFixedSize(1200, 750)  # 与用户登录界面大小一致
        self.setModal(True)

        # 默认管理员账号密码
        self.admin_credentials = {
            "admin": "admin123",
            "manager": "manager123",
            "root": "root123"
        }

        self.authenticated = False
        self._create_ui()

    def _create_ui(self):
        """创建登录界面"""
        layout = QVBoxLayout(self)
        layout.setSpacing(40)
        layout.setContentsMargins(60, 60, 60, 60)

        # 标题
        title_label = QLabel("疲劳检测系统管理员界面")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 48px; font-weight: bold; margin: 30px; color: #2e7d32;")
        layout.addWidget(title_label)

        # 添加弹性空间
        layout.addStretch(1)

        # 用户名区域
        username_layout = QHBoxLayout()
        username_label = QLabel("用户名:")
        username_label.setStyleSheet("font-size: 36px; font-weight: bold; min-width: 150px;")
        username_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.username_edit = QLineEdit()
        self.username_edit.setPlaceholderText("请输入管理员用户名")
        self.username_edit.setStyleSheet("padding: 30px; font-size: 42px; min-height: 45px; border: 2px solid #ccc; border-radius: 10px;")

        username_layout.addWidget(username_label)
        username_layout.addWidget(self.username_edit, 2)
        layout.addLayout(username_layout)

        # 密码区域
        password_layout = QHBoxLayout()
        password_label = QLabel("密码:")
        password_label.setStyleSheet("font-size: 36px; font-weight: bold; min-width: 150px;")
        password_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.password_edit = QLineEdit()
        self.password_edit.setPlaceholderText("请输入密码")
        self.password_edit.setEchoMode(QLineEdit.Password)
        self.password_edit.setStyleSheet("padding: 30px; font-size: 42px; min-height: 45px; border: 2px solid #ccc; border-radius: 10px;")

        password_layout.addWidget(password_label)
        password_layout.addWidget(self.password_edit, 2)
        layout.addLayout(password_layout)

        # 添加弹性空间
        layout.addStretch(1)

        # 提示信息标签
        self.message_label = QLabel("")
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                padding: 15px;
                border-radius: 8px;
                min-height: 30px;
            }
        """)
        self.message_label.hide()  # 初始隐藏
        layout.addWidget(self.message_label)

        # 登录按钮
        self.login_btn = QPushButton("登录")
        self.login_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 36px 60px;
                font-size: 42px;
                font-weight: bold;
                border-radius: 15px;
                min-height: 60px;
                min-width: 200px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.login_btn.clicked.connect(self._login)
        layout.addWidget(self.login_btn)

        # 添加底部弹性空间
        layout.addStretch(1)

        # 设置默认按钮和回车键绑定
        self.login_btn.setDefault(True)
        self.username_edit.returnPressed.connect(self._login)
        self.password_edit.returnPressed.connect(self._login)

        # 设置焦点
        self.username_edit.setFocus()

        # 添加关闭按钮的处理
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

    def _show_message(self, message, message_type="error"):
        """显示提示信息"""
        self.message_label.setText(message)

        if message_type == "error":
            self.message_label.setStyleSheet("""
                QLabel {
                    font-size: 24px;
                    font-weight: bold;
                    padding: 15px;
                    border-radius: 8px;
                    min-height: 30px;
                    background-color: #ffebee;
                    color: #c62828;
                    border: 2px solid #f44336;
                }
            """)
        elif message_type == "success":
            self.message_label.setStyleSheet("""
                QLabel {
                    font-size: 24px;
                    font-weight: bold;
                    padding: 15px;
                    border-radius: 8px;
                    min-height: 30px;
                    background-color: #e8f5e8;
                    color: #2e7d32;
                    border: 2px solid #4caf50;
                }
            """)

        self.message_label.show()

        # 3秒后自动隐藏提示信息
        QTimer.singleShot(3000, self._hide_message)

    def _hide_message(self):
        """隐藏提示信息"""
        self.message_label.hide()

    def _login(self):
        """执行登录验证"""
        username = self.username_edit.text().strip()
        password = self.password_edit.text()

        # 隐藏之前的提示信息
        self._hide_message()

        if not username or not password:
            self._show_message("请输入用户名和密码", "error")
            return

        # 验证管理员凭据
        if username in self.admin_credentials and self.admin_credentials[username] == password:
            self.authenticated = True
            self._show_message(f"登录成功，欢迎管理员 {username}!", "success")
            # 延迟关闭对话框，让用户看到成功提示
            QTimer.singleShot(1000, self.accept)
        else:
            self._show_message("用户名或密码错误，请重试", "error")
            self.password_edit.clear()
            self.password_edit.setFocus()

    def is_authenticated(self):
        """检查是否已认证"""
        return self.authenticated

    def closeEvent(self, event):
        """处理关闭事件"""
        if not self.authenticated:
            reply = QMessageBox.question(
                self, "确认退出", "确定要退出登录吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


class FatigueAdminGUI(QMainWindow):
    """疲劳检测系统管理员界面"""

    def __init__(self, admin_username="admin"):
        super().__init__()
        self.admin_username = admin_username
        self.setWindowTitle(f"疲劳检测系统 - 管理员界面 (当前用户: {admin_username})")
        self.setGeometry(100, 100, 1400, 900)

        # 初始化数据库
        self._init_database()

        # 设置样式
        self._setup_style()

        # 创建界面
        self._create_ui()

        # 加载初始数据
        self._load_initial_data()

        # 设置快捷键
        self._setup_shortcuts()

        # 设置状态栏
        self.statusBar().showMessage("就绪")
    
    def _init_database(self):
        """初始化数据库（简化版）"""
        try:
            init_database()
            print("数据库初始化成功")
        except Exception as e:
            print(f"数据库初始化失败: {e}")

    def _setup_shortcuts(self):
        """设置快捷键"""
        from PyQt5.QtWidgets import QShortcut
        from PyQt5.QtGui import QKeySequence

        # F5刷新快捷键
        refresh_shortcut = QShortcut(QKeySequence("F5"), self)
        refresh_shortcut.activated.connect(self._refresh_records)

        # Ctrl+R刷新快捷键
        refresh_shortcut2 = QShortcut(QKeySequence("Ctrl+R"), self)
        refresh_shortcut2.activated.connect(self._refresh_records)

    def _setup_style(self):
        """设置界面样式"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 8px 16px;
                text-align: center;
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QTableWidget {
                gridline-color: #d0d0d0;
                background-color: white;
                alternate-background-color: #f9f9f9;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QLineEdit, QComboBox, QDateTimeEdit {
                padding: 5px;
                border: 1px solid #ddd;
                border-radius: 3px;
            }
        """)
    
    def _create_ui(self):
        """创建用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QVBoxLayout(central_widget)

        # 管理员信息栏
        admin_info_group = QGroupBox("管理员信息")
        admin_info_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                padding: 20px;
                margin: 10px;
                min-height: 60px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px 0 10px;
            }
        """)
        admin_info_layout = QHBoxLayout(admin_info_group)

        admin_label = QLabel(f"当前管理员: {self.admin_username}")
        admin_label.setStyleSheet("font-weight: bold; font-size: 16px; color: #2e7d32; padding: 10px;")
        admin_info_layout.addWidget(admin_label)

        login_time_label = QLabel(f"登录时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        login_time_label.setStyleSheet("font-size: 12px; color: #666;")
        admin_info_layout.addWidget(login_time_label)

        admin_info_layout.addStretch()

        logout_btn = QPushButton("退出")
        logout_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                border: none;
                color: white;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        logout_btn.clicked.connect(self._logout)
        admin_info_layout.addWidget(logout_btn)

        main_layout.addWidget(admin_info_group)

        # 创建标签页
        tab_widget = QTabWidget()
        main_layout.addWidget(tab_widget)

        # 疲劳记录查询标签页
        fatigue_tab = self._create_fatigue_records_tab()
        tab_widget.addTab(fatigue_tab, "疲劳记录查询")

        # 统计分析标签页
        stats_tab = self._create_statistics_tab()
        tab_widget.addTab(stats_tab, "统计分析")

        # 用户管理标签页
        user_tab = self._create_user_management_tab()
        tab_widget.addTab(user_tab, "用户管理")
    
    def _create_user_management_tab(self):
        """创建用户管理标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # 用户列表
        user_group = QGroupBox("用户列表")
        user_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                padding: 15px;
                margin: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px 0 10px;
            }
        """)
        user_layout = QVBoxLayout(user_group)

        # 用户表格
        self.user_table = QTableWidget()
        self.user_table.setColumnCount(2)
        self.user_table.setHorizontalHeaderLabels([
            "用户名", "创建时间"
        ])
        self.user_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.user_table.setAlternatingRowColors(True)
        self.user_table.setStyleSheet("""
            QTableWidget {
                font-size: 14px;
                gridline-color: #ddd;
            }
            QTableWidget::item {
                padding: 8px;
                font-size: 14px;
            }
            QHeaderView::section {
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                background-color: #f0f0f0;
                border: 1px solid #ddd;
            }
        """)
        user_layout.addWidget(self.user_table)

        # 操作按钮
        button_layout = QHBoxLayout()

        refresh_btn = QPushButton("刷新用户列表")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
                min-height: 25px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        refresh_btn.clicked.connect(self._refresh_users)
        button_layout.addWidget(refresh_btn)

        add_user_btn = QPushButton("添加用户")
        add_user_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                border: none;
                color: white;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
                min-height: 25px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        add_user_btn.clicked.connect(self._add_user)
        button_layout.addWidget(add_user_btn)

        button_layout.addStretch()
        user_layout.addLayout(button_layout)

        layout.addWidget(user_group)
        return widget
    
    def _create_fatigue_records_tab(self):
        """创建疲劳记录查询标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 搜索条件
        search_group = QGroupBox("搜索条件")
        search_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                padding: 15px;
                margin: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px 0 10px;
            }
        """)
        search_layout = QHBoxLayout(search_group)
        
        # 用户名搜索
        username_label = QLabel("用户名:")
        username_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        search_layout.addWidget(username_label)
        self.username_search = QLineEdit()
        self.username_search.setPlaceholderText("输入用户名进行搜索")
        self.username_search.setStyleSheet("font-size: 14px; padding: 8px; min-height: 20px;")
        search_layout.addWidget(self.username_search)
        
        # 疲劳状态搜索
        fatigue_label = QLabel("疲劳状态:")
        fatigue_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        search_layout.addWidget(fatigue_label)
        self.fatigue_status_combo = QComboBox()
        self.fatigue_status_combo.addItems(["全部", "轻度疲劳", "中度疲劳", "重度疲劳"])
        self.fatigue_status_combo.setStyleSheet("font-size: 14px; padding: 8px; min-height: 20px;")
        search_layout.addWidget(self.fatigue_status_combo)
        
        # 时间范围（可选）
        start_label = QLabel("开始时间:")
        start_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        search_layout.addWidget(start_label)
        self.start_datetime = QDateTimeEdit()
        self.start_datetime.setSpecialValueText("不限制")  # 空值显示文本
        self.start_datetime.setDateTime(QDateTime.currentDateTime().addDays(-30))
        self.start_datetime.setCalendarPopup(True)
        self.start_datetime.setStyleSheet("font-size: 14px; padding: 8px; min-height: 20px;")
        self.start_datetime.clear()  # 初始为空
        search_layout.addWidget(self.start_datetime)

        end_label = QLabel("结束时间:")
        end_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        search_layout.addWidget(end_label)
        self.end_datetime = QDateTimeEdit()
        self.end_datetime.setSpecialValueText("不限制")  # 空值显示文本
        self.end_datetime.setDateTime(QDateTime.currentDateTime())
        self.end_datetime.setCalendarPopup(True)
        self.end_datetime.setStyleSheet("font-size: 14px; padding: 8px; min-height: 20px;")
        self.end_datetime.clear()  # 初始为空
        search_layout.addWidget(self.end_datetime)
        
        # 页面大小设置
        page_size_label = QLabel("每页显示:")
        page_size_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        search_layout.addWidget(page_size_label)
        self.page_size_combo = QComboBox()
        self.page_size_combo.addItems(["10", "20", "50", "100"])
        self.page_size_combo.setCurrentText("20")  # 默认20条
        self.page_size_combo.setStyleSheet("font-size: 14px; padding: 8px; min-height: 20px;")
        self.page_size_combo.currentTextChanged.connect(self._on_page_size_changed)
        search_layout.addWidget(self.page_size_combo)

        # 搜索和刷新按钮
        search_btn = QPushButton("搜索")
        search_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                border: none;
                color: white;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
                min-height: 25px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        search_btn.clicked.connect(self._search_fatigue_records)
        search_layout.addWidget(search_btn)

        clear_btn = QPushButton("重置")
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                border: none;
                color: white;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
                min-height: 25px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        clear_btn.clicked.connect(self._clear_search)
        search_layout.addWidget(clear_btn)

        refresh_btn = QPushButton("刷新")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
                min-height: 25px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        refresh_btn.clicked.connect(self._refresh_records)
        search_layout.addWidget(refresh_btn)

        layout.addWidget(search_group)
        
        # 疲劳记录表格
        records_group = QGroupBox("疲劳记录")
        records_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                padding: 15px;
                margin: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px 0 10px;
            }
        """)
        records_layout = QVBoxLayout(records_group)

        self.fatigue_table = QTableWidget()
        self.fatigue_table.setColumnCount(3)
        self.fatigue_table.setHorizontalHeaderLabels([
            "用户名", "时间", "疲劳等级"
        ])
        self.fatigue_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.fatigue_table.setAlternatingRowColors(True)

        # 设置表格字体大小
        self.fatigue_table.setStyleSheet("""
            QTableWidget {
                font-size: 14px;
                gridline-color: #ddd;
            }
            QTableWidget::item {
                padding: 8px;
                font-size: 14px;
            }
            QHeaderView::section {
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                background-color: #f0f0f0;
                border: 1px solid #ddd;
            }
        """)
        records_layout.addWidget(self.fatigue_table)

        # 分页控件
        pagination_layout = QHBoxLayout()

        self.prev_btn = QPushButton("上一页")
        self.prev_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
                min-height: 25px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.prev_btn.clicked.connect(self._prev_page)
        self.prev_btn.setEnabled(False)
        pagination_layout.addWidget(self.prev_btn)

        self.page_label = QLabel("第 1 页，共 0 条记录")
        self.page_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        pagination_layout.addWidget(self.page_label)

        self.next_btn = QPushButton("下一页")
        self.next_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
                min-height: 25px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.next_btn.clicked.connect(self._next_page)
        self.next_btn.setEnabled(False)
        pagination_layout.addWidget(self.next_btn)

        pagination_layout.addStretch()

        # 操作按钮
        export_btn = QPushButton("导出到CSV")
        export_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                border: none;
                color: white;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
                min-height: 25px;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
        """)
        export_btn.clicked.connect(self.export_records_to_csv)
        pagination_layout.addWidget(export_btn)

        clear_btn = QPushButton("清空搜索")
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF5722;
                border: none;
                color: white;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
                min-height: 25px;
            }
            QPushButton:hover {
                background-color: #E64A19;
            }
        """)
        clear_btn.clicked.connect(self._clear_search)
        pagination_layout.addWidget(clear_btn)

        records_layout.addLayout(pagination_layout)

        # 初始化分页变量
        self.current_page = 1
        self.total_records = 0

        layout.addWidget(records_group)
        return widget
    
    def _create_statistics_tab(self):
        """创建统计分析标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # 统计信息显示 - 方格模式
        stats_group = QGroupBox("统计信息")
        stats_group.setStyleSheet("""
            QGroupBox {
                font-size: 18px;
                font-weight: bold;
                padding: 25px;
                margin: 15px;
                background-color: white;
                border: 2px solid #ddd;
                border-radius: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 20px;
                padding: 0 15px 0 15px;
                background-color: white;
            }
        """)
        stats_layout = QVBoxLayout(stats_group)

        # 创建方格布局
        grid_layout = QGridLayout()

        # 第一行：基础统计
        self.total_records_label = QLabel()
        self.total_records_label.setStyleSheet("""
            QLabel {
                border: 2px solid red;
                padding: 20px;
                background-color: white;
                font-size: 16px;
                font-weight: bold;
                min-height: 80px;
                min-width: 150px;
                text-align: center;
            }
        """)
        grid_layout.addWidget(self.total_records_label, 0, 0)

        self.total_users_label = QLabel()
        self.total_users_label.setStyleSheet("""
            QLabel {
                border: 2px solid red;
                padding: 20px;
                background-color: white;
                font-size: 16px;
                font-weight: bold;
                min-height: 80px;
                min-width: 150px;
                text-align: center;
            }
        """)
        grid_layout.addWidget(self.total_users_label, 0, 1)

        self.recent_records_label = QLabel()
        self.recent_records_label.setStyleSheet("""
            QLabel {
                border: 2px solid red;
                padding: 20px;
                background-color: white;
                font-size: 16px;
                font-weight: bold;
                min-height: 80px;
                min-width: 150px;
                text-align: center;
            }
        """)
        grid_layout.addWidget(self.recent_records_label, 0, 2)

        # 第二行：疲劳等级统计
        self.mild_fatigue_label = QLabel()
        self.mild_fatigue_label.setStyleSheet("""
            QLabel {
                border: 2px solid red;
                padding: 20px;
                background-color: white;
                font-size: 16px;
                font-weight: bold;
                min-height: 80px;
                min-width: 150px;
                text-align: center;
            }
        """)
        grid_layout.addWidget(self.mild_fatigue_label, 1, 0)

        self.moderate_fatigue_label = QLabel()
        self.moderate_fatigue_label.setStyleSheet("""
            QLabel {
                border: 2px solid red;
                padding: 20px;
                background-color: white;
                font-size: 16px;
                font-weight: bold;
                min-height: 80px;
                min-width: 150px;
                text-align: center;
            }
        """)
        grid_layout.addWidget(self.moderate_fatigue_label, 1, 1)

        self.severe_fatigue_label = QLabel()
        self.severe_fatigue_label.setStyleSheet("""
            QLabel {
                border: 2px solid red;
                padding: 20px;
                background-color: white;
                font-size: 16px;
                font-weight: bold;
                min-height: 80px;
                min-width: 150px;
                text-align: center;
            }
        """)
        grid_layout.addWidget(self.severe_fatigue_label, 1, 2)

        stats_layout.addLayout(grid_layout)

        # 刷新统计按钮
        refresh_stats_btn = QPushButton("刷新统计")
        refresh_stats_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 12px 24px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 6px;
                margin: 15px;
                max-width: 120px;
                min-height: 40px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        refresh_stats_btn.clicked.connect(self._refresh_statistics)

        # 创建按钮布局，使按钮居中
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(refresh_stats_btn)
        button_layout.addStretch()
        stats_layout.addLayout(button_layout)

        layout.addWidget(stats_group)
        
        # 详细统计表格
        detail_group = QGroupBox("用户疲劳统计详情")
        detail_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                padding: 15px;
                margin: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px 0 10px;
            }
        """)
        detail_layout = QVBoxLayout(detail_group)

        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(5)
        self.stats_table.setHorizontalHeaderLabels([
            "用户名", "轻度疲劳", "中度疲劳", "重度疲劳", "最后记录时间"
        ])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.stats_table.setAlternatingRowColors(True)
        self.stats_table.setStyleSheet("""
            QTableWidget {
                font-size: 14px;
                gridline-color: #ddd;
            }
            QTableWidget::item {
                padding: 8px;
                font-size: 14px;
            }
            QHeaderView::section {
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                background-color: #f0f0f0;
                border: 1px solid #ddd;
            }
        """)
        detail_layout.addWidget(self.stats_table)
        
        layout.addWidget(detail_group)
        return widget
    
    def _load_initial_data(self):
        """加载初始数据"""
        self._load_users()
        self._load_fatigue_records()
        self._load_statistics()
    
    def _load_users(self):
        """加载用户列表"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    SELECT username, created_at
                    FROM users
                    ORDER BY created_at DESC
                ''')

                users = cursor.fetchall()

            self.user_table.setRowCount(len(users))

            for row, user in enumerate(users):
                for col, value in enumerate(user):
                    if value is None:
                        value = ""
                    self.user_table.setItem(row, col, QTableWidgetItem(str(value)))

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载用户列表失败: {e}")

    def _refresh_users(self):
        """刷新用户列表"""
        self.statusBar().showMessage("正在刷新用户列表...")
        self._load_users()
        self.statusBar().showMessage("用户列表已刷新", 3000)
        print("用户列表已刷新")

    def _add_user(self):
        """添加用户对话框"""
        from PyQt5.QtWidgets import QDialog, QFormLayout

        dialog = QDialog(self)
        dialog.setWindowTitle("添加用户")
        dialog.setFixedSize(300, 150)

        layout = QFormLayout(dialog)

        username_edit = QLineEdit()
        username_edit.setPlaceholderText("请输入用户名")
        layout.addRow("用户名:", username_edit)

        password_edit = QLineEdit()
        password_edit.setPlaceholderText("请输入密码")
        password_edit.setEchoMode(QLineEdit.Password)
        layout.addRow("密码:", password_edit)

        button_layout = QHBoxLayout()
        ok_btn = QPushButton("确定")
        cancel_btn = QPushButton("取消")

        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addRow(button_layout)

        def add_user():
            username = username_edit.text().strip()
            password = password_edit.text().strip()

            if not username:
                QMessageBox.warning(dialog, "警告", "用户名不能为空")
                return

            if not password:
                QMessageBox.warning(dialog, "警告", "密码不能为空")
                return

            try:
                with get_db_connection() as conn:
                    cursor = conn.cursor()

                    cursor.execute('''
                        INSERT INTO users (username, password)
                        VALUES (%s, %s)
                    ''', (username, password))

                    conn.commit()

                QMessageBox.information(dialog, "成功", "用户添加成功")
                dialog.accept()
                self._load_users()

            except Exception as e:
                if "Duplicate entry" in str(e):
                    QMessageBox.critical(dialog, "错误", "用户名已存在")
                else:
                    QMessageBox.critical(dialog, "错误", f"添加用户失败: {e}")

        ok_btn.clicked.connect(add_user)
        cancel_btn.clicked.connect(dialog.reject)

        dialog.exec_()

    def _logout(self):
        """退出程序"""
        reply = QMessageBox.question(
            self, "确认退出", "确定要退出管理员界面吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.close()
            QApplication.quit()

    def _search_fatigue_records(self):
        """搜索疲劳记录"""
        self.current_page = 1  # 重置到第一页
        self._load_fatigue_records()

    def _refresh_records(self):
        """刷新记录"""
        self.statusBar().showMessage("正在刷新所有数据...")
        # 重新加载所有数据
        self._load_initial_data()
        self.statusBar().showMessage("所有数据已刷新", 3000)
        print("所有数据已刷新")

    def _load_fatigue_records(self):
        """加载疲劳记录（支持分页）"""
        try:
            # 获取搜索条件
            username = self.username_search.text().strip()
            fatigue_status = self.fatigue_status_combo.currentText()
            page_size = int(self.page_size_combo.currentText())
            offset = (self.current_page - 1) * page_size

            # 获取时间条件
            start_time = None
            end_time = None

            if not self.start_datetime.text() == "不限制":
                start_time = self.start_datetime.dateTime().toPyDateTime()

            if not self.end_datetime.text() == "不限制":
                end_time = self.end_datetime.dateTime().toPyDateTime()

            with get_db_connection() as conn:
                cursor = conn.cursor()

                # 构建查询条件
                where_conditions = []
                params = []

                # 时间条件
                if start_time and end_time:
                    where_conditions.append("timestamp BETWEEN %s AND %s")
                    params.extend([start_time, end_time])
                elif start_time:
                    where_conditions.append("timestamp >= %s")
                    params.append(start_time)
                elif end_time:
                    where_conditions.append("timestamp <= %s")
                    params.append(end_time)

                # 用户名条件
                if username:
                    where_conditions.append("username LIKE %s")
                    params.append(f"%{username}%")

                # 疲劳状态条件
                if fatigue_status != "全部":
                    where_conditions.append("fatigue_level = %s")
                    params.append(fatigue_status)

                # 构建WHERE子句
                where_clause = ""
                if where_conditions:
                    where_clause = "WHERE " + " AND ".join(where_conditions)

                # 查询总记录数
                count_query = f"SELECT COUNT(*) FROM fatigue_records {where_clause}"
                cursor.execute(count_query, params)
                self.total_records = cursor.fetchone()[0]

                # 查询分页数据
                query = f'''
                    SELECT username, timestamp, fatigue_level
                    FROM fatigue_records
                    {where_clause}
                    ORDER BY timestamp DESC
                    LIMIT %s OFFSET %s
                '''
                cursor.execute(query, params + [page_size, offset])
                records = cursor.fetchall()

            # 更新表格
            self._update_table(records)

            # 更新分页信息
            self._update_pagination()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载疲劳记录失败: {e}")

    def _update_table(self, records):
        """更新表格数据"""
        self.fatigue_table.setRowCount(len(records))

        for row, record in enumerate(records):
            for col, value in enumerate(record):
                if value is None:
                    value = ""
                elif col == 1:  # 时间戳格式化 (第2列)
                    try:
                        if isinstance(value, str):
                            dt = datetime.datetime.fromisoformat(value.replace('Z', ''))
                            value = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        pass

                item = QTableWidgetItem(str(value))

                # 根据疲劳等级设置颜色
                if col == 2:  # 疲劳等级列 (第3列)
                    if value == "重度疲劳":
                        item.setBackground(QColor(255, 200, 200))
                    elif value == "中度疲劳":
                        item.setBackground(QColor(255, 230, 200))
                    elif value == "轻度疲劳":
                        item.setBackground(QColor(255, 255, 200))

                self.fatigue_table.setItem(row, col, item)

    def _update_pagination(self):
        """更新分页信息"""
        page_size = int(self.page_size_combo.currentText())
        total_pages = (self.total_records + page_size - 1) // page_size

        # 更新页面标签
        self.page_label.setText(f"第 {self.current_page} 页，共 {self.total_records} 条记录")

        # 更新按钮状态
        self.prev_btn.setEnabled(self.current_page > 1)
        self.next_btn.setEnabled(self.current_page < total_pages)

    def _prev_page(self):
        """上一页"""
        if self.current_page > 1:
            self.current_page -= 1
            self._load_fatigue_records()

    def _next_page(self):
        """下一页"""
        page_size = int(self.page_size_combo.currentText())
        total_pages = (self.total_records + page_size - 1) // page_size
        if self.current_page < total_pages:
            self.current_page += 1
            self._load_fatigue_records()

    def _on_page_size_changed(self):
        """页面大小改变时的处理"""
        self.current_page = 1  # 重置到第一页
        self._load_fatigue_records()

    def _clear_search(self):
        """清空搜索条件"""
        self.username_search.clear()
        self.fatigue_status_combo.setCurrentIndex(0)
        self.start_datetime.clear()
        self.end_datetime.clear()
        self.page_size_combo.setCurrentText("20")
        self.current_page = 1
        self._load_fatigue_records()
        self.statusBar().showMessage("搜索条件已重置", 2000)

    def _load_statistics(self):
        """加载统计信息"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()

                # 总体统计
                cursor.execute("SELECT COUNT(*) FROM fatigue_records")
                total_records = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(DISTINCT username) FROM fatigue_records")
                total_users = cursor.fetchone()[0]

                cursor.execute("""
                    SELECT fatigue_level, COUNT(*)
                    FROM fatigue_records
                    WHERE fatigue_level IN ('轻度疲劳', '中度疲劳', '重度疲劳')
                    GROUP BY fatigue_level
                """)
                fatigue_stats = dict(cursor.fetchall())

                # 最近7天的记录
                week_ago = datetime.datetime.now() - datetime.timedelta(days=7)
                cursor.execute("""
                    SELECT COUNT(*) FROM fatigue_records
                    WHERE timestamp >= %s
                """, (week_ago,))
                recent_records = cursor.fetchone()[0]

                # 更新方格显示
                self.total_records_label.setText(f"总记录数:\n{total_records}")
                self.total_users_label.setText(f"活跃用户数:\n{total_users}")
                self.recent_records_label.setText(f"最近7天记录:\n{recent_records}")

                self.mild_fatigue_label.setText(f"轻度疲劳:\n{fatigue_stats.get('轻度疲劳', 0)} 次")
                self.moderate_fatigue_label.setText(f"中度疲劳:\n{fatigue_stats.get('中度疲劳', 0)} 次")
                self.severe_fatigue_label.setText(f"重度疲劳:\n{fatigue_stats.get('重度疲劳', 0)} 次")

                # 用户详细统计
                cursor.execute("""
                    SELECT
                        username,
                        SUM(CASE WHEN fatigue_level = '轻度疲劳' THEN 1 ELSE 0 END) as mild_count,
                        SUM(CASE WHEN fatigue_level = '中度疲劳' THEN 1 ELSE 0 END) as moderate_count,
                        SUM(CASE WHEN fatigue_level = '重度疲劳' THEN 1 ELSE 0 END) as severe_count,
                        MAX(timestamp) as last_record
                    FROM fatigue_records
                    GROUP BY username
                    ORDER BY (SUM(CASE WHEN fatigue_level = '轻度疲劳' THEN 1 ELSE 0 END) +
                             SUM(CASE WHEN fatigue_level = '中度疲劳' THEN 1 ELSE 0 END) +
                             SUM(CASE WHEN fatigue_level = '重度疲劳' THEN 1 ELSE 0 END)) DESC
                """)

                user_stats = cursor.fetchall()

            # 更新用户统计表格
            self.stats_table.setRowCount(len(user_stats))

            for row, stats in enumerate(user_stats):
                for col, value in enumerate(stats):
                    if value is None:
                        value = ""
                    elif col == 4:  # 最后记录时间格式化
                        try:
                            if isinstance(value, str):
                                dt = datetime.datetime.fromisoformat(value.replace('Z', ''))
                                value = dt.strftime("%Y-%m-%d %H:%M:%S")
                        except:
                            pass

                    self.stats_table.setItem(row, col, QTableWidgetItem(str(value)))

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载统计信息失败: {e}")

    def _refresh_statistics(self):
        """刷新统计信息"""
        self.statusBar().showMessage("正在刷新统计信息...")
        self._load_statistics()
        self.statusBar().showMessage("统计信息已刷新", 3000)
        print("统计信息已刷新")

    def add_fatigue_record(self, username: str, fatigue_level: str,
                          fatigue_score: float = 0.0, event_type: str = "detection",
                          confidence: float = 0.0, duration: float = 0.0,
                          additional_info: str = ""):
        """
        添加疲劳记录（供其他模块调用）

        Args:
            username: 用户名
            fatigue_level: 疲劳等级
            fatigue_score: 疲劳评分
            event_type: 事件类型
            confidence: 置信度
            duration: 持续时间
            additional_info: 附加信息
        """
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()

                # 确保用户存在
                cursor.execute("SELECT username FROM users WHERE username = %s", (username,))
                if not cursor.fetchone():
                    cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, "123456"))

                cursor.execute('''
                    INSERT INTO fatigue_records
                    (username, timestamp, fatigue_level)
                    VALUES (%s, %s, %s)
                ''', (username, datetime.datetime.now(), fatigue_level))

                conn.commit()

            print(f"疲劳记录已添加: {username} - {fatigue_level}")

        except Exception as e:
            print(f"添加疲劳记录失败: {e}")

    def export_records_to_csv(self):
        """导出当前搜索结果到CSV文件"""
        try:
            # 选择保存文件
            filename, _ = QFileDialog.getSaveFileName(
                self, "导出疲劳记录", "fatigue_records.csv", "CSV Files (*.csv)"
            )

            if not filename:
                return

            # 获取当前搜索条件的所有记录（不分页）
            username = self.username_search.text().strip()
            fatigue_status = self.fatigue_status_combo.currentText()

            # 获取时间条件
            start_time = None
            end_time = None

            if not self.start_datetime.text() == "不限制":
                start_time = self.start_datetime.dateTime().toPyDateTime()

            if not self.end_datetime.text() == "不限制":
                end_time = self.end_datetime.dateTime().toPyDateTime()

            with get_db_connection() as conn:
                cursor = conn.cursor()

                # 构建查询条件（与搜索相同的逻辑）
                where_conditions = []
                params = []

                # 时间条件
                if start_time and end_time:
                    where_conditions.append("timestamp BETWEEN %s AND %s")
                    params.extend([start_time, end_time])
                elif start_time:
                    where_conditions.append("timestamp >= %s")
                    params.append(start_time)
                elif end_time:
                    where_conditions.append("timestamp <= %s")
                    params.append(end_time)

                # 用户名条件
                if username:
                    where_conditions.append("username LIKE %s")
                    params.append(f"%{username}%")

                # 疲劳状态条件
                if fatigue_status != "全部":
                    where_conditions.append("fatigue_level = %s")
                    params.append(fatigue_status)

                # 构建WHERE子句
                where_clause = ""
                if where_conditions:
                    where_clause = "WHERE " + " AND ".join(where_conditions)

                # 查询所有符合条件的记录
                query = f'''
                    SELECT username, timestamp, fatigue_level
                    FROM fatigue_records
                    {where_clause}
                    ORDER BY timestamp DESC
                '''
                cursor.execute(query, params)
                records = cursor.fetchall()

            # 写入CSV文件
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['用户名', '时间', '疲劳等级'])

                # 格式化时间戳
                for record in records:
                    formatted_record = list(record)
                    try:
                        if isinstance(formatted_record[1], str):  # 时间戳现在是第2列
                            dt = datetime.datetime.fromisoformat(formatted_record[1].replace('Z', ''))
                            formatted_record[1] = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        pass
                    writer.writerow(formatted_record)

            QMessageBox.information(self, "成功", f"已导出 {len(records)} 条记录到: {filename}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败: {e}")


# 全局管理员界面实例（用于其他模块调用）
admin_gui_instance = None


def get_admin_gui():
    """获取管理员界面实例"""
    global admin_gui_instance
    return admin_gui_instance


def set_admin_gui(gui):
    """设置管理员界面实例"""
    global admin_gui_instance
    admin_gui_instance = gui


def record_user_fatigue(username: str, fatigue_level: str, **kwargs):
    """
    记录用户疲劳状态（供其他模块调用的便捷函数）

    Args:
        username: 用户名
        fatigue_level: 疲劳等级
        **kwargs: 其他参数
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # 确保用户存在（为测试数据创建用户时使用默认密码）
            cursor.execute("SELECT username FROM users WHERE username = %s", (username,))
            if not cursor.fetchone():
                # 为测试用户创建默认密码
                default_password = "123456"
                cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, default_password))

            # 插入疲劳记录
            cursor.execute('''
                INSERT INTO fatigue_records
                (username, timestamp, fatigue_level)
                VALUES (%s, %s, %s)
            ''', (
                username,
                datetime.datetime.now(),
                fatigue_level
            ))

            conn.commit()

        print(f"疲劳记录已记录: {username} - {fatigue_level}")

    except Exception as e:
        print(f"记录疲劳状态失败: {e}")


def add_test_data():
    """添加测试数据"""
    import random

    # 首先检查是否已经有疲劳记录
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM fatigue_records")
            record_count = cursor.fetchone()[0]

            if record_count > 0:
                print("数据库中已存在疲劳记录，跳过测试数据添加")
                return

            # 检查是否已经有测试用户
            test_users = ["张三", "李四", "王五", "赵六", "钱七"]
            placeholders = ','.join(['%s' for _ in test_users])
            cursor.execute(f"SELECT COUNT(*) FROM fatigue_records WHERE username IN ({placeholders})", test_users)
            test_record_count = cursor.fetchone()[0]

            if test_record_count > 0:
                print("数据库中已存在测试数据，跳过重复添加")
                return

    except Exception as e:
        print(f"检查数据库状态失败: {e}")
        return

    fatigue_levels = ["轻度疲劳", "中度疲劳", "重度疲劳"]

    print("开始添加测试数据...")

    for user in test_users:
        # 为每个用户生成一些随机的疲劳记录
        for _ in range(random.randint(5, 15)):
            fatigue_level = random.choice(fatigue_levels)
            record_user_fatigue(
                username=user,
                fatigue_level=fatigue_level
            )

    print("测试数据添加完成")


def reset_database():
    """重置数据库"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # 删除现有表
            cursor.execute("DROP TABLE IF EXISTS fatigue_records")
            cursor.execute("DROP TABLE IF EXISTS users")

            conn.commit()
            print("现有表已删除")

        # 重新初始化数据库
        init_database()
        print("数据库重置成功")

    except Exception as e:
        print(f"数据库重置失败: {e}")


def main():
    """主函数"""
    app = QApplication(sys.argv)

    # 设置应用程序属性
    app.setApplicationName("疲劳检测系统管理员界面")
    app.setApplicationVersion("1.0")

    # 显示登录对话框
    login_dialog = AdminLoginDialog()
    if login_dialog.exec_() != QDialog.Accepted or not login_dialog.is_authenticated():
        print("登录取消或失败，程序退出")
        sys.exit(0)

    # 获取登录的用户名
    admin_username = login_dialog.username_edit.text().strip()

    # 检查数据库结构
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT username, created_at FROM users LIMIT 1")
    except:
        print("检测到数据库结构问题，正在重置数据库...")
        reset_database()

    # 创建并显示主窗口
    window = FatigueAdminGUI(admin_username)
    window.show()

    # 检查数据库是否为空，只有为空时才询问是否添加测试数据
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # 只检查疲劳记录是否为空，因为用户登录时会自动创建用户记录
            cursor.execute("SELECT COUNT(*) FROM fatigue_records")
            record_count = cursor.fetchone()[0]

        # 只有在没有任何疲劳记录时才询问是否添加测试数据
        if record_count == 0:
            reply = QMessageBox.question(
                window, "测试数据", "检测到没有疲劳记录，是否添加测试数据？\n（建议添加以便查看界面效果）",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )

            if reply == QMessageBox.Yes:
                add_test_data()
                window._load_initial_data()  # 重新加载数据

    except Exception as e:
        print(f"检查数据库状态失败: {e}")

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
