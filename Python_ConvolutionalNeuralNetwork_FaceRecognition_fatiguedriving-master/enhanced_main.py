#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版主程序 - 集成用户权限管理和系统日志功能
基于原有的疲劳检测系统，添加用户管理和权限控制
"""

import sys
import os
import datetime
import threading
from typing import Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QMenuBar, QMenu, QStatusBar, QLabel,
    QMessageBox, QDialog, QSplitter, QFrame
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QIcon, QAction

# 导入用户管理和日志模块
from user_management import UserManager, UserRole, Permission
from system_logger import SystemLogger, LogLevel, LogCategory
from user_interface import LoginDialog, UserManagementWidget

# 尝试导入日志查看器，如果失败则使用占位符
try:
    from log_viewer import LogViewerWidget
    LOG_VIEWER_AVAILABLE = True
except Exception as e:  # 捕获所有异常，不仅仅是ImportError
    print(f"日志查看器导入失败: {e}")
    LOG_VIEWER_AVAILABLE = False
    # 创建占位符类
    class LogViewerWidget:
        def __init__(self, *args, **kwargs):
            pass

# 尝试导入PyTorch用于CNN+LSTM打哈欠检测
try:
    import torch
    import torch.nn as nn
    import numpy as np
    PYTORCH_AVAILABLE = True
    print("✅ PyTorch可用，将启用CNN+LSTM打哈欠检测")
except ImportError as e:
    print(f"PyTorch导入失败: {e}")
    PYTORCH_AVAILABLE = False

# 导入原有的疲劳检测模块
try:
    from main import MainUI, YawnCNNLSTM, YawnDetector
    MAIN_UI_AVAILABLE = True
    # 如果main.py中有CNN+LSTM相关类，也导入
    if PYTORCH_AVAILABLE:
        print("✅ 成功导入CNN+LSTM相关类")
except ImportError:
    MAIN_UI_AVAILABLE = False
    print("警告: 无法导入原有的主界面模块")
    # 如果无法从main导入，定义占位符类
    if not PYTORCH_AVAILABLE:
        class YawnCNNLSTM:
            def __init__(self, *args, **kwargs):
                pass

        class YawnDetector:
            def __init__(self, *args, **kwargs):
                self.is_available = False

from fatigue_statistics import FatigueStatistics


class EnhancedFatigueDetectionSystem(QMainWindow):
    """增强版疲劳检测系统主窗口"""
    
    # 信号定义
    user_logged_in = Signal(str)
    user_logged_out = Signal()
    
    def __init__(self):
        super().__init__()
        
        # 初始化管理器
        self.user_manager = UserManager()
        self.logger = SystemLogger()
        self.fatigue_stats = FatigueStatistics()
        
        # 当前用户信息
        self.current_user = None
        self.session_token = None
        
        # 原有的疲劳检测界面
        self.fatigue_ui = None

        # 初始化CNN+LSTM打哈欠检测器
        self.yawn_detector = None
        self.init_yawn_detector()

        # 界面组件
        self.central_widget = None
        self.tab_widget = None
        self.status_bar = None

        # 初始化事件计数器
        self._last_blink_count = 0
        self._last_yawn_count = 0
        self._last_nod_count = 0
        
        # 设置窗口
        self.setup_window()
        self.setup_menu()
        self.setup_status_bar()
        
        # 显示登录对话框
        self.show_login_dialog()

        # 记录系统启动日志
        self.logger.log_system_event(
            action="system_start",
            description="疲劳检测系统启动",
            level=LogLevel.INFO
        )

    def connect_detection_to_stats(self):
        """连接疲劳检测模块和统计模块"""
        if self.fatigue_ui and hasattr(self.fatigue_ui, 'fatigue_stats'):
            # 让main.py使用enhanced_main.py的统计实例，避免重复记录
            self.fatigue_ui.fatigue_stats = self.fatigue_stats
            print("✅ 疲劳检测模块已连接到统计数据库")

            # 可选：连接信号用于UI更新，但不重复记录数据
            if hasattr(self.fatigue_ui, 'thread_signal'):
                self.fatigue_ui.thread_signal.connect(self.on_detection_update)
                print("✅ 疲劳检测信号已连接到UI更新")

    def on_detection_update(self, data):
        """处理疲劳检测更新 - 仅用于UI更新，不重复记录数据"""
        try:
            if not isinstance(data, dict):
                return

            data_type = data.get('type', '')

            # 仅处理UI更新，数据记录已在main.py中完成
            if data_type == 'msg':
                message = data.get('value', '')
                # 可以在这里添加UI状态更新逻辑
                # 例如：更新状态栏、发送通知等
                if any(keyword in message for keyword in ['眨眼', '哈欠', '点头', 'CNN检测到疲劳']):
                    # 更新状态栏显示最新检测事件
                    if hasattr(self, 'status_bar'):
                        self.status_bar.showMessage(f"检测事件: {message}", 3000)

            elif data_type == 'res':
                values = data.get('value', [])
                if len(values) > 1:
                    fatigue_level_str = values[1]
                    # 可以在这里更新UI显示当前疲劳状态
                    # 例如：改变状态指示器颜色等
                    pass

        except Exception as e:
            print(f"处理UI更新失败: {e}")
    
    def setup_window(self):
        """设置主窗口"""
        self.setWindowTitle("🚗 智能疲劳驾驶检测系统 - 企业版")
        self.setGeometry(100, 100, 1400, 900)
        
        # 设置窗口图标（如果有的话）
        # self.setWindowIcon(QIcon("icon.png"))
        
        # 创建中央组件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 主布局
        main_layout = QVBoxLayout()
        self.central_widget.setLayout(main_layout)
        
        # 创建选项卡组件
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # 初始时隐藏，等待登录
        self.central_widget.setVisible(False)
    
    def setup_menu(self):
        """设置菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件(&F)")
        
        # 登录/登出
        self.login_action = QAction("登录(&L)", self)
        self.login_action.triggered.connect(self.show_login_dialog)
        file_menu.addAction(self.login_action)
        
        self.logout_action = QAction("登出(&O)", self)
        self.logout_action.triggered.connect(self.logout)
        self.logout_action.setEnabled(False)
        file_menu.addAction(self.logout_action)
        
        file_menu.addSeparator()
        
        # 退出
        exit_action = QAction("退出(&X)", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 系统菜单
        self.system_menu = menubar.addMenu("系统(&S)")
        
        # 用户管理
        self.user_mgmt_action = QAction("用户管理(&U)", self)
        self.user_mgmt_action.triggered.connect(self.show_user_management)
        self.user_mgmt_action.setEnabled(False)
        self.system_menu.addAction(self.user_mgmt_action)
        
        # 系统日志
        self.log_viewer_action = QAction("系统日志(&L)", self)
        self.log_viewer_action.triggered.connect(self.show_log_viewer)
        self.log_viewer_action.setEnabled(False)
        self.system_menu.addAction(self.log_viewer_action)

        # 分隔符
        self.system_menu.addSeparator()

        # CNN+LSTM状态
        self.cnn_lstm_status_action = QAction("CNN+LSTM状态(&C)", self)
        self.cnn_lstm_status_action.triggered.connect(self.show_cnn_lstm_status)
        self.system_menu.addAction(self.cnn_lstm_status_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助(&H)")
        
        about_action = QAction("关于(&A)", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_status_bar(self):
        """设置状态栏"""
        self.status_bar = self.statusBar()
        
        # 用户信息标签
        self.user_label = QLabel("未登录")
        self.status_bar.addPermanentWidget(self.user_label)
        
        # 时间标签
        self.time_label = QLabel()
        self.status_bar.addPermanentWidget(self.time_label)
        
        # 更新时间的定时器
        self.time_timer = QTimer()
        self.time_timer.timeout.connect(self.update_time)
        self.time_timer.start(1000)  # 每秒更新
        
        self.update_time()
    
    def update_time(self):
        """更新时间显示"""
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.setText(current_time)

    def init_yawn_detector(self):
        """初始化CNN+LSTM打哈欠检测器"""
        try:
            if not PYTORCH_AVAILABLE:
                print("⚠️ PyTorch不可用，跳过CNN+LSTM打哈欠检测器初始化")
                return

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
                    print("✅ Enhanced版本: CNN+LSTM打哈欠检测器已加载")
                    # 记录到系统日志
                    self.logger.log_system_event(
                        "CNN+LSTM打哈欠检测器初始化成功",
                        LogLevel.INFO,
                        LogCategory.SYSTEM_EVENT,
                        {"model_path": model_path}
                    )
                else:
                    print("⚠️ Enhanced版本: CNN+LSTM打哈欠检测器加载失败")
                    self.yawn_detector = None
            else:
                print("⚠️ Enhanced版本: 未找到CNN+LSTM打哈欠检测模型")
                self.yawn_detector = None

        except Exception as e:
            print(f"Enhanced版本: CNN+LSTM打哈欠检测器初始化失败: {e}")
            self.yawn_detector = None
            # 记录错误到系统日志
            self.logger.log_system_event(
                f"CNN+LSTM打哈欠检测器初始化失败: {e}",
                LogLevel.ERROR,
                LogCategory.SYSTEM_EVENT
            )

    def get_cnn_lstm_status(self):
        """获取CNN+LSTM检测器状态信息"""
        status_info = {
            "pytorch_available": PYTORCH_AVAILABLE,
            "detector_available": self.yawn_detector is not None and self.yawn_detector.is_available,
            "model_loaded": False,
            "device": "未知",
            "seq_length": 0,
            "consecutive_frames": 0
        }

        if self.yawn_detector and self.yawn_detector.is_available:
            status_info["model_loaded"] = True
            status_info["device"] = str(self.yawn_detector.device) if hasattr(self.yawn_detector, 'device') else "未知"
            status_info["seq_length"] = getattr(self.yawn_detector, 'seq_length', 0)
            status_info["consecutive_frames"] = getattr(self.yawn_detector, 'consecutive_frames', 0)

        return status_info

    def show_cnn_lstm_status(self):
        """显示CNN+LSTM检测器状态对话框"""
        status = self.get_cnn_lstm_status()

        status_text = "CNN+LSTM打哈欠检测器状态:\n\n"
        status_text += f"PyTorch: {'✅ 可用' if status['pytorch_available'] else '❌ 不可用'}\n"
        status_text += f"检测器: {'✅ 已加载' if status['detector_available'] else '❌ 未加载'}\n"
        status_text += f"模型: {'✅ 已加载' if status['model_loaded'] else '❌ 未加载'}\n"

        if status['detector_available']:
            status_text += f"设备: {status['device']}\n"
            status_text += f"序列长度: {status['seq_length']} 帧\n"
            status_text += f"连续帧阈值: {status['consecutive_frames']} 帧\n"

        status_text += "\n模型功能:\n"
        status_text += "• 专门用于打哈欠检测\n"
        status_text += "• 基于CNN+LSTM深度学习\n"
        status_text += "• 需要连续帧验证\n"
        status_text += "• 具有冷却机制防止重复计数\n"

        QMessageBox.information(self, "CNN+LSTM状态", status_text)
    
    def show_login_dialog(self):
        """显示登录对话框"""
        login_dialog = LoginDialog(self.user_manager, self.logger)
        login_dialog.login_successful.connect(self.on_login_successful)
        
        if login_dialog.exec() == QDialog.Accepted:
            pass  # 登录成功的处理在信号槽中
        else:
            # 如果取消登录且没有当前用户，则退出程序
            if not self.current_user:
                self.close()
    
    def on_login_successful(self, username: str):
        """登录成功处理"""
        self.current_user = self.user_manager.current_user
        self.session_token = self.user_manager.session_token
        
        # 更新界面
        self.update_ui_for_user()
        
        # 显示主界面
        self.central_widget.setVisible(True)
        
        # 发出登录信号
        self.user_logged_in.emit(username)
        
        # 显示欢迎消息
        QMessageBox.information(
            self, "登录成功", 
            f"欢迎，{self.current_user.full_name or username}！\n"
            f"您的角色是：{self.current_user.role.value}"
        )
    
    def logout(self):
        """用户登出"""
        if self.current_user:
            # 记录登出日志
            self.logger.log_user_action(
                action="logout",
                description="用户登出",
                user_id=self.current_user.user_id,
                username=self.current_user.username,
                session_id=self.session_token
            )

            # 停止疲劳检测（如果正在运行）
            self.stop_all_detection_threads()

            # 执行登出
            self.user_manager.logout()
            self.current_user = None
            self.session_token = None

            # 更新界面
            self.update_ui_for_logout()

            # 隐藏主界面
            self.central_widget.setVisible(False)

            # 发出登出信号
            self.user_logged_out.emit()

            # 显示登录对话框
            self.show_login_dialog()

    def stop_all_detection_threads(self):
        """停止所有检测相关的线程"""
        try:
            # 停止疲劳检测线程
            if self.fatigue_ui:
                if hasattr(self.fatigue_ui, 'is_running'):
                    self.fatigue_ui.is_running = False

                # 停止检测
                if hasattr(self.fatigue_ui, 'stop_detection'):
                    self.fatigue_ui.stop_detection()

                # 等待线程结束
                if hasattr(self.fatigue_ui, 'thread') and self.fatigue_ui.thread:
                    if hasattr(self.fatigue_ui.thread, 'is_alive') and self.fatigue_ui.thread.is_alive():
                        try:
                            self.fatigue_ui.thread.join(timeout=2)
                            if self.fatigue_ui.thread.is_alive():
                                print("⚠️ 疲劳检测线程未能在2秒内停止")
                            else:
                                print("✅ 疲劳检测线程已停止")
                        except Exception as e:
                            print(f"停止疲劳检测线程时出错: {e}")

            # 停止统计更新线程
            for i in range(self.tab_widget.count()):
                widget = self.tab_widget.widget(i)
                if hasattr(widget, 'update_thread'):
                    try:
                        if widget.update_thread.isRunning():
                            widget.update_thread.stop()
                            print("✅ 统计更新线程已停止")
                    except Exception as e:
                        print(f"停止统计更新线程时出错: {e}")

        except Exception as e:
            print(f"停止检测线程时出错: {e}")
    
    def update_ui_for_user(self):
        """根据用户权限更新界面"""
        if not self.current_user:
            return
        
        # 更新状态栏
        user_info = f"{self.current_user.username} ({self.current_user.role.value})"
        self.user_label.setText(user_info)
        
        # 更新菜单状态
        self.login_action.setEnabled(False)
        self.logout_action.setEnabled(True)
        
        # 根据权限启用/禁用菜单项
        self.user_mgmt_action.setEnabled(
            self.user_manager.has_permission(Permission.USER_MANAGE)
        )
        self.log_viewer_action.setEnabled(
            self.user_manager.has_permission(Permission.LOG_MANAGE)
        )
        
        # 创建选项卡
        self.create_tabs()
    
    def update_ui_for_logout(self):
        """登出后更新界面"""
        # 更新状态栏
        self.user_label.setText("未登录")
        
        # 更新菜单状态
        self.login_action.setEnabled(True)
        self.logout_action.setEnabled(False)
        self.user_mgmt_action.setEnabled(False)
        self.log_viewer_action.setEnabled(False)
        
        # 清空选项卡
        self.tab_widget.clear()
    
    def create_tabs(self):
        """创建功能选项卡"""
        self.tab_widget.clear()
        
        # 疲劳检测选项卡（所有用户都可以访问）
        if self.user_manager.has_permission(Permission.DETECTION_START):
            self.create_fatigue_detection_tab()
        
        # 数据统计选项卡
        if self.user_manager.has_permission(Permission.DATA_VIEW):
            self.create_statistics_tab()
        
        # 用户管理选项卡（仅管理员）
        if self.user_manager.has_permission(Permission.USER_MANAGE):
            self.create_user_management_tab()
        
        # 系统日志选项卡（管理员和监控人员）
        if self.user_manager.has_permission(Permission.LOG_MANAGE):
            self.create_log_viewer_tab()
    
    def create_fatigue_detection_tab(self):
        """创建疲劳检测选项卡"""
        if MAIN_UI_AVAILABLE:
            try:
                # 创建原有的疲劳检测界面
                self.fatigue_ui = MainUI()

                # 将Enhanced版本的CNN+LSTM检测器传递给MainUI
                if self.yawn_detector and self.yawn_detector.is_available:
                    self.fatigue_ui.yawn_detector = self.yawn_detector
                    print("✅ Enhanced版本: CNN+LSTM检测器已传递给MainUI")
                    # 记录到系统日志
                    self.logger.log_system_event(
                        "CNN+LSTM检测器已集成到疲劳检测界面",
                        LogLevel.INFO,
                        LogCategory.SYSTEM_EVENT
                    )
                else:
                    print("⚠️ Enhanced版本: CNN+LSTM检测器不可用")

                # 连接疲劳检测和统计模块
                self.connect_detection_to_stats()

                # 包装在一个容器中
                container = QWidget()
                layout = QVBoxLayout()

                # 添加用户信息显示
                info_label = QLabel(f"当前用户: {self.current_user.username} | 检测权限: 已授权")
                info_label.setStyleSheet("background-color: #e8f5e8; padding: 5px; border-radius: 3px;")
                layout.addWidget(info_label)

                # 添加CNN+LSTM检测器状态显示
                cnn_lstm_status = "✅ 可用" if (self.yawn_detector and self.yawn_detector.is_available) else "❌ 不可用"
                pytorch_status = "✅ 已安装" if PYTORCH_AVAILABLE else "❌ 未安装"
                status_label = QLabel(f"CNN+LSTM打哈欠检测: {cnn_lstm_status} | PyTorch: {pytorch_status}")
                status_color = "#e8f5e8" if (self.yawn_detector and self.yawn_detector.is_available) else "#ffe8e8"
                status_label.setStyleSheet(f"background-color: {status_color}; padding: 5px; border-radius: 3px;")
                layout.addWidget(status_label)

                # 添加原有界面
                layout.addWidget(self.fatigue_ui)
                container.setLayout(layout)

                self.tab_widget.addTab(container, "🎯 疲劳检测")

                # 记录检测模块加载日志
                self.logger.log_user_action(
                    action="load_detection_module",
                    description="加载疲劳检测模块",
                    user_id=self.current_user.user_id,
                    username=self.current_user.username
                )

            except Exception as e:
                error_widget = QLabel(f"疲劳检测模块加载失败: {e}")
                error_widget.setAlignment(Qt.AlignCenter)
                self.tab_widget.addTab(error_widget, "❌ 疲劳检测")

                # 记录错误日志
                self.logger.log_error(
                    action="load_detection_module_failed",
                    description="疲劳检测模块加载失败",
                    error_details=str(e),
                    user_id=self.current_user.user_id,
                    username=self.current_user.username
                )
        else:
            placeholder = QLabel("疲劳检测模块不可用\n请检查相关依赖是否正确安装")
            placeholder.setAlignment(Qt.AlignCenter)
            self.tab_widget.addTab(placeholder, "❌ 疲劳检测")
    
    def create_statistics_tab(self):
        """创建统计分析选项卡"""
        try:
            # 创建疲劳统计界面
            from statistics_widget import FatigueStatisticsWidget
            stats_widget = FatigueStatisticsWidget(self.fatigue_stats, self.user_manager, self.logger)
            self.tab_widget.addTab(stats_widget, "📊 数据统计")

            # 记录统计模块加载日志
            self.logger.log_user_action(
                action="load_statistics_module",
                description="加载数据统计模块",
                user_id=self.current_user.user_id,
                username=self.current_user.username
            )

        except Exception as e:
            error_widget = QLabel(f"统计模块加载失败: {e}")
            error_widget.setAlignment(Qt.AlignCenter)
            self.tab_widget.addTab(error_widget, "❌ 数据统计")
    
    def create_user_management_tab(self):
        """创建用户管理选项卡"""
        try:
            user_mgmt_widget = UserManagementWidget(self.user_manager, self.logger)
            self.tab_widget.addTab(user_mgmt_widget, "👥 用户管理")
            
        except Exception as e:
            error_widget = QLabel(f"用户管理模块加载失败: {e}")
            error_widget.setAlignment(Qt.AlignCenter)
            self.tab_widget.addTab(error_widget, "❌ 用户管理")
    
    def create_log_viewer_tab(self):
        """创建日志查看选项卡"""
        if not LOG_VIEWER_AVAILABLE:
            error_widget = QLabel("日志查看器不可用\n可能是matplotlib兼容性问题\n\n基本功能仍然可用")
            error_widget.setAlignment(Qt.AlignCenter)
            error_widget.setStyleSheet("color: #666; font-size: 14px;")
            self.tab_widget.addTab(error_widget, "❌ 系统日志")
            return

        try:
            log_viewer_widget = LogViewerWidget(self.user_manager, self.logger)
            self.tab_widget.addTab(log_viewer_widget, "📋 系统日志")

        except Exception as e:
            error_widget = QLabel(f"日志查看模块加载失败: {e}")
            error_widget.setAlignment(Qt.AlignCenter)
            self.tab_widget.addTab(error_widget, "❌ 系统日志")
    
    def show_user_management(self):
        """显示用户管理（菜单项）"""
        if self.user_manager.has_permission(Permission.USER_MANAGE):
            # 切换到用户管理选项卡
            for i in range(self.tab_widget.count()):
                if "用户管理" in self.tab_widget.tabText(i):
                    self.tab_widget.setCurrentIndex(i)
                    break
        else:
            QMessageBox.warning(self, "权限不足", "您没有用户管理权限")
    
    def show_log_viewer(self):
        """显示日志查看器（菜单项）"""
        if not LOG_VIEWER_AVAILABLE:
            QMessageBox.warning(self, "功能不可用", "日志查看器不可用，可能是matplotlib兼容性问题")
            return

        if self.user_manager.has_permission(Permission.LOG_MANAGE):
            # 切换到日志查看选项卡
            for i in range(self.tab_widget.count()):
                if "系统日志" in self.tab_widget.tabText(i):
                    self.tab_widget.setCurrentIndex(i)
                    break
        else:
            QMessageBox.warning(self, "权限不足", "您没有日志查看权限")
    
    def show_about(self):
        """显示关于对话框"""
        about_text = """
        🚗 智能疲劳驾驶检测系统 - 企业版
        
        版本: 2.0.0
        
        功能特性:
        • 实时疲劳检测
        • 用户权限管理
        • 系统日志记录
        • 数据统计分析
        
        支持的用户角色:
        • 管理员: 完全权限
        • 监控人员: 查看和分析权限
        • 驾驶员: 基本检测功能
        • 访客: 只读权限
        
        技术支持: support@fatigue-system.com
        """
        
        QMessageBox.about(self, "关于系统", about_text)
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        print("正在关闭系统...")

        # 记录系统关闭日志
        if self.current_user:
            self.logger.log_user_action(
                action="system_shutdown",
                description="用户关闭系统",
                user_id=self.current_user.user_id,
                username=self.current_user.username
            )

        self.logger.log_system_event(
            action="system_shutdown",
            description="疲劳检测系统关闭",
            level=LogLevel.INFO
        )

        # 停止所有检测线程
        self.stop_all_detection_threads()

        # 停止时间更新定时器
        if hasattr(self, 'time_timer'):
            self.time_timer.stop()

        # 登出用户
        if self.current_user:
            self.user_manager.logout()

        print("✅ 系统已安全关闭")
        event.accept()


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序信息
    app.setApplicationName("智能疲劳驾驶检测系统")
    app.setApplicationVersion("2.0.0")
    app.setOrganizationName("疲劳检测系统开发团队")
    
    # 创建主窗口
    window = EnhancedFatigueDetectionSystem()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
