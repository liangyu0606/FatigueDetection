#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志查看和分析界面模块
提供日志筛选、搜索、导出等功能
"""

import csv
import datetime
from typing import List, Dict, Any
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QComboBox, QDateTimeEdit, QSpinBox,
    QGroupBox, QGridLayout, QHeaderView, QMessageBox, QFileDialog,
    QTextEdit, QSplitter, QTabWidget, QProgressBar, QCheckBox, QInputDialog
)
from PySide6.QtCore import Qt, QDateTime, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QColor

from system_logger import SystemLogger, LogLevel, LogCategory, LogEntry
from user_management import UserManager, Permission

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class LogLoadThread(QThread):
    """日志加载线程"""
    
    logs_loaded = Signal(list)
    progress_updated = Signal(int)
    
    def __init__(self, logger: SystemLogger, filters: Dict[str, Any]):
        super().__init__()
        self.logger = logger
        self.filters = filters
    
    def run(self):
        """执行日志加载"""
        try:
            logs = self.logger.get_logs(**self.filters)
            self.logs_loaded.emit(logs)
        except Exception as e:
            print(f"日志加载失败: {e}")
            self.logs_loaded.emit([])


class LogViewerWidget(QWidget):
    """日志查看器组件"""
    
    def __init__(self, user_manager: UserManager, logger: SystemLogger):
        super().__init__()
        self.user_manager = user_manager
        self.logger = logger
        self.current_logs = []
        self.setup_ui()
        
        # 自动刷新定时器
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.auto_refresh)
        
    def setup_ui(self):
        """设置界面"""
        layout = QVBoxLayout()
        
        # 创建选项卡
        self.tab_widget = QTabWidget()
        
        # 日志查看选项卡
        self.log_view_tab = self.create_log_view_tab()
        self.tab_widget.addTab(self.log_view_tab, "日志查看")
        
        # 统计分析选项卡
        if MATPLOTLIB_AVAILABLE:
            self.stats_tab = self.create_stats_tab()
            self.tab_widget.addTab(self.stats_tab, "统计分析")
        
        layout.addWidget(self.tab_widget)
        self.setLayout(layout)
        
        # 初始加载
        self.load_logs()
    
    def create_log_view_tab(self) -> QWidget:
        """创建日志查看选项卡"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 筛选器组
        filter_group = QGroupBox("筛选条件")
        filter_layout = QGridLayout()
        
        # 时间范围
        filter_layout.addWidget(QLabel("开始时间:"), 0, 0)
        self.start_time_edit = QDateTimeEdit()
        self.start_time_edit.setDateTime(QDateTime.currentDateTime().addDays(-7))
        self.start_time_edit.setCalendarPopup(True)
        filter_layout.addWidget(self.start_time_edit, 0, 1)
        
        filter_layout.addWidget(QLabel("结束时间:"), 0, 2)
        self.end_time_edit = QDateTimeEdit()
        self.end_time_edit.setDateTime(QDateTime.currentDateTime())
        self.end_time_edit.setCalendarPopup(True)
        filter_layout.addWidget(self.end_time_edit, 0, 3)
        
        # 日志级别
        filter_layout.addWidget(QLabel("日志级别:"), 1, 0)
        self.level_combo = QComboBox()
        self.level_combo.addItem("全部", None)
        for level in LogLevel:
            self.level_combo.addItem(level.value, level)
        filter_layout.addWidget(self.level_combo, 1, 1)
        
        # 日志分类
        filter_layout.addWidget(QLabel("日志分类:"), 1, 2)
        self.category_combo = QComboBox()
        self.category_combo.addItem("全部", None)
        for category in LogCategory:
            self.category_combo.addItem(category.value, category)
        filter_layout.addWidget(self.category_combo, 1, 3)
        
        # 用户筛选
        filter_layout.addWidget(QLabel("用户:"), 2, 0)
        self.user_combo = QComboBox()
        self.user_combo.addItem("全部用户", None)
        self.load_users_for_filter()
        filter_layout.addWidget(self.user_combo, 2, 1)
        
        # 搜索关键词
        filter_layout.addWidget(QLabel("关键词:"), 2, 2)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索操作或描述...")
        filter_layout.addWidget(self.search_edit, 2, 3)
        
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)
        
        # 操作按钮
        button_layout = QHBoxLayout()
        
        self.search_button = QPushButton("搜索")
        self.search_button.clicked.connect(self.load_logs)
        
        self.refresh_button = QPushButton("刷新")
        self.refresh_button.clicked.connect(self.load_logs)
        
        self.export_button = QPushButton("导出")
        self.export_button.clicked.connect(self.export_logs)
        
        self.clear_button = QPushButton("清理旧日志")
        self.clear_button.clicked.connect(self.clear_old_logs)
        
        # 自动刷新
        self.auto_refresh_checkbox = QCheckBox("自动刷新")
        self.auto_refresh_checkbox.toggled.connect(self.toggle_auto_refresh)
        
        self.refresh_interval_spin = QSpinBox()
        self.refresh_interval_spin.setRange(5, 300)
        self.refresh_interval_spin.setValue(30)
        self.refresh_interval_spin.setSuffix(" 秒")
        
        button_layout.addWidget(self.search_button)
        button_layout.addWidget(self.refresh_button)
        button_layout.addWidget(self.export_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()
        button_layout.addWidget(self.auto_refresh_checkbox)
        button_layout.addWidget(self.refresh_interval_spin)
        
        layout.addLayout(button_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 日志表格
        self.log_table = QTableWidget()
        self.log_table.setColumnCount(7)
        self.log_table.setHorizontalHeaderLabels([
            "时间", "级别", "分类", "用户", "操作", "描述", "详情"
        ])
        
        # 设置表格属性
        header = self.log_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # 时间
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # 级别
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # 分类
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # 用户
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # 操作
        header.setSectionResizeMode(5, QHeaderView.Stretch)           # 描述
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)  # 详情
        
        self.log_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.log_table.setAlternatingRowColors(True)
        self.log_table.itemSelectionChanged.connect(self.show_log_details)
        
        layout.addWidget(self.log_table)
        
        # 详情显示
        self.details_text = QTextEdit()
        self.details_text.setMaximumHeight(150)
        self.details_text.setReadOnly(True)
        layout.addWidget(self.details_text)
        
        widget.setLayout(layout)
        return widget
    
    def create_stats_tab(self) -> QWidget:
        """创建统计分析选项卡"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 统计控制面板
        control_group = QGroupBox("统计设置")
        control_layout = QHBoxLayout()
        
        control_layout.addWidget(QLabel("统计时间范围:"))
        self.stats_start_time = QDateTimeEdit()
        self.stats_start_time.setDateTime(QDateTime.currentDateTime().addDays(-30))
        self.stats_start_time.setCalendarPopup(True)
        control_layout.addWidget(self.stats_start_time)
        
        control_layout.addWidget(QLabel("至"))
        self.stats_end_time = QDateTimeEdit()
        self.stats_end_time.setDateTime(QDateTime.currentDateTime())
        self.stats_end_time.setCalendarPopup(True)
        control_layout.addWidget(self.stats_end_time)
        
        self.generate_stats_button = QPushButton("生成统计")
        self.generate_stats_button.clicked.connect(self.generate_statistics)
        control_layout.addWidget(self.generate_stats_button)
        
        control_layout.addStretch()
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # 统计图表
        if MATPLOTLIB_AVAILABLE:
            self.stats_figure = Figure(figsize=(12, 8))
            self.stats_canvas = FigureCanvas(self.stats_figure)
            layout.addWidget(self.stats_canvas)
        
        widget.setLayout(layout)
        return widget
    
    def load_users_for_filter(self):
        """加载用户列表用于筛选"""
        try:
            users = self.user_manager.get_all_users()
            for user in users:
                self.user_combo.addItem(user.username, user.user_id)
        except Exception as e:
            print(f"加载用户列表失败: {e}")
    
    def load_logs(self):
        """加载日志"""
        if not self.user_manager.has_permission(Permission.LOG_MANAGE):
            QMessageBox.warning(self, "权限不足", "您没有日志查看权限")
            return
        
        # 构建筛选条件
        filters = {
            'start_time': self.start_time_edit.dateTime().toPython(),
            'end_time': self.end_time_edit.dateTime().toPython(),
            'level': self.level_combo.currentData(),
            'category': self.category_combo.currentData(),
            'user_id': self.user_combo.currentData(),
            'limit': 1000
        }
        
        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 不确定进度
        
        # 启动加载线程
        self.load_thread = LogLoadThread(self.logger, filters)
        self.load_thread.logs_loaded.connect(self.on_logs_loaded)
        self.load_thread.start()
    
    def on_logs_loaded(self, logs: List[LogEntry]):
        """日志加载完成"""
        self.progress_bar.setVisible(False)
        self.current_logs = logs
        
        # 应用关键词筛选
        search_keyword = self.search_edit.text().strip().lower()
        if search_keyword:
            filtered_logs = []
            for log in logs:
                if (search_keyword in log.action.lower() or 
                    search_keyword in log.description.lower() or
                    (log.username and search_keyword in log.username.lower())):
                    filtered_logs.append(log)
            logs = filtered_logs
        
        # 更新表格
        self.update_log_table(logs)
    
    def update_log_table(self, logs: List[LogEntry]):
        """更新日志表格"""
        self.log_table.setRowCount(len(logs))
        
        for row, log in enumerate(logs):
            # 时间
            time_item = QTableWidgetItem(log.timestamp.strftime("%Y-%m-%d %H:%M:%S"))
            self.log_table.setItem(row, 0, time_item)
            
            # 级别
            level_item = QTableWidgetItem(log.level.value)
            if log.level == LogLevel.ERROR or log.level == LogLevel.CRITICAL:
                level_item.setBackground(QColor(255, 200, 200))
            elif log.level == LogLevel.WARNING:
                level_item.setBackground(QColor(255, 255, 200))
            self.log_table.setItem(row, 1, level_item)
            
            # 分类
            self.log_table.setItem(row, 2, QTableWidgetItem(log.category.value))
            
            # 用户
            self.log_table.setItem(row, 3, QTableWidgetItem(log.username or "系统"))
            
            # 操作
            self.log_table.setItem(row, 4, QTableWidgetItem(log.action))
            
            # 描述
            self.log_table.setItem(row, 5, QTableWidgetItem(log.description))
            
            # 详情
            details_text = "有详情" if log.details else "无"
            self.log_table.setItem(row, 6, QTableWidgetItem(details_text))
            
            # 存储完整日志对象
            self.log_table.item(row, 0).setData(Qt.UserRole, log)
    
    def show_log_details(self):
        """显示日志详情"""
        current_row = self.log_table.currentRow()
        if current_row >= 0:
            log = self.log_table.item(current_row, 0).data(Qt.UserRole)
            if log and log.details:
                details_text = f"详细信息:\n"
                for key, value in log.details.items():
                    details_text += f"{key}: {value}\n"
                
                if log.ip_address:
                    details_text += f"IP地址: {log.ip_address}\n"
                if log.user_agent:
                    details_text += f"用户代理: {log.user_agent}\n"
                if log.session_id:
                    details_text += f"会话ID: {log.session_id}\n"
                
                self.details_text.setText(details_text)
            else:
                self.details_text.setText("无详细信息")
    
    def export_logs(self):
        """导出日志"""
        if not self.current_logs:
            QMessageBox.warning(self, "警告", "没有可导出的日志")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出日志", 
            f"logs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV文件 (*.csv)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    # 写入标题行
                    writer.writerow([
                        "时间", "级别", "分类", "用户", "操作", "描述", 
                        "IP地址", "会话ID", "详情"
                    ])
                    
                    # 写入数据行
                    for log in self.current_logs:
                        details_str = str(log.details) if log.details else ""
                        writer.writerow([
                            log.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                            log.level.value,
                            log.category.value,
                            log.username or "系统",
                            log.action,
                            log.description,
                            log.ip_address,
                            log.session_id,
                            details_str
                        ])
                
                # 记录导出日志
                self.logger.log_user_action(
                    action="export_logs",
                    description=f"导出日志到文件: {file_path}",
                    user_id=self.user_manager.current_user.user_id,
                    username=self.user_manager.current_user.username,
                    details={"file_path": file_path, "log_count": len(self.current_logs)}
                )
                
                QMessageBox.information(self, "成功", f"日志已导出到: {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败: {e}")
    
    def clear_old_logs(self):
        """清理旧日志"""
        if not self.user_manager.has_permission(Permission.LOG_MANAGE):
            QMessageBox.warning(self, "权限不足", "您没有日志管理权限")
            return
        
        days, ok = QInputDialog.getInt(
            self, "清理旧日志", "保留最近多少天的日志:", 
            90, 1, 365, 1
        )
        
        if ok:
            reply = QMessageBox.question(
                self, "确认清理", 
                f"确定要删除 {days} 天前的日志吗？\n此操作不可撤销。",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                deleted_count = self.logger.clean_old_logs(days)
                
                # 记录清理日志
                self.logger.log_user_action(
                    action="clean_logs",
                    description=f"清理了 {deleted_count} 条旧日志",
                    user_id=self.user_manager.current_user.user_id,
                    username=self.user_manager.current_user.username,
                    details={"days_kept": days, "deleted_count": deleted_count}
                )
                
                QMessageBox.information(self, "完成", f"已清理 {deleted_count} 条旧日志")
                self.load_logs()  # 重新加载
    
    def toggle_auto_refresh(self, enabled: bool):
        """切换自动刷新"""
        if enabled:
            interval = self.refresh_interval_spin.value() * 1000  # 转换为毫秒
            self.refresh_timer.start(interval)
        else:
            self.refresh_timer.stop()
    
    def auto_refresh(self):
        """自动刷新"""
        self.load_logs()
    
    def generate_statistics(self):
        """生成统计图表"""
        if not MATPLOTLIB_AVAILABLE:
            QMessageBox.warning(self, "功能不可用", "需要安装matplotlib库")
            return
        
        start_time = self.stats_start_time.dateTime().toPython()
        end_time = self.stats_end_time.dateTime().toPython()
        
        try:
            stats = self.logger.get_log_statistics(start_time, end_time)
            self.plot_statistics(stats)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"生成统计失败: {e}")
    
    def plot_statistics(self, stats: Dict[str, Any]):
        """绘制统计图表"""
        self.stats_figure.clear()
        
        # 创建子图
        axes = self.stats_figure.subplots(2, 2, figsize=(12, 8))
        
        # 日志级别分布饼图
        if stats.get('level_statistics'):
            levels = list(stats['level_statistics'].keys())
            counts = list(stats['level_statistics'].values())
            axes[0, 0].pie(counts, labels=levels, autopct='%1.1f%%')
            axes[0, 0].set_title('日志级别分布')
        
        # 日志分类分布柱状图
        if stats.get('category_statistics'):
            categories = list(stats['category_statistics'].keys())
            counts = list(stats['category_statistics'].values())
            axes[0, 1].bar(categories, counts)
            axes[0, 1].set_title('日志分类分布')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 用户活动统计
        if stats.get('user_statistics'):
            users = list(stats['user_statistics'].keys())[:10]  # 前10个用户
            counts = list(stats['user_statistics'].values())[:10]
            axes[1, 0].barh(users, counts)
            axes[1, 0].set_title('用户活动统计 (前10)')
        
        # 总体统计信息
        total_count = stats.get('total_count', 0)
        axes[1, 1].text(0.1, 0.8, f"总日志数: {total_count}", fontsize=14)
        axes[1, 1].text(0.1, 0.6, f"时间范围:", fontsize=12)
        axes[1, 1].text(0.1, 0.5, f"  开始: {stats['time_range']['start_time']}", fontsize=10)
        axes[1, 1].text(0.1, 0.4, f"  结束: {stats['time_range']['end_time']}", fontsize=10)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('统计摘要')
        
        self.stats_figure.tight_layout()
        self.stats_canvas.draw()
