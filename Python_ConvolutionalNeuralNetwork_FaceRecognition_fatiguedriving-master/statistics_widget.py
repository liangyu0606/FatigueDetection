#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
疲劳检测统计界面组件
提供数据统计、可视化图表和报告生成功能
"""

import sys
import datetime
import json
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QTabWidget, QGroupBox, QLabel, QPushButton,
    QDateTimeEdit, QComboBox, QTextEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, QProgressBar,
    QMessageBox, QFileDialog, QSplitter, QFrame,
    QScrollArea, QSpinBox
)
from PySide6.QtCore import Qt, QDateTime, QTimer, Signal, QThread
from PySide6.QtGui import QFont, QColor, QPalette

from fatigue_statistics import FatigueStatistics, FatigueEvent
from user_management import UserManager, Permission
from system_logger import SystemLogger, LogLevel

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    # 修复matplotlib后端导入
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    except ImportError:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import numpy as np

    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class StatisticsUpdateThread(QThread):
    """统计数据更新线程"""
    
    data_updated = Signal(dict)
    
    def __init__(self, fatigue_stats: FatigueStatistics):
        super().__init__()
        self.fatigue_stats = fatigue_stats
        self.running = False
        
    def run(self):
        """运行统计更新"""
        self.running = True
        while self.running:
            try:
                # 获取最新统计数据
                summary = self.fatigue_stats.get_session_summary()
                trends = self.fatigue_stats.get_recent_trends(30)  # 最近30分钟

                data = {
                    'summary': summary,
                    'trends': trends,
                    'timestamp': datetime.datetime.now()
                }

                self.data_updated.emit(data)
                self.msleep(2000)  # 每2秒更新一次，提高响应性

            except Exception as e:
                print(f"统计更新线程错误: {e}")
                import traceback
                traceback.print_exc()
                self.msleep(5000)  # 出错时等待5秒
    
    def stop(self):
        """停止更新"""
        print("正在停止统计更新线程...")
        self.running = False

        # 等待线程自然结束
        if self.isRunning():
            self.quit()
            if not self.wait(3000):  # 等待3秒
                print("⚠️ 统计更新线程未能在3秒内停止，强制终止")
                self.terminate()
                self.wait(1000)  # 再等待1秒确保终止
            else:
                print("✅ 统计更新线程已正常停止")


class FatigueStatisticsWidget(QWidget):
    """疲劳检测统计界面组件"""
    
    def __init__(self, fatigue_stats: FatigueStatistics, user_manager: UserManager, logger: SystemLogger):
        super().__init__()
        self.fatigue_stats = fatigue_stats
        self.user_manager = user_manager
        self.logger = logger
        
        # 数据更新线程
        self.update_thread = StatisticsUpdateThread(fatigue_stats)
        self.update_thread.data_updated.connect(self.update_statistics_display)
        
        # 当前统计数据
        self.current_data = {}
        
        self.setup_ui()
        self.start_auto_update()
        
    def setup_ui(self):
        """设置界面"""
        layout = QVBoxLayout()
        
        # 创建选项卡
        self.tab_widget = QTabWidget()
        
        # 实时统计选项卡
        self.realtime_tab = self.create_realtime_tab()
        self.tab_widget.addTab(self.realtime_tab, "📊 实时统计")
        
        # 历史分析选项卡
        self.history_tab = self.create_history_tab()
        self.tab_widget.addTab(self.history_tab, "📈 历史分析")
        
        # 图表可视化选项卡
        if MATPLOTLIB_AVAILABLE:
            self.charts_tab = self.create_charts_tab()
            self.tab_widget.addTab(self.charts_tab, "📉 图表分析")
        
        # 报告生成选项卡
        self.reports_tab = self.create_reports_tab()
        self.tab_widget.addTab(self.reports_tab, "📋 报告生成")
        
        layout.addWidget(self.tab_widget)
        self.setLayout(layout)
        
    def create_realtime_tab(self):
        """创建实时统计选项卡"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 控制面板
        control_group = QGroupBox("控制面板")
        control_layout = QHBoxLayout()
        
        self.auto_update_btn = QPushButton("开始自动更新")
        self.auto_update_btn.clicked.connect(self.toggle_auto_update)
        control_layout.addWidget(self.auto_update_btn)
        
        self.refresh_btn = QPushButton("手动刷新")
        self.refresh_btn.clicked.connect(self.manual_refresh)
        control_layout.addWidget(self.refresh_btn)
        
        control_layout.addStretch()
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # 创建统计卡片区域
        cards_scroll = QScrollArea()
        cards_widget = QWidget()
        self.cards_layout = QGridLayout()
        
        # 创建统计卡片
        self.create_statistics_cards()
        
        cards_widget.setLayout(self.cards_layout)
        cards_scroll.setWidget(cards_widget)
        cards_scroll.setWidgetResizable(True)
        layout.addWidget(cards_scroll)
        
        widget.setLayout(layout)
        return widget
        
    def create_statistics_cards(self):
        """创建统计卡片"""
        # 会话信息卡片
        self.session_card = self.create_info_card("会话信息", {
            "会话时长": "0 分钟",
            "开始时间": "未开始",
            "当前状态": "待机中"
        })
        self.cards_layout.addWidget(self.session_card, 0, 0)
        
        # 检测统计卡片
        self.detection_card = self.create_info_card("检测统计", {
            "眨眼次数": "0",
            "哈欠次数": "0", 
            "点头次数": "0",
            "疲劳事件": "0"
        })
        self.cards_layout.addWidget(self.detection_card, 0, 1)
        
        # 疲劳分析卡片
        self.fatigue_card = self.create_info_card("疲劳分析", {
            "当前疲劳等级": "正常",
            "疲劳持续时间": "0 秒",
            "最高疲劳等级": "正常",
            "平均注意力": "100%"
        })
        self.cards_layout.addWidget(self.fatigue_card, 1, 0)
        
        # 频率分析卡片
        self.frequency_card = self.create_info_card("频率分析", {
            "眨眼频率": "0 次/分钟",
            "哈欠频率": "0 次/分钟",
            "疲劳频率": "0 次/小时",
            "警告频率": "0 次/小时"
        })
        self.cards_layout.addWidget(self.frequency_card, 1, 1)
        
    def create_info_card(self, title: str, data: Dict[str, str]) -> QGroupBox:
        """创建信息卡片"""
        card = QGroupBox(title)
        card.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 12pt;
                border: 2px solid #cccccc;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #f8f9fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #2c3e50;
            }
        """)
        
        layout = QGridLayout()
        
        for i, (key, value) in enumerate(data.items()):
            # 标签
            label = QLabel(key + ":")
            label.setStyleSheet("font-weight: bold; color: #555;")
            
            # 数值
            value_label = QLabel(value)
            value_label.setObjectName(f"{title}_{key}")
            value_label.setStyleSheet("color: #2980b9; font-size: 11pt;")
            
            row = i // 2
            col = (i % 2) * 2
            layout.addWidget(label, row, col)
            layout.addWidget(value_label, row, col + 1)
        
        card.setLayout(layout)
        return card
        
    def create_history_tab(self):
        """创建历史分析选项卡"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 时间范围选择
        time_group = QGroupBox("时间范围选择")
        time_layout = QHBoxLayout()
        
        time_layout.addWidget(QLabel("开始时间:"))
        self.start_time = QDateTimeEdit()
        self.start_time.setDateTime(QDateTime.currentDateTime().addDays(-1))
        time_layout.addWidget(self.start_time)
        
        time_layout.addWidget(QLabel("结束时间:"))
        self.end_time = QDateTimeEdit()
        self.end_time.setDateTime(QDateTime.currentDateTime())
        time_layout.addWidget(self.end_time)
        
        self.query_btn = QPushButton("查询历史数据")
        self.query_btn.clicked.connect(self.query_history_data)
        time_layout.addWidget(self.query_btn)
        
        time_layout.addStretch()
        time_group.setLayout(time_layout)
        layout.addWidget(time_group)
        
        # 历史数据表格
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(6)
        self.history_table.setHorizontalHeaderLabels([
            "时间", "事件类型", "数值", "置信度", "持续时间", "附加信息"
        ])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.history_table)
        
        widget.setLayout(layout)
        return widget

    def create_charts_tab(self):
        """创建图表分析选项卡"""
        widget = QWidget()
        layout = QVBoxLayout()

        # 图表控制面板
        chart_control_group = QGroupBox("图表控制")
        chart_control_layout = QHBoxLayout()

        chart_control_layout.addWidget(QLabel("图表类型:"))
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems([
            "疲劳水平时间线", "注意力变化趋势", "事件统计柱状图",
            "疲劳等级分布", "频率分析", "综合仪表板"
        ])
        chart_control_layout.addWidget(self.chart_type_combo)

        chart_control_layout.addWidget(QLabel("时间范围:"))
        self.chart_time_range = QComboBox()
        self.chart_time_range.addItems([
            "最近1小时", "最近6小时", "最近24小时", "最近7天", "自定义"
        ])
        chart_control_layout.addWidget(self.chart_time_range)

        self.generate_chart_btn = QPushButton("生成图表")
        self.generate_chart_btn.clicked.connect(self.generate_chart)
        chart_control_layout.addWidget(self.generate_chart_btn)

        self.save_chart_btn = QPushButton("保存图表")
        self.save_chart_btn.clicked.connect(self.save_chart)
        chart_control_layout.addWidget(self.save_chart_btn)

        chart_control_layout.addStretch()
        chart_control_group.setLayout(chart_control_layout)
        layout.addWidget(chart_control_group)

        # 图表显示区域
        if MATPLOTLIB_AVAILABLE:
            self.chart_figure = Figure(figsize=(12, 8))
            self.chart_canvas = FigureCanvas(self.chart_figure)
            layout.addWidget(self.chart_canvas)
        else:
            no_chart_label = QLabel("需要安装matplotlib库才能显示图表")
            no_chart_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(no_chart_label)

        widget.setLayout(layout)
        return widget

    def create_reports_tab(self):
        """创建报告生成选项卡"""
        widget = QWidget()
        layout = QVBoxLayout()

        # 报告配置面板
        report_config_group = QGroupBox("报告配置")
        report_config_layout = QGridLayout()

        report_config_layout.addWidget(QLabel("报告类型:"), 0, 0)
        self.report_type_combo = QComboBox()
        self.report_type_combo.addItems([
            "会话摘要报告", "详细分析报告", "疲劳趋势报告", "自定义报告"
        ])
        report_config_layout.addWidget(self.report_type_combo, 0, 1)

        report_config_layout.addWidget(QLabel("报告格式:"), 0, 2)
        self.report_format_combo = QComboBox()
        self.report_format_combo.addItems(["文本格式", "HTML格式", "JSON格式"])
        report_config_layout.addWidget(self.report_format_combo, 0, 3)

        report_config_layout.addWidget(QLabel("包含图表:"), 1, 0)
        self.include_charts_combo = QComboBox()
        self.include_charts_combo.addItems(["是", "否"])
        report_config_layout.addWidget(self.include_charts_combo, 1, 1)

        self.generate_report_btn = QPushButton("生成报告")
        self.generate_report_btn.clicked.connect(self.generate_report)
        report_config_layout.addWidget(self.generate_report_btn, 1, 2)

        self.save_report_btn = QPushButton("保存报告")
        self.save_report_btn.clicked.connect(self.save_report)
        report_config_layout.addWidget(self.save_report_btn, 1, 3)

        report_config_group.setLayout(report_config_layout)
        layout.addWidget(report_config_group)

        # 报告预览区域
        preview_group = QGroupBox("报告预览")
        preview_layout = QVBoxLayout()

        self.report_preview = QTextEdit()
        self.report_preview.setReadOnly(True)
        self.report_preview.setFont(QFont("Consolas", 10))
        preview_layout.addWidget(self.report_preview)

        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        widget.setLayout(layout)
        return widget

    def start_auto_update(self):
        """开始自动更新"""
        if not self.update_thread.isRunning():
            self.update_thread.start()
            self.auto_update_btn.setText("停止自动更新")

    def toggle_auto_update(self):
        """切换自动更新状态"""
        if self.update_thread.isRunning():
            self.update_thread.stop()
            self.auto_update_btn.setText("开始自动更新")
        else:
            self.start_auto_update()

    def manual_refresh(self):
        """手动刷新数据"""
        try:
            summary = self.fatigue_stats.get_session_summary()
            trends = self.fatigue_stats.get_recent_trends(30)

            data = {
                'summary': summary,
                'trends': trends,
                'timestamp': datetime.datetime.now()
            }

            self.update_statistics_display(data)

        except Exception as e:
            QMessageBox.warning(self, "刷新失败", f"无法刷新统计数据: {e}")

    def update_statistics_display(self, data: Dict[str, Any]):
        """更新统计显示"""
        self.current_data = data
        summary = data.get('summary', {})

        # 更新会话信息卡片
        self.update_card_data("会话信息", {
            "会话时长": f"{summary.get('duration_minutes', 0):.1f} 分钟",
            "开始时间": summary.get('start_time', '未开始'),
            "当前状态": "检测中" if summary.get('duration_minutes', 0) > 0 else "待机中"
        })

        # 更新检测统计卡片
        self.update_card_data("检测统计", {
            "眨眼次数": str(summary.get('blink_count', 0)),
            "哈欠次数": str(summary.get('yawn_count', 0)),
            "点头次数": str(summary.get('nod_count', 0)),
            "疲劳事件": str(summary.get('fatigue_episodes', 0))
        })

        # 更新疲劳分析卡片
        fatigue_levels = ['正常', '轻微疲劳', '中度疲劳', '重度疲劳']
        current_level = fatigue_levels[min(summary.get('current_fatigue_level', 0), 3)]
        max_level = fatigue_levels[min(summary.get('max_fatigue_level', 0), 3)]

        self.update_card_data("疲劳分析", {
            "当前疲劳等级": current_level,
            "疲劳持续时间": f"{summary.get('total_fatigue_duration', 0):.1f} 秒",
            "最高疲劳等级": max_level,
            "平均注意力": f"{summary.get('avg_attention_level', 1.0) * 100:.1f}%"
        })

        # 更新频率分析卡片
        self.update_card_data("频率分析", {
            "眨眼频率": f"{summary.get('blink_rate_per_minute', 0):.1f} 次/分钟",
            "哈欠频率": f"{summary.get('yawn_rate_per_minute', 0):.1f} 次/分钟",
            "疲劳频率": f"{summary.get('fatigue_episodes', 0) * 60 / max(summary.get('duration_minutes', 1), 1):.1f} 次/小时",
            "警告频率": f"{summary.get('warning_count', 0) * 60 / max(summary.get('duration_minutes', 1), 1):.1f} 次/小时"
        })

    def update_card_data(self, card_title: str, data: Dict[str, str]):
        """更新卡片数据"""
        for key, value in data.items():
            label_name = f"{card_title}_{key}"
            label = self.findChild(QLabel, label_name)
            if label:
                label.setText(value)

    def query_history_data(self):
        """查询历史数据"""
        try:
            start_time = self.start_time.dateTime().toPython()
            end_time = self.end_time.dateTime().toPython()

            # 获取历史事件数据
            events = self.fatigue_stats.get_events_in_range(start_time, end_time)

            # 更新表格
            self.history_table.setRowCount(len(events))

            for row, event in enumerate(events):
                self.history_table.setItem(row, 0, QTableWidgetItem(
                    event.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                ))
                self.history_table.setItem(row, 1, QTableWidgetItem(event.event_type))
                self.history_table.setItem(row, 2, QTableWidgetItem(f"{event.value:.2f}"))
                self.history_table.setItem(row, 3, QTableWidgetItem(f"{event.confidence:.2f}"))
                self.history_table.setItem(row, 4, QTableWidgetItem(f"{event.duration:.2f}s"))
                self.history_table.setItem(row, 5, QTableWidgetItem(
                    json.dumps(event.additional_data) if event.additional_data else ""
                ))

            # 记录查询日志
            self.logger.log_user_action(
                action="query_history_data",
                description=f"查询历史数据: {start_time} 到 {end_time}",
                user_id=self.user_manager.current_user.user_id if self.user_manager.current_user else None,
                username=self.user_manager.current_user.username if self.user_manager.current_user else "unknown"
            )

        except Exception as e:
            QMessageBox.critical(self, "查询失败", f"无法查询历史数据: {e}")

    def generate_chart(self):
        """生成图表"""
        if not MATPLOTLIB_AVAILABLE:
            QMessageBox.warning(self, "功能不可用", "需要安装matplotlib库")
            return

        try:
            chart_type = self.chart_type_combo.currentText()
            time_range = self.chart_time_range.currentText()

            # 获取数据
            if time_range == "最近1小时":
                hours = 1
            elif time_range == "最近6小时":
                hours = 6
            elif time_range == "最近24小时":
                hours = 24
            elif time_range == "最近7天":
                hours = 24 * 7
            else:
                hours = 24  # 默认24小时

            trends = self.fatigue_stats.get_recent_trends(hours * 60)  # 转换为分钟

            # 清除之前的图表
            self.chart_figure.clear()

            if chart_type == "疲劳水平时间线":
                self.plot_fatigue_timeline(trends)
            elif chart_type == "注意力变化趋势":
                self.plot_attention_trends(trends)
            elif chart_type == "事件统计柱状图":
                self.plot_event_statistics()
            elif chart_type == "疲劳等级分布":
                self.plot_fatigue_distribution()
            elif chart_type == "频率分析":
                self.plot_frequency_analysis()
            elif chart_type == "综合仪表板":
                self.plot_comprehensive_dashboard()

            self.chart_canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "生成图表失败", f"无法生成图表: {e}")

    def plot_fatigue_timeline(self, trends: Dict[str, Any]):
        """绘制疲劳水平时间线"""
        ax = self.chart_figure.add_subplot(111)

        if 'fatigue_timeline' in trends and trends['fatigue_timeline']:
            times, levels = zip(*trends['fatigue_timeline'])
            ax.plot(times, levels, 'r-', linewidth=2, marker='o', markersize=4)
            ax.set_title('疲劳水平时间线', fontsize=14, fontweight='bold')
            ax.set_ylabel('疲劳等级')
            ax.set_xlabel('时间')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.5, 3.5)
            ax.set_yticks([0, 1, 2, 3])
            ax.set_yticklabels(['正常', '轻微', '中度', '重度'])

            # 格式化时间轴
            if len(times) > 1:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax.text(0.5, 0.5, '暂无疲劳数据', ha='center', va='center', transform=ax.transAxes, fontsize=12)

    def plot_attention_trends(self, trends: Dict[str, Any]):
        """绘制注意力变化趋势"""
        ax = self.chart_figure.add_subplot(111)

        if 'attention_timeline' in trends and trends['attention_timeline']:
            times, levels = zip(*trends['attention_timeline'])
            ax.plot(times, levels, 'b-', linewidth=2, marker='s', markersize=4)
            ax.set_title('注意力水平变化趋势', fontsize=14, fontweight='bold')
            ax.set_ylabel('注意力水平')
            ax.set_xlabel('时间')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.1)

            # 格式化时间轴
            if len(times) > 1:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax.text(0.5, 0.5, '暂无注意力数据', ha='center', va='center', transform=ax.transAxes, fontsize=12)

    def plot_event_statistics(self):
        """绘制事件统计柱状图"""
        ax = self.chart_figure.add_subplot(111)

        summary = self.current_data.get('summary', {})
        events = ['眨眼', '哈欠', '点头', '疲劳事件']
        counts = [
            summary.get('blink_count', 0),
            summary.get('yawn_count', 0),
            summary.get('nod_count', 0),
            summary.get('fatigue_episodes', 0)
        ]

        colors = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6']
        bars = ax.bar(events, counts, color=colors)

        ax.set_title('检测事件统计', fontsize=14, fontweight='bold')
        ax.set_ylabel('次数')

        # 在柱状图上显示数值
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{count}', ha='center', va='bottom')

    def plot_fatigue_distribution(self):
        """绘制疲劳等级分布饼图"""
        ax = self.chart_figure.add_subplot(111)

        trends = self.current_data.get('trends', {})
        if 'fatigue_timeline' in trends and trends['fatigue_timeline']:
            levels = [v for _, v in trends['fatigue_timeline']]
            level_counts = [levels.count(i) for i in range(4)]
            level_labels = ['正常', '轻微疲劳', '中度疲劳', '重度疲劳']
            colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']

            # 只显示非零的等级
            non_zero_indices = [i for i, count in enumerate(level_counts) if count > 0]
            if non_zero_indices:
                filtered_counts = [level_counts[i] for i in non_zero_indices]
                filtered_labels = [level_labels[i] for i in non_zero_indices]
                filtered_colors = [colors[i] for i in non_zero_indices]

                ax.pie(filtered_counts, labels=filtered_labels, colors=filtered_colors,
                      autopct='%1.1f%%', startangle=90)
                ax.set_title('疲劳等级分布', fontsize=14, fontweight='bold')
            else:
                ax.text(0.5, 0.5, '暂无疲劳等级数据', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, '暂无疲劳等级数据', ha='center', va='center', transform=ax.transAxes)

    def plot_frequency_analysis(self):
        """绘制频率分析图"""
        ax = self.chart_figure.add_subplot(111)

        summary = self.current_data.get('summary', {})
        frequencies = ['眨眼频率', '哈欠频率']
        values = [
            summary.get('blink_rate_per_minute', 0),
            summary.get('yawn_rate_per_minute', 0)
        ]

        x_pos = np.arange(len(frequencies))
        bars = ax.bar(x_pos, values, color=['#3498db', '#e74c3c'])

        ax.set_title('频率分析 (次/分钟)', fontsize=14, fontweight='bold')
        ax.set_ylabel('频率 (次/分钟)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(frequencies)

        # 在柱状图上显示数值
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.1f}', ha='center', va='bottom')

    def plot_comprehensive_dashboard(self):
        """绘制综合仪表板"""
        # 创建2x2子图
        axes = self.chart_figure.subplots(2, 2, figsize=(12, 8))

        summary = self.current_data.get('summary', {})
        trends = self.current_data.get('trends', {})

        # 疲劳水平时间线 (左上)
        if 'fatigue_timeline' in trends and trends['fatigue_timeline']:
            times, levels = zip(*trends['fatigue_timeline'][-20:])  # 最近20个点
            axes[0, 0].plot(times, levels, 'r-', linewidth=2)
            axes[0, 0].set_title('疲劳水平趋势')
            axes[0, 0].set_ylabel('疲劳等级')
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, '暂无数据', ha='center', va='center', transform=axes[0, 0].transAxes)

        # 事件统计 (右上)
        events = ['眨眼', '哈欠', '点头']
        counts = [summary.get('blink_count', 0), summary.get('yawn_count', 0), summary.get('nod_count', 0)]
        axes[0, 1].bar(events, counts, color=['#3498db', '#e74c3c', '#f39c12'])
        axes[0, 1].set_title('事件统计')
        axes[0, 1].set_ylabel('次数')

        # 注意力水平 (左下)
        if 'attention_timeline' in trends and trends['attention_timeline']:
            times, levels = zip(*trends['attention_timeline'][-20:])  # 最近20个点
            axes[1, 0].plot(times, levels, 'b-', linewidth=2)
            axes[1, 0].set_title('注意力水平')
            axes[1, 0].set_ylabel('注意力')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, '暂无数据', ha='center', va='center', transform=axes[1, 0].transAxes)

        # 统计摘要 (右下)
        axes[1, 1].axis('off')
        stats_text = f"""
        会话时长: {summary.get('duration_minutes', 0):.1f} 分钟
        疲劳事件: {summary.get('fatigue_episodes', 0)} 次
        平均注意力: {summary.get('avg_attention_level', 1.0) * 100:.1f}%
        最高疲劳等级: {['正常', '轻微', '中度', '重度'][min(summary.get('max_fatigue_level', 0), 3)]}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='center')
        axes[1, 1].set_title('统计摘要')

        self.chart_figure.tight_layout()

    def save_chart(self):
        """保存图表"""
        if not MATPLOTLIB_AVAILABLE:
            QMessageBox.warning(self, "功能不可用", "需要安装matplotlib库")
            return

        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存图表", f"fatigue_chart_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                "PNG文件 (*.png);;JPG文件 (*.jpg);;PDF文件 (*.pdf)"
            )

            if file_path:
                self.chart_figure.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "保存成功", f"图表已保存到: {file_path}")

                # 记录保存日志
                self.logger.log_user_action(
                    action="save_chart",
                    description=f"保存图表到: {file_path}",
                    user_id=self.user_manager.current_user.user_id if self.user_manager.current_user else None,
                    username=self.user_manager.current_user.username if self.user_manager.current_user else "unknown"
                )

        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"无法保存图表: {e}")

    def generate_report(self):
        """生成报告"""
        try:
            report_type = self.report_type_combo.currentText()
            report_format = self.report_format_combo.currentText()
            include_charts = self.include_charts_combo.currentText() == "是"

            if report_type == "会话摘要报告":
                report_content = self.generate_session_summary_report(report_format)
            elif report_type == "详细分析报告":
                report_content = self.generate_detailed_analysis_report(report_format)
            elif report_type == "疲劳趋势报告":
                report_content = self.generate_fatigue_trend_report(report_format)
            else:
                report_content = self.generate_custom_report(report_format)

            self.report_preview.setText(report_content)

        except Exception as e:
            QMessageBox.critical(self, "生成报告失败", f"无法生成报告: {e}")

    def generate_session_summary_report(self, format_type: str) -> str:
        """生成会话摘要报告"""
        summary = self.current_data.get('summary', {})

        if format_type == "HTML格式":
            return f"""
            <html>
            <head><title>疲劳检测会话摘要报告</title></head>
            <body>
            <h1>疲劳检测会话摘要报告</h1>
            <h2>基本信息</h2>
            <ul>
                <li>会话时长: {summary.get('duration_minutes', 0):.1f} 分钟</li>
                <li>开始时间: {summary.get('start_time', '未知')}</li>
                <li>生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</li>
            </ul>
            <h2>检测统计</h2>
            <ul>
                <li>眨眼次数: {summary.get('blink_count', 0)}</li>
                <li>哈欠次数: {summary.get('yawn_count', 0)}</li>
                <li>点头次数: {summary.get('nod_count', 0)}</li>
                <li>疲劳事件: {summary.get('fatigue_episodes', 0)}</li>
            </ul>
            <h2>疲劳分析</h2>
            <ul>
                <li>最高疲劳等级: {['正常', '轻微疲劳', '中度疲劳', '重度疲劳'][min(summary.get('max_fatigue_level', 0), 3)]}</li>
                <li>平均注意力水平: {summary.get('avg_attention_level', 1.0) * 100:.1f}%</li>
                <li>疲劳持续时间: {summary.get('total_fatigue_duration', 0):.1f} 秒</li>
            </ul>
            </body>
            </html>
            """
        elif format_type == "JSON格式":
            return json.dumps(summary, indent=2, ensure_ascii=False, default=str)
        else:  # 文本格式
            return f"""
疲劳检测会话摘要报告
==================

基本信息:
- 会话时长: {summary.get('duration_minutes', 0):.1f} 分钟
- 开始时间: {summary.get('start_time', '未知')}
- 生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

检测统计:
- 眨眼次数: {summary.get('blink_count', 0)}
- 哈欠次数: {summary.get('yawn_count', 0)}
- 点头次数: {summary.get('nod_count', 0)}
- 疲劳事件: {summary.get('fatigue_episodes', 0)}

疲劳分析:
- 最高疲劳等级: {['正常', '轻微疲劳', '中度疲劳', '重度疲劳'][min(summary.get('max_fatigue_level', 0), 3)]}
- 平均注意力水平: {summary.get('avg_attention_level', 1.0) * 100:.1f}%
- 疲劳持续时间: {summary.get('total_fatigue_duration', 0):.1f} 秒

频率分析:
- 眨眼频率: {summary.get('blink_rate_per_minute', 0):.1f} 次/分钟
- 哈欠频率: {summary.get('yawn_rate_per_minute', 0):.1f} 次/分钟
            """

    def generate_detailed_analysis_report(self, format_type: str) -> str:
        """生成详细分析报告"""
        # 这里可以添加更详细的分析逻辑
        return self.fatigue_stats.generate_report()

    def generate_fatigue_trend_report(self, format_type: str) -> str:
        """生成疲劳趋势报告"""
        trends = self.current_data.get('trends', {})

        report = f"""
疲劳趋势分析报告
==============

生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

趋势分析:
"""

        if 'fatigue_timeline' in trends and trends['fatigue_timeline']:
            fatigue_data = trends['fatigue_timeline']
            report += f"- 疲劳数据点数: {len(fatigue_data)}\n"

            levels = [v for _, v in fatigue_data]
            if levels:
                avg_level = sum(levels) / len(levels)
                max_level = max(levels)
                report += f"- 平均疲劳等级: {avg_level:.2f}\n"
                report += f"- 最高疲劳等级: {max_level}\n"

        if 'attention_timeline' in trends and trends['attention_timeline']:
            attention_data = trends['attention_timeline']
            report += f"- 注意力数据点数: {len(attention_data)}\n"

            levels = [v for _, v in attention_data]
            if levels:
                avg_attention = sum(levels) / len(levels)
                min_attention = min(levels)
                report += f"- 平均注意力水平: {avg_attention:.2f}\n"
                report += f"- 最低注意力水平: {min_attention:.2f}\n"

        return report

    def generate_custom_report(self, format_type: str) -> str:
        """生成自定义报告"""
        return "自定义报告功能开发中..."

    def save_report(self):
        """保存报告"""
        try:
            content = self.report_preview.toPlainText()
            if not content.strip():
                QMessageBox.warning(self, "保存失败", "请先生成报告")
                return

            format_type = self.report_format_combo.currentText()
            if format_type == "HTML格式":
                ext = "html"
            elif format_type == "JSON格式":
                ext = "json"
            else:
                ext = "txt"

            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存报告",
                f"fatigue_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}",
                f"{format_type} (*.{ext})"
            )

            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                QMessageBox.information(self, "保存成功", f"报告已保存到: {file_path}")

                # 记录保存日志
                self.logger.log_user_action(
                    action="save_report",
                    description=f"保存报告到: {file_path}",
                    user_id=self.user_manager.current_user.user_id if self.user_manager.current_user else None,
                    username=self.user_manager.current_user.username if self.user_manager.current_user else "unknown"
                )

        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"无法保存报告: {e}")

    def closeEvent(self, event):
        """关闭事件"""
        print("正在关闭统计界面...")
        try:
            if self.update_thread.isRunning():
                self.update_thread.stop()
            print("✅ 统计界面已关闭")
        except Exception as e:
            print(f"关闭统计界面时出错: {e}")
        finally:
            event.accept()

    def __del__(self):
        """析构函数"""
        try:
            if hasattr(self, 'update_thread') and self.update_thread.isRunning():
                self.update_thread.stop()
        except Exception:
            pass
