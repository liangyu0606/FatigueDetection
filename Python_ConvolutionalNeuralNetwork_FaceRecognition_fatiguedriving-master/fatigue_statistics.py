#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
疲劳检测统计数据管理模块
提供疲劳检测数据的记录、分析和可视化功能
"""

import json
import sqlite3
import datetime
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import os

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class FatigueEvent:
    """疲劳事件数据类"""
    timestamp: datetime.datetime
    event_type: str  # 'blink', 'yawn', 'nod', 'fatigue_state'
    value: float
    confidence: float
    duration: float = 0.0
    additional_data: Dict = None


class FatigueStatistics:
    """疲劳检测统计管理器"""

    _instance = None
    _initialized = False

    def __new__(cls, db_path: str = "fatigue_data.db"):
        """单例模式，确保只有一个实例"""
        if cls._instance is None:
            cls._instance = super(FatigueStatistics, cls).__new__(cls)
        return cls._instance

    def __init__(self, db_path: str = "fatigue_data.db"):
        """
        初始化统计管理器

        Args:
            db_path: 数据库文件路径
        """
        # 避免重复初始化
        if self._initialized:
            return

        self.db_path = db_path
        self.session_start_time = datetime.datetime.now()
        self.current_session_id = None

        # 当前会话统计
        self.session_stats = {
            'blink_count': 0,
            'yawn_count': 0,
            'nod_count': 0,
            'fatigue_episodes': 0,
            'total_fatigue_duration': 0.0,
            'max_fatigue_level': 0,
            'avg_attention_level': 0.0
        }

        # 实时数据缓存
        self.recent_events = []
        self.fatigue_timeline = []
        self.attention_timeline = []

        # 初始化数据库
        self._init_database()
        self._start_new_session()

        self._initialized = True
    
    def _init_database(self):
        """初始化数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 创建会话表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    duration_minutes REAL,
                    total_blinks INTEGER,
                    total_yawns INTEGER,
                    total_nods INTEGER,
                    fatigue_episodes INTEGER,
                    max_fatigue_level INTEGER,
                    avg_attention_level REAL,
                    notes TEXT
                )
            ''')
            
            # 创建事件表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    timestamp TIMESTAMP,
                    event_type TEXT,
                    value REAL,
                    confidence REAL,
                    duration REAL,
                    additional_data TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                )
            ''')
            
            # 创建疲劳状态表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fatigue_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    timestamp TIMESTAMP,
                    fatigue_level INTEGER,
                    drowsiness_prob REAL,
                    attention_level REAL,
                    confidence REAL,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            print("数据库初始化成功")
            
        except Exception as e:
            print(f"数据库初始化失败: {e}")
    
    def _start_new_session(self):
        """开始新的检测会话"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO sessions (start_time) VALUES (?)
            ''', (self.session_start_time,))
            
            self.current_session_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            print(f"新会话开始 - ID: {self.current_session_id}")
            
        except Exception as e:
            print(f"创建新会话失败: {e}")
    
    def record_event(self, event: FatigueEvent):
        """
        记录疲劳事件
        
        Args:
            event: 疲劳事件
        """
        try:
            # 更新会话统计
            self._update_session_stats(event)
            
            # 添加到实时缓存
            self.recent_events.append(event)
            if len(self.recent_events) > 1000:  # 保持最近1000个事件
                self.recent_events.pop(0)
            
            # 保存到数据库
            self._save_event_to_db(event)
            
        except Exception as e:
            print(f"记录事件失败: {e}")
    
    def record_fatigue_state(self, fatigue_level: int, drowsiness_prob: float, 
                           attention_level: float, confidence: float):
        """
        记录疲劳状态
        
        Args:
            fatigue_level: 疲劳等级 (0-3)
            drowsiness_prob: 瞌睡概率
            attention_level: 注意力水平
            confidence: 置信度
        """
        try:
            timestamp = datetime.datetime.now()
            
            # 更新时间线数据
            self.fatigue_timeline.append((timestamp, fatigue_level))
            self.attention_timeline.append((timestamp, attention_level))
            
            # 保持最近1小时的数据
            cutoff_time = timestamp - datetime.timedelta(hours=1)
            self.fatigue_timeline = [(t, v) for t, v in self.fatigue_timeline if t > cutoff_time]
            self.attention_timeline = [(t, v) for t, v in self.attention_timeline if t > cutoff_time]
            
            # 保存到数据库
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO fatigue_states 
                (session_id, timestamp, fatigue_level, drowsiness_prob, attention_level, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (self.current_session_id, timestamp, fatigue_level, 
                  drowsiness_prob, attention_level, confidence))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"记录疲劳状态失败: {e}")
    
    def _update_session_stats(self, event: FatigueEvent):
        """更新会话统计"""
        if event.event_type == 'blink':
            self.session_stats['blink_count'] += 1
        elif event.event_type == 'yawn':
            self.session_stats['yawn_count'] += 1
        elif event.event_type == 'nod':
            self.session_stats['nod_count'] += 1
        elif event.event_type == 'fatigue_state':
            if event.value > 1:  # 中度疲劳以上
                self.session_stats['fatigue_episodes'] += 1
            self.session_stats['max_fatigue_level'] = max(
                self.session_stats['max_fatigue_level'], int(event.value)
            )

        # 实时更新数据库中的会话统计
        self._update_session_in_db()

    def _update_session_in_db(self):
        """实时更新数据库中的会话统计"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 计算平均注意力水平
            if self.attention_timeline:
                avg_attention = np.mean([v for _, v in self.attention_timeline])
            else:
                avg_attention = 0.0

            cursor.execute('''
                UPDATE sessions SET
                total_blinks = ?, total_yawns = ?, total_nods = ?,
                fatigue_episodes = ?, max_fatigue_level = ?, avg_attention_level = ?
                WHERE id = ?
            ''', (self.session_stats['blink_count'], self.session_stats['yawn_count'],
                  self.session_stats['nod_count'], self.session_stats['fatigue_episodes'],
                  self.session_stats['max_fatigue_level'], avg_attention,
                  self.current_session_id))

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"实时更新会话统计失败: {e}")

    def _save_event_to_db(self, event: FatigueEvent):
        """保存事件到数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            additional_data_json = json.dumps(event.additional_data) if event.additional_data else None
            
            cursor.execute('''
                INSERT INTO events 
                (session_id, timestamp, event_type, value, confidence, duration, additional_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (self.current_session_id, event.timestamp, event.event_type,
                  event.value, event.confidence, event.duration, additional_data_json))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"保存事件到数据库失败: {e}")
    
    def get_session_summary(self) -> Dict:
        """获取当前会话摘要"""
        current_time = datetime.datetime.now()
        session_duration = (current_time - self.session_start_time).total_seconds() / 60  # 分钟
        
        # 计算平均注意力水平
        if self.attention_timeline:
            avg_attention = np.mean([v for _, v in self.attention_timeline])
        else:
            avg_attention = 0.0
        
        summary = {
            'session_id': self.current_session_id,
            'start_time': self.session_start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'duration_minutes': round(session_duration, 2),
            'blink_count': self.session_stats['blink_count'],
            'yawn_count': self.session_stats['yawn_count'],
            'nod_count': self.session_stats['nod_count'],
            'fatigue_episodes': self.session_stats['fatigue_episodes'],
            'max_fatigue_level': self.session_stats['max_fatigue_level'],
            'avg_attention_level': round(avg_attention, 3),
            'blink_rate_per_minute': round(self.session_stats['blink_count'] / max(session_duration, 1), 2),
            'yawn_rate_per_minute': round(self.session_stats['yawn_count'] / max(session_duration, 1), 2)
        }
        
        return summary
    
    def get_recent_trends(self, minutes: int = 10) -> Dict:
        """
        获取最近趋势数据

        Args:
            minutes: 分析最近多少分钟的数据

        Returns:
            趋势数据字典
        """
        cutoff_time = datetime.datetime.now() - datetime.timedelta(minutes=minutes)

        # 过滤最近的事件
        recent_events = [e for e in self.recent_events if e.timestamp > cutoff_time]

        # 统计最近事件
        recent_blinks = len([e for e in recent_events if e.event_type == 'blink'])
        recent_yawns = len([e for e in recent_events if e.event_type == 'yawn'])
        recent_nods = len([e for e in recent_events if e.event_type == 'nod'])

        # 最近疲劳水平和注意力水平
        recent_fatigue_timeline = [(t, v) for t, v in self.fatigue_timeline if t > cutoff_time]
        recent_attention_timeline = [(t, v) for t, v in self.attention_timeline if t > cutoff_time]

        recent_fatigue = [v for t, v in recent_fatigue_timeline]
        recent_attention = [v for t, v in recent_attention_timeline]

        trends = {
            'time_window_minutes': minutes,
            'recent_blinks': recent_blinks,
            'recent_yawns': recent_yawns,
            'recent_nods': recent_nods,
            'avg_fatigue_level': round(np.mean(recent_fatigue), 2) if recent_fatigue else 0,
            'avg_attention_level': round(np.mean(recent_attention), 2) if recent_attention else 0,
            'fatigue_trend': self._calculate_trend(recent_fatigue),
            'attention_trend': self._calculate_trend(recent_attention),
            # 添加时间线数据供图表使用
            'fatigue_timeline': recent_fatigue_timeline,
            'attention_timeline': recent_attention_timeline
        }

        return trends

    def get_recent_events(self, hours: int = 24) -> List[FatigueEvent]:
        """
        获取最近指定小时内的事件

        Args:
            hours: 小时数

        Returns:
            事件列表
        """
        end_time = datetime.datetime.now()
        start_time = end_time - datetime.timedelta(hours=hours)
        return self.get_events_in_range(start_time, end_time)

    def get_events_in_range(self, start_time: datetime.datetime, end_time: datetime.datetime) -> List[FatigueEvent]:
        """
        获取指定时间范围内的事件

        Args:
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            事件列表
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT timestamp, event_type, value, confidence, duration, additional_data
                FROM events
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp DESC
                LIMIT 1000
            ''', (start_time, end_time))

            events = []
            for row in cursor.fetchall():
                timestamp_str, event_type, value, confidence, duration, additional_data_str = row

                # 解析时间戳
                timestamp = datetime.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00').replace('+00:00', ''))

                # 解析附加数据
                additional_data = None
                if additional_data_str:
                    try:
                        additional_data = json.loads(additional_data_str)
                    except json.JSONDecodeError:
                        pass

                event = FatigueEvent(
                    timestamp=timestamp,
                    event_type=event_type,
                    value=float(value),
                    confidence=float(confidence),
                    duration=float(duration) if duration else 0.0,
                    additional_data=additional_data
                )
                events.append(event)

            conn.close()
            return events

        except Exception as e:
            print(f"获取历史事件失败: {e}")
            return []

    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势方向"""
        if len(values) < 2:
            return 'stable'
        
        # 使用线性回归计算趋势
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        生成详细报告
        
        Args:
            save_path: 报告保存路径
            
        Returns:
            报告内容
        """
        summary = self.get_session_summary()
        trends = self.get_recent_trends(30)  # 最近30分钟
        
        report = f"""
疲劳检测会话报告
================

会话信息:
- 会话ID: {summary['session_id']}
- 开始时间: {summary['start_time']}
- 持续时间: {summary['duration_minutes']} 分钟

检测统计:
- 眨眼次数: {summary['blink_count']} (每分钟 {summary['blink_rate_per_minute']} 次)
- 打哈欠次数: {summary['yawn_count']} (每分钟 {summary['yawn_rate_per_minute']} 次)
- 点头次数: {summary['nod_count']} 次
- 疲劳事件: {summary['fatigue_episodes']} 次
- 最高疲劳等级: {summary['max_fatigue_level']}
- 平均注意力水平: {summary['avg_attention_level']}

最近趋势 (30分钟):
- 疲劳水平趋势: {trends['fatigue_trend']}
- 注意力趋势: {trends['attention_trend']}
- 平均疲劳等级: {trends['avg_fatigue_level']}
- 平均注意力水平: {trends['avg_attention_level']}

建议:
{self._generate_recommendations(summary, trends)}
        """
        
        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"报告已保存到: {save_path}")
            except Exception as e:
                print(f"保存报告失败: {e}")
        
        return report
    
    def _generate_recommendations(self, summary: Dict, trends: Dict) -> str:
        """生成建议"""
        recommendations = []
        
        # 基于疲劳等级的建议
        if summary['max_fatigue_level'] >= 3:
            recommendations.append("- 检测到重度疲劳，建议立即停止驾驶并休息")
        elif summary['max_fatigue_level'] >= 2:
            recommendations.append("- 检测到中度疲劳，建议尽快找安全地点休息")
        
        # 基于哈欠频率的建议
        if summary['yawn_rate_per_minute'] > 0.5:
            recommendations.append("- 哈欠频率较高，表明疲劳程度较重")
        
        # 基于眨眼频率的建议
        if summary['blink_rate_per_minute'] > 25:
            recommendations.append("- 眨眼频率过高，可能存在眼部疲劳")
        
        # 基于注意力水平的建议
        if summary['avg_attention_level'] < 0.6:
            recommendations.append("- 注意力水平较低，建议提高警觉性")
        
        # 基于趋势的建议
        if trends['fatigue_trend'] == 'increasing':
            recommendations.append("- 疲劳水平呈上升趋势，需要密切关注")
        
        if not recommendations:
            recommendations.append("- 整体状态良好，继续保持警觉")
        
        return '\n'.join(recommendations)
    
    def create_visualization(self, save_path: str = "fatigue_analysis.png") -> bool:
        """
        创建可视化图表
        
        Args:
            save_path: 图表保存路径
            
        Returns:
            是否成功创建
        """
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib未安装，无法创建可视化图表")
            return False
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('疲劳检测分析报告', fontsize=16)
            
            # 疲劳水平时间线
            if self.fatigue_timeline:
                times, levels = zip(*self.fatigue_timeline)
                axes[0, 0].plot(times, levels, 'r-', linewidth=2)
                axes[0, 0].set_title('疲劳水平时间线')
                axes[0, 0].set_ylabel('疲劳等级')
                axes[0, 0].grid(True)
            
            # 注意力水平时间线
            if self.attention_timeline:
                times, levels = zip(*self.attention_timeline)
                axes[0, 1].plot(times, levels, 'b-', linewidth=2)
                axes[0, 1].set_title('注意力水平时间线')
                axes[0, 1].set_ylabel('注意力水平')
                axes[0, 1].grid(True)
            
            # 事件统计柱状图
            summary = self.get_session_summary()
            events = ['眨眼', '哈欠', '点头']
            counts = [summary['blink_count'], summary['yawn_count'], summary['nod_count']]
            axes[1, 0].bar(events, counts, color=['blue', 'orange', 'green'])
            axes[1, 0].set_title('检测事件统计')
            axes[1, 0].set_ylabel('次数')
            
            # 疲劳等级分布饼图
            if self.fatigue_timeline:
                levels = [v for _, v in self.fatigue_timeline]
                level_counts = [levels.count(i) for i in range(4)]
                level_labels = ['正常', '轻微疲劳', '中度疲劳', '重度疲劳']
                colors = ['green', 'yellow', 'orange', 'red']
                
                # 只显示非零的等级
                non_zero_indices = [i for i, count in enumerate(level_counts) if count > 0]
                if non_zero_indices:
                    filtered_counts = [level_counts[i] for i in non_zero_indices]
                    filtered_labels = [level_labels[i] for i in non_zero_indices]
                    filtered_colors = [colors[i] for i in non_zero_indices]
                    
                    axes[1, 1].pie(filtered_counts, labels=filtered_labels, colors=filtered_colors, autopct='%1.1f%%')
                    axes[1, 1].set_title('疲劳等级分布')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"可视化图表已保存到: {save_path}")
            return True
            
        except Exception as e:
            print(f"创建可视化图表失败: {e}")
            return False
    
    def end_session(self):
        """结束当前会话"""
        try:
            end_time = datetime.datetime.now()
            duration = (end_time - self.session_start_time).total_seconds() / 60
            
            summary = self.get_session_summary()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE sessions SET 
                end_time = ?, duration_minutes = ?, total_blinks = ?, 
                total_yawns = ?, total_nods = ?, fatigue_episodes = ?,
                max_fatigue_level = ?, avg_attention_level = ?
                WHERE id = ?
            ''', (end_time, duration, summary['blink_count'], summary['yawn_count'],
                  summary['nod_count'], summary['fatigue_episodes'], 
                  summary['max_fatigue_level'], summary['avg_attention_level'],
                  self.current_session_id))
            
            conn.commit()
            conn.close()
            
            print(f"会话结束 - 持续时间: {duration:.2f} 分钟")

        except Exception as e:
            print(f"结束会话失败: {e}")

    def get_historical_sessions(self, limit: int = 10) -> List[Dict]:
        """
        获取历史会话数据

        Args:
            limit: 返回的会话数量限制

        Returns:
            历史会话列表
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT id, start_time, end_time, duration_minutes,
                       total_blinks, total_yawns, total_nods,
                       fatigue_episodes, max_fatigue_level, avg_attention_level
                FROM sessions
                ORDER BY start_time DESC
                LIMIT ?
            ''', (limit,))

            sessions = []
            for row in cursor.fetchall():
                session_id, start_time, end_time, duration, blinks, yawns, nods, episodes, max_fatigue, avg_attention = row

                sessions.append({
                    'id': session_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration_minutes': duration or 0,
                    'total_blinks': blinks or 0,
                    'total_yawns': yawns or 0,
                    'total_nods': nods or 0,
                    'fatigue_episodes': episodes or 0,
                    'max_fatigue_level': max_fatigue or 0,
                    'avg_attention_level': avg_attention or 0
                })

            conn.close()
            return sessions

        except Exception as e:
            print(f"获取历史会话失败: {e}")
            return []
