#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统日志记录模块
记录用户操作行为、系统状态变化、疲劳检测事件等信息
"""

import sqlite3
import datetime
import json
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


class LogLevel(Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogCategory(Enum):
    """日志分类"""
    USER_ACTION = "user_action"         # 用户操作
    SYSTEM_EVENT = "system_event"       # 系统事件
    FATIGUE_DETECTION = "fatigue_detection"  # 疲劳检测
    SECURITY = "security"               # 安全相关
    PERFORMANCE = "performance"         # 性能监控
    ERROR = "error"                     # 错误日志


@dataclass
class LogEntry:
    """日志条目数据类"""
    log_id: str
    timestamp: datetime.datetime
    level: LogLevel
    category: LogCategory
    user_id: str = None
    username: str = None
    action: str = ""
    description: str = ""
    details: Dict[str, Any] = None
    ip_address: str = ""
    user_agent: str = ""
    session_id: str = ""


class SystemLogger:
    """系统日志记录器"""
    
    def __init__(self, db_path: str = "fatigue_data.db"):
        """
        初始化日志记录器
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self._init_log_database()
    
    def _init_log_database(self):
        """初始化日志数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 创建系统日志表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_logs (
                    log_id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    level TEXT NOT NULL,
                    category TEXT NOT NULL,
                    user_id TEXT,
                    username TEXT,
                    action TEXT,
                    description TEXT,
                    details TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    session_id TEXT
                )
            ''')
            
            # 创建索引以提高查询性能
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON system_logs(timestamp)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_logs_category ON system_logs(category)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_logs_user ON system_logs(user_id)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_logs_level ON system_logs(level)
            ''')
            
            conn.commit()
            conn.close()
            print("日志数据库初始化成功")
            
        except Exception as e:
            print(f"日志数据库初始化失败: {e}")
    
    def log(self, level: LogLevel, category: LogCategory, action: str, 
            description: str = "", user_id: str = None, username: str = None,
            details: Dict[str, Any] = None, ip_address: str = "",
            user_agent: str = "", session_id: str = "") -> bool:
        """
        记录日志
        
        Args:
            level: 日志级别
            category: 日志分类
            action: 操作动作
            description: 描述信息
            user_id: 用户ID
            username: 用户名
            details: 详细信息
            ip_address: IP地址
            user_agent: 用户代理
            session_id: 会话ID
            
        Returns:
            是否记录成功
        """
        try:
            log_entry = LogEntry(
                log_id=str(uuid.uuid4()),
                timestamp=datetime.datetime.now(),
                level=level,
                category=category,
                user_id=user_id,
                username=username,
                action=action,
                description=description,
                details=details,
                ip_address=ip_address,
                user_agent=user_agent,
                session_id=session_id
            )
            
            return self._save_log_to_db(log_entry)
            
        except Exception as e:
            print(f"记录日志失败: {e}")
            return False
    
    def log_user_action(self, action: str, description: str = "", 
                       user_id: str = None, username: str = None,
                       details: Dict[str, Any] = None, session_id: str = "") -> bool:
        """记录用户操作日志"""
        return self.log(
            level=LogLevel.INFO,
            category=LogCategory.USER_ACTION,
            action=action,
            description=description,
            user_id=user_id,
            username=username,
            details=details,
            session_id=session_id
        )
    
    def log_system_event(self, action: str, description: str = "", 
                        level: LogLevel = LogLevel.INFO,
                        details: Dict[str, Any] = None) -> bool:
        """记录系统事件日志"""
        return self.log(
            level=level,
            category=LogCategory.SYSTEM_EVENT,
            action=action,
            description=description,
            details=details
        )
    
    def log_fatigue_detection(self, action: str, description: str = "",
                             user_id: str = None, username: str = None,
                             details: Dict[str, Any] = None) -> bool:
        """记录疲劳检测日志"""
        return self.log(
            level=LogLevel.INFO,
            category=LogCategory.FATIGUE_DETECTION,
            action=action,
            description=description,
            user_id=user_id,
            username=username,
            details=details
        )
    
    def log_security_event(self, action: str, description: str = "",
                          level: LogLevel = LogLevel.WARNING,
                          user_id: str = None, username: str = None,
                          ip_address: str = "", details: Dict[str, Any] = None) -> bool:
        """记录安全事件日志"""
        return self.log(
            level=level,
            category=LogCategory.SECURITY,
            action=action,
            description=description,
            user_id=user_id,
            username=username,
            ip_address=ip_address,
            details=details
        )
    
    def log_error(self, action: str, description: str = "", 
                  error_details: str = "", user_id: str = None,
                  username: str = None) -> bool:
        """记录错误日志"""
        details = {"error_details": error_details} if error_details else None
        return self.log(
            level=LogLevel.ERROR,
            category=LogCategory.ERROR,
            action=action,
            description=description,
            user_id=user_id,
            username=username,
            details=details
        )
    
    def get_logs(self, start_time: datetime.datetime = None,
                end_time: datetime.datetime = None,
                level: LogLevel = None, category: LogCategory = None,
                user_id: str = None, limit: int = 1000,
                offset: int = 0) -> List[LogEntry]:
        """
        获取日志记录
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            level: 日志级别过滤
            category: 分类过滤
            user_id: 用户ID过滤
            limit: 返回记录数限制
            offset: 偏移量
            
        Returns:
            日志记录列表
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 构建查询条件
            conditions = []
            params = []
            
            if start_time:
                conditions.append("timestamp >= ?")
                params.append(start_time.isoformat())
            
            if end_time:
                conditions.append("timestamp <= ?")
                params.append(end_time.isoformat())
            
            if level:
                conditions.append("level = ?")
                params.append(level.value)
            
            if category:
                conditions.append("category = ?")
                params.append(category.value)
            
            if user_id:
                conditions.append("user_id = ?")
                params.append(user_id)
            
            where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
            
            query = f'''
                SELECT * FROM system_logs
                {where_clause}
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            '''
            
            params.extend([limit, offset])
            cursor.execute(query, params)
            results = cursor.fetchall()
            conn.close()
            
            return [self._row_to_log_entry(row) for row in results]
            
        except Exception as e:
            print(f"获取日志失败: {e}")
            return []
    
    def get_log_statistics(self, start_time: datetime.datetime = None,
                          end_time: datetime.datetime = None) -> Dict[str, Any]:
        """
        获取日志统计信息
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            统计信息字典
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 时间条件
            time_condition = ""
            params = []
            if start_time:
                time_condition += " AND timestamp >= ?"
                params.append(start_time.isoformat())
            if end_time:
                time_condition += " AND timestamp <= ?"
                params.append(end_time.isoformat())
            
            # 按级别统计
            cursor.execute(f'''
                SELECT level, COUNT(*) FROM system_logs 
                WHERE 1=1 {time_condition}
                GROUP BY level
            ''', params)
            level_stats = dict(cursor.fetchall())
            
            # 按分类统计
            cursor.execute(f'''
                SELECT category, COUNT(*) FROM system_logs 
                WHERE 1=1 {time_condition}
                GROUP BY category
            ''', params)
            category_stats = dict(cursor.fetchall())
            
            # 按用户统计
            cursor.execute(f'''
                SELECT username, COUNT(*) FROM system_logs 
                WHERE username IS NOT NULL {time_condition}
                GROUP BY username
                ORDER BY COUNT(*) DESC
                LIMIT 10
            ''', params)
            user_stats = dict(cursor.fetchall())
            
            # 总数统计
            cursor.execute(f'''
                SELECT COUNT(*) FROM system_logs 
                WHERE 1=1 {time_condition}
            ''', params)
            total_count = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'total_count': total_count,
                'level_statistics': level_stats,
                'category_statistics': category_stats,
                'user_statistics': user_stats,
                'time_range': {
                    'start_time': start_time.isoformat() if start_time else None,
                    'end_time': end_time.isoformat() if end_time else None
                }
            }
            
        except Exception as e:
            print(f"获取日志统计失败: {e}")
            return {}
    
    def clean_old_logs(self, days_to_keep: int = 90) -> int:
        """
        清理旧日志
        
        Args:
            days_to_keep: 保留天数
            
        Returns:
            删除的记录数
        """
        try:
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_to_keep)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                DELETE FROM system_logs WHERE timestamp < ?
            ''', (cutoff_date.isoformat(),))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            print(f"清理了 {deleted_count} 条旧日志记录")
            return deleted_count
            
        except Exception as e:
            print(f"清理日志失败: {e}")
            return 0
    
    def _save_log_to_db(self, log_entry: LogEntry) -> bool:
        """保存日志到数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            details_json = json.dumps(log_entry.details) if log_entry.details else None
            
            cursor.execute('''
                INSERT INTO system_logs 
                (log_id, timestamp, level, category, user_id, username, 
                 action, description, details, ip_address, user_agent, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                log_entry.log_id, log_entry.timestamp.isoformat(),
                log_entry.level.value, log_entry.category.value,
                log_entry.user_id, log_entry.username,
                log_entry.action, log_entry.description,
                details_json, log_entry.ip_address,
                log_entry.user_agent, log_entry.session_id
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"保存日志失败: {e}")
            return False
    
    def _row_to_log_entry(self, row) -> LogEntry:
        """数据库行转换为日志条目对象"""
        details = None
        if row[8]:  # details字段
            try:
                details = json.loads(row[8])
            except:
                details = None
        
        return LogEntry(
            log_id=row[0],
            timestamp=datetime.datetime.fromisoformat(row[1]),
            level=LogLevel(row[2]),
            category=LogCategory(row[3]),
            user_id=row[4],
            username=row[5],
            action=row[6],
            description=row[7],
            details=details,
            ip_address=row[9] or "",
            user_agent=row[10] or "",
            session_id=row[11] or ""
        )
