#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用户权限管理模块
提供用户认证、权限控制、会话管理等功能
"""

import hashlib
import sqlite3
import datetime
import uuid
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum


class UserRole(Enum):
    """用户角色枚举"""
    ADMIN = "admin"           # 管理员：完全权限
    DRIVER = "driver"         # 驾驶员：基本检测功能
    MONITOR = "monitor"       # 监控人员：查看和分析权限
    GUEST = "guest"          # 访客：只读权限


class Permission(Enum):
    """权限枚举"""
    # 系统管理权限
    USER_MANAGE = "user_manage"           # 用户管理
    SYSTEM_CONFIG = "system_config"       # 系统配置
    LOG_MANAGE = "log_manage"            # 日志管理
    
    # 检测功能权限
    DETECTION_START = "detection_start"   # 启动检测
    DETECTION_STOP = "detection_stop"     # 停止检测
    DETECTION_CONFIG = "detection_config" # 检测配置
    
    # 数据权限
    DATA_VIEW = "data_view"              # 查看数据
    DATA_EXPORT = "data_export"          # 导出数据
    DATA_DELETE = "data_delete"          # 删除数据
    
    # 报告权限
    REPORT_VIEW = "report_view"          # 查看报告
    REPORT_GENERATE = "report_generate"   # 生成报告
    REPORT_EXPORT = "report_export"      # 导出报告


@dataclass
class User:
    """用户数据类"""
    user_id: str
    username: str
    password_hash: str
    role: UserRole
    email: str = ""
    full_name: str = ""
    created_at: datetime.datetime = None
    last_login: datetime.datetime = None
    is_active: bool = True
    permissions: List[Permission] = None


class UserManager:
    """用户管理器"""
    
    # 角色权限映射
    ROLE_PERMISSIONS = {
        UserRole.ADMIN: [
            Permission.USER_MANAGE, Permission.SYSTEM_CONFIG, Permission.LOG_MANAGE,
            Permission.DETECTION_START, Permission.DETECTION_STOP, Permission.DETECTION_CONFIG,
            Permission.DATA_VIEW, Permission.DATA_EXPORT, Permission.DATA_DELETE,
            Permission.REPORT_VIEW, Permission.REPORT_GENERATE, Permission.REPORT_EXPORT
        ],
        UserRole.MONITOR: [
            Permission.DATA_VIEW, Permission.DATA_EXPORT,
            Permission.REPORT_VIEW, Permission.REPORT_GENERATE, Permission.REPORT_EXPORT,
            Permission.LOG_MANAGE
        ],
        UserRole.DRIVER: [
            Permission.DETECTION_START, Permission.DETECTION_STOP,
            Permission.DATA_VIEW, Permission.REPORT_VIEW
        ],
        UserRole.GUEST: [
            Permission.DATA_VIEW, Permission.REPORT_VIEW
        ]
    }
    
    def __init__(self, db_path: str = "fatigue_data.db"):
        """
        初始化用户管理器
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.current_user = None
        self.session_token = None
        self._init_user_database()
        self._create_default_admin()
    
    def _init_user_database(self):
        """初始化用户数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 创建用户表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    email TEXT,
                    full_name TEXT,
                    created_at TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    permissions TEXT
                )
            ''')
            
            # 创建会话表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    created_at TIMESTAMP,
                    expires_at TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            conn.commit()
            conn.close()
            print("用户数据库初始化成功")
            
        except Exception as e:
            print(f"用户数据库初始化失败: {e}")
    
    def _create_default_admin(self):
        """创建默认管理员账户"""
        try:
            # 检查是否已存在管理员
            if not self.get_user_by_username("admin"):
                admin_user = User(
                    user_id=str(uuid.uuid4()),
                    username="admin",
                    password_hash=self._hash_password("admin123"),
                    role=UserRole.ADMIN,
                    email="admin@fatigue-system.com",
                    full_name="系统管理员",
                    created_at=datetime.datetime.now(),
                    is_active=True,
                    permissions=self.ROLE_PERMISSIONS[UserRole.ADMIN]
                )
                
                if self._save_user_to_db(admin_user):
                    print("默认管理员账户创建成功 (用户名: admin, 密码: admin123)")
                else:
                    print("默认管理员账户创建失败")
                    
        except Exception as e:
            print(f"创建默认管理员失败: {e}")
    
    def _hash_password(self, password: str) -> str:
        """密码哈希"""
        salt = "fatigue_detection_system_salt"
        return hashlib.sha256((password + salt).encode()).hexdigest()
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """验证密码"""
        return self._hash_password(password) == password_hash
    
    def register_user(self, username: str, password: str, role: UserRole, 
                     email: str = "", full_name: str = "") -> bool:
        """
        注册新用户
        
        Args:
            username: 用户名
            password: 密码
            role: 用户角色
            email: 邮箱
            full_name: 全名
            
        Returns:
            是否注册成功
        """
        try:
            # 检查用户名是否已存在
            if self.get_user_by_username(username):
                print(f"用户名 '{username}' 已存在")
                return False
            
            # 创建新用户
            user = User(
                user_id=str(uuid.uuid4()),
                username=username,
                password_hash=self._hash_password(password),
                role=role,
                email=email,
                full_name=full_name,
                created_at=datetime.datetime.now(),
                is_active=True,
                permissions=self.ROLE_PERMISSIONS.get(role, [])
            )
            
            return self._save_user_to_db(user)
            
        except Exception as e:
            print(f"用户注册失败: {e}")
            return False
    
    def login(self, username: str, password: str) -> Tuple[bool, str]:
        """
        用户登录
        
        Args:
            username: 用户名
            password: 密码
            
        Returns:
            (是否登录成功, 会话令牌或错误信息)
        """
        try:
            user = self.get_user_by_username(username)
            if not user:
                return False, "用户不存在"
            
            if not user.is_active:
                return False, "用户已被禁用"
            
            if not self._verify_password(password, user.password_hash):
                return False, "密码错误"
            
            # 创建会话
            session_token = str(uuid.uuid4())
            expires_at = datetime.datetime.now() + datetime.timedelta(hours=8)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO user_sessions (session_id, user_id, created_at, expires_at)
                VALUES (?, ?, ?, ?)
            ''', (session_token, user.user_id, datetime.datetime.now(), expires_at))
            
            # 更新最后登录时间
            cursor.execute('''
                UPDATE users SET last_login = ? WHERE user_id = ?
            ''', (datetime.datetime.now(), user.user_id))
            
            conn.commit()
            conn.close()
            
            self.current_user = user
            self.session_token = session_token
            
            print(f"用户 '{username}' 登录成功")
            return True, session_token
            
        except Exception as e:
            print(f"登录失败: {e}")
            return False, "登录过程中发生错误"
    
    def logout(self) -> bool:
        """用户登出"""
        try:
            if self.session_token:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE user_sessions SET is_active = 0 WHERE session_id = ?
                ''', (self.session_token,))
                
                conn.commit()
                conn.close()
            
            self.current_user = None
            self.session_token = None
            print("用户已登出")
            return True
            
        except Exception as e:
            print(f"登出失败: {e}")
            return False
    
    def verify_session(self, session_token: str) -> bool:
        """验证会话令牌"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_id, expires_at FROM user_sessions 
                WHERE session_id = ? AND is_active = 1
            ''', (session_token,))
            
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                return False
            
            user_id, expires_at_str = result
            expires_at = datetime.datetime.fromisoformat(expires_at_str)
            
            if datetime.datetime.now() > expires_at:
                return False
            
            # 加载用户信息
            user = self.get_user_by_id(user_id)
            if user and user.is_active:
                self.current_user = user
                self.session_token = session_token
                return True
            
            return False
            
        except Exception as e:
            print(f"会话验证失败: {e}")
            return False
    
    def has_permission(self, permission: Permission) -> bool:
        """检查当前用户是否有指定权限"""
        if not self.current_user:
            return False
        
        return permission in self.current_user.permissions
    
    def require_permission(self, permission: Permission) -> bool:
        """要求指定权限，如果没有权限则抛出异常"""
        if not self.has_permission(permission):
            raise PermissionError(f"需要权限: {permission.value}")
        return True
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """根据用户名获取用户"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM users WHERE username = ?
            ''', (username,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return self._row_to_user(result)
            return None
            
        except Exception as e:
            print(f"获取用户失败: {e}")
            return None
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """根据用户ID获取用户"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM users WHERE user_id = ?
            ''', (user_id,))

            result = cursor.fetchone()
            conn.close()

            if result:
                return self._row_to_user(result)
            return None

        except Exception as e:
            print(f"获取用户失败: {e}")
            return None

    def get_all_users(self) -> List[User]:
        """获取所有用户"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('SELECT * FROM users ORDER BY created_at DESC')
            results = cursor.fetchall()
            conn.close()

            users = []
            for row in results:
                try:
                    user = self._row_to_user(row)
                    users.append(user)
                except Exception as e:
                    print(f"转换用户数据失败 (row: {row}): {e}")
                    continue

            return users

        except Exception as e:
            print(f"获取用户列表失败: {e}")
            return []

    def update_user(self, user_id: str, **kwargs) -> bool:
        """更新用户信息"""
        try:
            user = self.get_user_by_id(user_id)
            if not user:
                return False

            # 更新用户属性
            for key, value in kwargs.items():
                if hasattr(user, key):
                    if key == 'password':
                        user.password_hash = self._hash_password(value)
                    elif key == 'role':
                        user.role = UserRole(value) if isinstance(value, str) else value
                        user.permissions = self.ROLE_PERMISSIONS.get(user.role, [])
                    else:
                        setattr(user, key, value)

            return self._save_user_to_db(user, update=True)

        except Exception as e:
            print(f"更新用户失败: {e}")
            return False

    def delete_user(self, user_id: str) -> bool:
        """删除用户"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 删除用户会话
            cursor.execute('DELETE FROM user_sessions WHERE user_id = ?', (user_id,))

            # 删除用户
            cursor.execute('DELETE FROM users WHERE user_id = ?', (user_id,))

            conn.commit()
            conn.close()

            print(f"用户 {user_id} 已删除")
            return True

        except Exception as e:
            print(f"删除用户失败: {e}")
            return False

    def _save_user_to_db(self, user: User, update: bool = False) -> bool:
        """保存用户到数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            permissions_json = json.dumps([p.value for p in user.permissions]) if user.permissions else None

            if update:
                cursor.execute('''
                    UPDATE users SET
                    username = ?, password_hash = ?, role = ?, email = ?,
                    full_name = ?, last_login = ?, is_active = ?, permissions = ?
                    WHERE user_id = ?
                ''', (user.username, user.password_hash, user.role.value, user.email,
                      user.full_name, user.last_login, user.is_active,
                      permissions_json, user.user_id))
            else:
                cursor.execute('''
                    INSERT INTO users
                    (user_id, username, password_hash, role, email, full_name,
                     created_at, last_login, is_active, permissions)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (user.user_id, user.username, user.password_hash, user.role.value,
                      user.email, user.full_name, user.created_at, user.last_login,
                      user.is_active, permissions_json))

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            print(f"保存用户失败: {e}")
            return False

    def _row_to_user(self, row) -> User:
        """数据库行转换为用户对象"""
        try:
            # 解析权限
            permissions = []
            if row[9]:  # permissions字段
                try:
                    permission_values = json.loads(row[9])
                    permissions = [Permission(p) for p in permission_values]
                except Exception as e:
                    print(f"解析权限失败: {e}, 使用默认权限")
                    permissions = self.ROLE_PERMISSIONS.get(UserRole(row[3]), [])
            else:
                permissions = self.ROLE_PERMISSIONS.get(UserRole(row[3]), [])

            # 解析时间字段
            created_at = None
            if row[6]:
                try:
                    created_at = datetime.datetime.fromisoformat(row[6])
                except:
                    try:
                        created_at = datetime.datetime.strptime(row[6], "%Y-%m-%d %H:%M:%S.%f")
                    except:
                        print(f"无法解析创建时间: {row[6]}")

            last_login = None
            if row[7]:
                try:
                    last_login = datetime.datetime.fromisoformat(row[7])
                except:
                    try:
                        last_login = datetime.datetime.strptime(row[7], "%Y-%m-%d %H:%M:%S.%f")
                    except:
                        print(f"无法解析最后登录时间: {row[7]}")

            return User(
                user_id=row[0] or "",
                username=row[1] or "",
                password_hash=row[2] or "",
                role=UserRole(row[3]) if row[3] else UserRole.GUEST,
                email=row[4] or "",
                full_name=row[5] or "",
                created_at=created_at,
                last_login=last_login,
                is_active=bool(row[8]) if row[8] is not None else True,
                permissions=permissions
            )
        except Exception as e:
            print(f"转换用户对象失败: {e}, row: {row}")
            raise
