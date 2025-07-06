"""
数据库配置文件
用于管理MySQL数据库连接配置
"""

import pymysql
import datetime
from contextlib import contextmanager

# MySQL数据库配置
DB_CONFIG = {
    'host': '101.245.79.154',
    'port': 3306,
    'user': 'root',
    'password': '123456',
    'database': 'pljc',
    'charset': 'utf8mb4',
    'autocommit': True
}

@contextmanager
def get_db_connection():
    """
    获取数据库连接的上下文管理器
    自动处理连接的打开和关闭
    """
    connection = None
    try:
        connection = pymysql.connect(**DB_CONFIG)
        yield connection
    except Exception as e:
        if connection:
            connection.rollback()
        raise e
    finally:
        if connection:
            connection.close()

def init_database():
    """
    初始化数据库表结构
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 创建用户表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    username VARCHAR(50) PRIMARY KEY,
                    password VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            ''')
            
            # 创建疲劳记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fatigue_records (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(50) NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    fatigue_level VARCHAR(20) NOT NULL,
                    INDEX idx_username (username),
                    INDEX idx_timestamp (timestamp),
                    FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            ''')
            
            conn.commit()
            print("数据库初始化成功")
            
    except Exception as e:
        print(f"数据库初始化失败: {e}")
        raise e

def test_connection():
    """
    测试数据库连接
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            if result:
                print("数据库连接测试成功")
                return True
            else:
                print("数据库连接测试失败")
                return False
    except Exception as e:
        print(f"数据库连接测试失败: {e}")
        return False

if __name__ == "__main__":
    # 测试数据库连接和初始化
    print("测试数据库连接...")
    if test_connection():
        print("初始化数据库表...")
        init_database()
        print("数据库配置完成")
    else:
        print("数据库连接失败，请检查配置")
