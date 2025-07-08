#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
疲劳检测系统管理员Web服务启动脚本
运行在 localhost:8001
"""

import uvicorn
import sys
import os


def check_dependencies():
    """检查依赖包"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'jinja2',
        'python-multipart',
        'pymysql'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n请运行以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    return True


def check_database():
    """检查数据库连接"""
    try:
        from database_config import test_connection, init_database
        print("🔍 检查数据库连接...")

        if test_connection():
            print("✅ 数据库连接成功")
            print("🔧 初始化数据库表...")
            init_database()
            print("✅ 数据库初始化完成")
            return True
        else:
            print("❌ 数据库连接失败")
            print("请检查 database_config.py 中的数据库配置")
            return False

    except Exception as e:
        print(f"❌ 数据库检查失败: {e}")
        return False


def main():
    """主函数"""
    print("🔧 疲劳检测系统 - 管理员Web服务")
    print("=" * 50)

    # 检查依赖
    if not check_dependencies():
        return

    # 检查数据库
    if not check_database():
        print("\n⚠️  数据库连接失败，但服务仍可启动（部分功能可能不可用）")
        response = input("是否继续启动服务？(y/N): ")
        if response.lower() != 'y':
            return

    print("\n🌐 启动管理员Web服务...")
    print("📍 服务地址: http://127.0.0.1:8001")
    print("👤 用户界面: http://127.0.0.1:8000")
    print("=" * 50)
    print("💡 功能说明:")
    print("   - 管理员登录")
    print("   - 疲劳记录查询和统计")
    print("   - 用户管理")
    print("   - 数据导出")
    print("=" * 50)
    print("按 Ctrl+C 停止服务")
    print()

    try:
        # 检查是否在Docker容器中运行
        is_docker = os.path.exists('/.dockerenv')
        host = "0.0.0.0" if is_docker else "127.0.0.1"
        reload = not is_docker  # 在Docker中禁用reload

        uvicorn.run(
            "fatigue_web_admin:app",
            host=host,
            port=8001,
            reload=reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 管理员服务已停止")
    except Exception as e:
        print(f"❌ 服务启动失败: {e}")


if __name__ == "__main__":
    main()
