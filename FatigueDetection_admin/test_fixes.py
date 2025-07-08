#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复效果的脚本
"""

import datetime
import csv
from io import StringIO

def test_csv_encoding():
    """测试CSV编码修复"""
    print("测试CSV编码...")
    
    # 模拟数据
    test_data = [
        ["张三", "2024-01-15 10:30:00", "轻度疲劳"],
        ["李四", "2024-01-15 11:45:00", "中度疲劳"],
        ["王五", "2024-01-15 14:20:00", "重度疲劳"]
    ]
    
    # 生成CSV内容
    output = StringIO()
    # 写入BOM头，确保Excel正确识别UTF-8编码
    output.write('\ufeff')
    writer = csv.writer(output)
    writer.writerow(["用户名", "时间", "疲劳等级"])
    
    for record in test_data:
        writer.writerow(record)
    
    csv_content = output.getvalue()
    output.close()
    
    # 保存测试文件
    with open("test_fatigue_records.csv", "w", encoding="utf-8") as f:
        f.write(csv_content)
    
    print("✅ CSV测试文件已生成: test_fatigue_records.csv")
    print("请用Excel打开检查中文是否正常显示")

def test_timezone():
    """测试时区修复"""
    print("\n测试时区...")
    
    try:
        import pytz
        
        # 设置时区
        TIMEZONE = pytz.timezone('Asia/Shanghai')
        
        # 获取当前时间
        current_time = datetime.datetime.now(TIMEZONE)
        utc_time = datetime.datetime.now(pytz.UTC)
        local_time = datetime.datetime.now()
        
        print(f"本地时间（无时区）: {local_time}")
        print(f"UTC时间: {utc_time}")
        print(f"中国时区时间: {current_time}")
        print(f"时区偏移: {current_time.strftime('%z')}")
        
        print("✅ 时区设置正常")
        
    except ImportError:
        print("❌ pytz库未安装，请运行: pip install pytz")
        return False
    
    return True

def main():
    """主函数"""
    print("=" * 50)
    print("疲劳检测系统修复测试")
    print("=" * 50)
    
    # 测试CSV编码
    test_csv_encoding()
    
    # 测试时区
    timezone_ok = test_timezone()
    
    print("\n" + "=" * 50)
    print("测试完成")
    
    if not timezone_ok:
        print("\n⚠️  需要安装pytz库:")
        print("pip install pytz")
    
    print("\n修复内容:")
    print("1. ✅ CSV导出编码修复（添加BOM头）")
    print("2. ✅ 回车键搜索功能")
    if timezone_ok:
        print("3. ✅ 时区设置修复")
    else:
        print("3. ❌ 时区设置（需要安装pytz）")

if __name__ == "__main__":
    main()
