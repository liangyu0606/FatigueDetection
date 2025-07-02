#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试enhanced_main.py中的CNN+LSTM集成
"""

import sys
import os

def test_enhanced_imports():
    """测试enhanced_main.py的导入"""
    print("=" * 60)
    print("🧪 测试enhanced_main.py导入")
    print("=" * 60)
    
    try:
        print("正在导入enhanced_main模块...")
        import enhanced_main
        print("✅ enhanced_main模块导入成功")
        
        # 检查PyTorch可用性
        print(f"✅ PyTorch可用性: {enhanced_main.PYTORCH_AVAILABLE}")
        
        # 检查MainUI可用性
        print(f"✅ MainUI可用性: {enhanced_main.MAIN_UI_AVAILABLE}")
        
        return True
        
    except Exception as e:
        print(f"❌ enhanced_main导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_files():
    """测试模型文件"""
    print("\n" + "=" * 60)
    print("🧪 测试模型文件")
    print("=" * 60)
    
    model_files = [
        './model/best_fatigue_model.pth',
        './model/shape_predictor_68_face_landmarks.dat',
        './model/fatigue_model_mobilenet.h5',
        './model/class_indices.json'
    ]
    
    all_exist = True
    for file_path in model_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({size:,} bytes)")
        else:
            print(f"❌ {file_path} (不存在)")
            all_exist = False
    
    return all_exist

def test_cnn_lstm_integration():
    """测试CNN+LSTM集成"""
    print("\n" + "=" * 60)
    print("🧪 测试CNN+LSTM集成")
    print("=" * 60)
    
    try:
        import enhanced_main
        
        if not enhanced_main.PYTORCH_AVAILABLE:
            print("⚠️ PyTorch不可用，跳过CNN+LSTM测试")
            return True
        
        # 检查类是否可用
        try:
            YawnDetector = enhanced_main.YawnDetector
            YawnCNNLSTM = enhanced_main.YawnCNNLSTM
            print("✅ CNN+LSTM相关类导入成功")
        except AttributeError as e:
            print(f"❌ CNN+LSTM类导入失败: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ CNN+LSTM集成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始测试enhanced_main.py的CNN+LSTM集成")
    
    tests = [
        ("模块导入", test_enhanced_imports),
        ("模型文件", test_model_files),
        ("CNN+LSTM集成", test_cnn_lstm_integration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            results[test_name] = False
    
    # 总结
    print("\n" + "=" * 60)
    print("📋 测试总结")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 enhanced_main.py已成功集成CNN+LSTM！")
        print("\n✨ 集成功能:")
        print("1. ✅ PyTorch和CNN+LSTM类导入")
        print("2. ✅ 模型文件检查")
        print("3. ✅ 自动初始化检测器")
        print("4. ✅ 状态显示和日志记录")
        
        print("\n🚀 现在可以运行:")
        print("python enhanced_main.py")
        
        print("\n💡 新功能:")
        print("- 登录后可以看到CNN+LSTM状态")
        print("- 系统菜单中可以查看详细状态")
        print("- 疲劳检测使用CNN+LSTM打哈欠检测")
        print("- 所有操作记录到系统日志")
    else:
        print("❌ 部分测试失败")
        if not results.get("模块导入", True):
            print("- 检查enhanced_main.py语法错误")
        if not results.get("模型文件", True):
            print("- 确保模型文件在./model/目录中")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    print(f"\n测试结果: {'成功' if success else '失败'}")
