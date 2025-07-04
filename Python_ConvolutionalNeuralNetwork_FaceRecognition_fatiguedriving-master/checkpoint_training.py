"""
断点训练脚本 - 从上次中断的地方继续训练
"""
import os
import argparse
import torch
from datetime import datetime
from train import main as train_main
from config import MODEL_SAVE_PATH

def check_checkpoint_exists():
    """检查是否存在检查点文件"""
    checkpoint_path = os.path.join(MODEL_SAVE_PATH, 'latest_model.pth')
    return os.path.exists(checkpoint_path)

def get_checkpoint_info(checkpoint_path):
    """获取检查点信息"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        info = {
            'epoch': checkpoint.get('epoch', 0),
            'val_acc': checkpoint.get('val_acc', 0.0),
            'best_val_acc': checkpoint.get('best_val_acc', 0.0),
            'timestamp': checkpoint.get('timestamp', 'Unknown'),
            'config': checkpoint.get('config', {})
        }
        return info
    except Exception as e:
        print(f"读取检查点信息失败: {e}")
        return None

def list_available_checkpoints():
    """列出所有可用的检查点"""
    if not os.path.exists(MODEL_SAVE_PATH):
        return []
    
    checkpoints = []
    for file in os.listdir(MODEL_SAVE_PATH):
        if file.endswith('.pth'):
            file_path = os.path.join(MODEL_SAVE_PATH, file)
            file_size = os.path.getsize(file_path) / (1024*1024)  # MB
            
            # 获取检查点信息
            info = get_checkpoint_info(file_path)
            
            checkpoints.append({
                'filename': file,
                'filepath': file_path,
                'size_mb': file_size,
                'info': info
            })
    
    return checkpoints

def display_checkpoint_info(checkpoint_info):
    """显示检查点详细信息"""
    if checkpoint_info['info']:
        info = checkpoint_info['info']
        print(f"\n📋 检查点详细信息:")
        print(f"  文件名: {checkpoint_info['filename']}")
        print(f"  文件大小: {checkpoint_info['size_mb']:.1f} MB")
        print(f"  训练轮次: {info['epoch']}")
        print(f"  验证准确率: {info['val_acc']:.4f}")
        print(f"  最佳准确率: {info['best_val_acc']:.4f}")
        print(f"  保存时间: {info['timestamp']}")
        
        if info['config']:
            print(f"  训练配置:")
            for key, value in info['config'].items():
                print(f"    {key}: {value}")
    else:
        print(f"  文件名: {checkpoint_info['filename']}")
        print(f"  文件大小: {checkpoint_info['size_mb']:.1f} MB")
        print(f"  ⚠️ 无法读取检查点信息")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='断点训练脚本')
    parser.add_argument('--force', action='store_true', 
                       help='强制开始新训练（忽略现有检查点）')
    parser.add_argument('--list', action='store_true',
                       help='列出所有可用的检查点')
    parser.add_argument('--info', type=str,
                       help='显示指定检查点的详细信息')
    
    args = parser.parse_args()
    
    print("🔄 疲劳检测模型断点训练")
    print("="*50)
    
    # 显示检查点信息
    if args.info:
        checkpoint_path = os.path.join(MODEL_SAVE_PATH, args.info)
        if os.path.exists(checkpoint_path):
            info = get_checkpoint_info(checkpoint_path)
            if info:
                display_checkpoint_info({
                    'filename': args.info,
                    'filepath': checkpoint_path,
                    'size_mb': os.path.getsize(checkpoint_path) / (1024*1024),
                    'info': info
                })
            else:
                print(f"❌ 无法读取检查点信息: {args.info}")
        else:
            print(f"❌ 检查点文件不存在: {args.info}")
        return
    
    # 列出检查点
    if args.list:
        checkpoints = list_available_checkpoints()
        if checkpoints:
            print("📁 可用的检查点文件:")
            for i, checkpoint in enumerate(checkpoints, 1):
                print(f"\n{i}. {checkpoint['filename']} ({checkpoint['size_mb']:.1f} MB)")
                if checkpoint['info']:
                    info = checkpoint['info']
                    print(f"   轮次: {info['epoch']}, 验证准确率: {info['val_acc']:.4f}")
                    print(f"   时间: {info['timestamp']}")
        else:
            print("❌ 没有找到检查点文件")
        return
    
    # 检查是否存在检查点
    if check_checkpoint_exists() and not args.force:
        checkpoint_path = os.path.join(MODEL_SAVE_PATH, 'latest_model.pth')
        info = get_checkpoint_info(checkpoint_path)
        
        print("✅ 找到现有的检查点文件")
        if info:
            print(f"   上次训练到第 {info['epoch']} 轮")
            print(f"   验证准确率: {info['val_acc']:.4f}")
            print(f"   最佳准确率: {info['best_val_acc']:.4f}")
        
        print("\n选择操作:")
        print("1. 继续训练（从上次中断处开始）")
        print("2. 开始新训练（覆盖现有检查点）")
        print("3. 退出")
        
        while True:
            choice = input("\n请选择 (1-3): ").strip()
            if choice == '1':
                print("🔄 开始断点训练...")
                train_main(resume_training=True)
                break
            elif choice == '2':
                confirm = input("⚠️ 确定要开始新训练吗？这将覆盖现有检查点 (y/n): ").lower().strip()
                if confirm in ['y', 'yes', '是']:
                    print("🚀 开始新的训练...")
                    train_main(resume_training=False)
                    break
                else:
                    print("操作已取消")
                    continue
            elif choice == '3':
                print("👋 退出程序")
                return
            else:
                print("❌ 无效选择，请输入 1、2 或 3")
    else:
        if args.force:
            print("🚀 强制开始新训练...")
        else:
            print("❌ 没有找到检查点文件")
            print("🚀 开始新训练...")
        train_main(resume_training=False)

if __name__ == "__main__":
    main()
