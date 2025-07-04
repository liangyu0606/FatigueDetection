"""
恢复训练脚本 - 处理中断后的恢复
"""
import os
from config import *

def main():
    """主函数"""
    print("🔄 恢复完整训练流程")
    print("="*50)
    
    # 检查是否有部分处理的数据
    processed_file = os.path.join(PROCESSED_DATA_PATH, "processed_samples.pkl")
    
    if os.path.exists(processed_file):
        print(f"✅ 找到已处理的数据: {processed_file}")
        
        # 直接开始训练
        print("\n=== 开始模型训练 ===")
        from train import main as train_main
        train_main()
        
        # 训练完成后评估
        print("\n=== 开始模型评估 ===")
        from evaluate import main as eval_main
        eval_main()
        
    else:
        print("❌ 没有找到预处理数据")
        print("重新开始完整流程...")
        
        # 重新预处理
        print("\n=== 开始数据预处理 ===")
        from data_preprocessing import FatigueDataPreprocessor
        
        preprocessor = FatigueDataPreprocessor()
        samples = preprocessor.process_dataset(DATASET_ROOT)
        
        if samples:
            output_file = os.path.join(PROCESSED_DATA_PATH, "processed_samples.pkl")
            preprocessor.save_processed_data(samples, output_file)
            print("✅ 数据预处理完成!")
            
            # 开始训练
            print("\n=== 开始模型训练 ===")
            from train import main as train_main
            train_main()
            
            # 评估
            print("\n=== 开始模型评估 ===")
            from evaluate import main as eval_main
            eval_main()
        else:
            print("❌ 数据预处理失败")
    
    print("\n🎉 完整流程执行完成!")

if __name__ == "__main__":
    main()
