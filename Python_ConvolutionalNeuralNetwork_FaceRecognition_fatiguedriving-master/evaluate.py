"""
模型评估脚本
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report, roc_curve, auc
)
from tqdm import tqdm

from config import *
from model import create_model
from dataset import load_processed_data, create_data_loaders
from utils import setup_logging

class ModelEvaluator:
    def __init__(self, model_path):
        self.logger = setup_logging()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self.model = create_model().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.logger.info(f"模型加载完成: {model_path}")
        self.logger.info(f"使用设备: {self.device}")
    
    def evaluate(self, test_loader):
        """评估模型"""
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        self.logger.info("开始评估...")
        
        with torch.no_grad():
            for faces, landmarks, labels in tqdm(test_loader, desc="评估"):
                faces = faces.to(self.device)
                landmarks = landmarks.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(faces, landmarks)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_labels), np.array(all_predictions), np.array(all_probabilities)
    
    def compute_metrics(self, y_true, y_pred, y_prob):
        """计算评估指标"""
        # 基本指标
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        # 加权平均指标
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC曲线和AUC
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'confusion_matrix': cm,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc
        }
        
        return metrics
    
    def print_metrics(self, metrics):
        """打印评估指标"""
        print("=== 模型评估结果 ===")
        print(f"总体准确率: {metrics['accuracy']:.4f}")
        print(f"加权精确率: {metrics['precision_weighted']:.4f}")
        print(f"加权召回率: {metrics['recall_weighted']:.4f}")
        print(f"加权F1分数: {metrics['f1_weighted']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        
        print("\n=== 各类别详细指标 ===")
        class_names = ['正常/说话', '打哈欠']
        for i, class_name in enumerate(class_names):
            print(f"{class_name}:")
            print(f"  精确率: {metrics['precision'][i]:.4f}")
            print(f"  召回率: {metrics['recall'][i]:.4f}")
            print(f"  F1分数: {metrics['f1'][i]:.4f}")
            print(f"  样本数: {metrics['support'][i]}")
        
        print("\n=== 混淆矩阵 ===")
        print(metrics['confusion_matrix'])
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['正常/说话', '打哈欠'],
                   yticklabels=['正常/说话', '打哈欠'])
        plt.title('测试集混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵已保存: {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, fpr, tpr, roc_auc, save_path=None):
        """绘制ROC曲线"""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC曲线 (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率 (False Positive Rate)')
        plt.ylabel('真正率 (True Positive Rate)')
        plt.title('ROC曲线')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC曲线已保存: {save_path}")
        
        plt.show()
    
    def analyze_errors(self, y_true, y_pred, test_loader):
        """分析错误预测"""
        errors = []
        correct = []
        
        # 找出错误预测的样本
        for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
            if true_label != pred_label:
                errors.append({
                    'index': i,
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'error_type': 'FP' if pred_label == 1 else 'FN'
                })
            else:
                correct.append(i)
        
        print(f"\n=== 错误分析 ===")
        print(f"总样本数: {len(y_true)}")
        print(f"正确预测: {len(correct)} ({len(correct)/len(y_true)*100:.1f}%)")
        print(f"错误预测: {len(errors)} ({len(errors)/len(y_true)*100:.1f}%)")
        
        # 分析错误类型
        fp_count = sum(1 for e in errors if e['error_type'] == 'FP')
        fn_count = sum(1 for e in errors if e['error_type'] == 'FN')
        
        print(f"假正例 (FP): {fp_count} - 将正常/说话误判为打哈欠")
        print(f"假负例 (FN): {fn_count} - 将打哈欠误判为正常/说话")
        
        return errors, correct

def main():
    """主函数"""
    # 检查模型文件
    model_path = os.path.join(MODEL_SAVE_PATH, 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先运行 train.py 进行模型训练")
        return
    
    # 加载数据
    data_path = os.path.join(PROCESSED_DATA_PATH, "processed_samples.pkl")
    if not os.path.exists(data_path):
        print(f"预处理数据文件不存在: {data_path}")
        print("请先运行 data_preprocessing.py 进行数据预处理")
        return
    
    print("加载数据...")
    samples = load_processed_data(data_path)
    
    # 创建数据加载器
    _, _, test_loader = create_data_loaders(samples)
    
    # 创建评估器
    evaluator = ModelEvaluator(model_path)
    
    # 评估模型
    y_true, y_pred, y_prob = evaluator.evaluate(test_loader)
    
    # 计算指标
    metrics = evaluator.compute_metrics(y_true, y_pred, y_prob)
    
    # 打印结果
    evaluator.print_metrics(metrics)
    
    # 绘制图表
    cm_path = os.path.join(OUTPUT_DIR, 'test_confusion_matrix.png')
    evaluator.plot_confusion_matrix(metrics['confusion_matrix'], cm_path)
    
    roc_path = os.path.join(OUTPUT_DIR, 'test_roc_curve.png')
    evaluator.plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['roc_auc'], roc_path)
    
    # 错误分析
    errors, correct = evaluator.analyze_errors(y_true, y_pred, test_loader)
    
    print("\n评估完成!")

if __name__ == "__main__":
    main()
