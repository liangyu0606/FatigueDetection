"""
训练脚本 - 打哈欠检测模型
"""
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
from datetime import datetime

from config import *
from model import create_model, ImprovedFocalLoss, count_parameters
from dataset import load_processed_data, create_data_loaders, analyze_dataset
from utils import setup_logging

class YawnDetectionTrainer:
    def __init__(self, resume_from_checkpoint=False):
        self.logger = setup_logging()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"使用设备: {self.device}")

        # 创建模型
        self.model = create_model().to(self.device)
        self.logger.info(f"模型参数数量: {count_parameters(self.model):,}")

        # 改进的损失函数和优化器
        # 计算类别权重
        class_weights = torch.tensor([1.0, 3.0]).to(self.device)  # 给打哈欠类别更高权重
        self.criterion = ImprovedFocalLoss(alpha=class_weights, gamma=2, label_smoothing=0.1)

        # 使用AdamW优化器，更好的权重衰减
        self.optimizer = optim.AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

        # 改进的学习率调度
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=LEARNING_RATE * 3,
            epochs=NUM_EPOCHS,
            steps_per_epoch=1,  # 将在训练中更新
            pct_start=0.3,
            anneal_strategy='cos'
        )

        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        self.best_model_path = None
        self.start_epoch = 0

        # 断点训练
        if resume_from_checkpoint:
            self.load_checkpoint()

        # TensorBoard
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if resume_from_checkpoint:
            timestamp += "_resumed"
        self.writer = SummaryWriter(log_dir=os.path.join(LOG_PATH, f"run_{timestamp}"))
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc="训练")
        for batch_idx, (faces, landmarks, labels) in enumerate(pbar):
            faces = faces.to(self.device)
            landmarks = landmarks.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(faces, landmarks)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            # 更强的梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="验证")
            for faces, landmarks, labels in pbar:
                faces = faces.to(self.device)
                landmarks = landmarks.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(faces, landmarks)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # 计算详细指标
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        return avg_loss, accuracy, precision, recall, f1, all_labels, all_predictions
    
    def save_model(self, epoch, val_acc, is_best=False):
        """保存模型（兼容性方法，调用save_checkpoint）"""
        self.save_checkpoint(epoch, val_acc, is_best)

    def load_checkpoint(self, checkpoint_path=None):
        """加载检查点进行断点训练"""
        if checkpoint_path is None:
            # 默认加载最新的检查点
            checkpoint_path = os.path.join(MODEL_SAVE_PATH, 'latest_model.pth')

        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"检查点文件不存在: {checkpoint_path}")
            return False

        try:
            self.logger.info(f"加载检查点: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # 加载模型状态
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # 加载学习率调度器状态（如果存在）
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # 加载训练记录
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_acc = checkpoint.get('val_acc', 0.0)
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.train_accuracies = checkpoint.get('train_accuracies', [])
            self.val_accuracies = checkpoint.get('val_accuracies', [])

            self.logger.info(f"成功加载检查点，从第 {self.start_epoch} 轮开始训练")
            self.logger.info(f"当前最佳验证准确率: {self.best_val_acc:.4f}")
            return True

        except Exception as e:
            self.logger.error(f"加载检查点失败: {e}")
            return False

    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """保存检查点（改进版保存方法）"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'learning_rate': LEARNING_RATE,
                'batch_size': BATCH_SIZE,
                'sequence_length': SEQUENCE_LENGTH,
                'num_epochs': NUM_EPOCHS
            }
        }

        # 保存最新模型
        latest_path = os.path.join(MODEL_SAVE_PATH, 'latest_model.pth')
        torch.save(checkpoint, latest_path)
        self.logger.info(f"保存检查点: {latest_path}")

        # 保存最佳模型
        if is_best:
            best_path = os.path.join(MODEL_SAVE_PATH, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.best_model_path = best_path
            self.logger.info(f"保存最佳模型: {best_path}, 验证准确率: {val_acc:.4f}")

        # 定期保存带时间戳的备份
        if epoch % 10 == 0:  # 每10轮保存一次备份
            backup_path = os.path.join(MODEL_SAVE_PATH, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, backup_path)
            self.logger.info(f"保存备份检查点: {backup_path}")
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(self.train_losses, label='训练损失')
        ax1.plot(self.val_losses, label='验证损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('训练和验证损失')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(self.train_accuracies, label='训练准确率')
        ax2.plot(self.val_accuracies, label='验证准确率')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('训练和验证准确率')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, epoch):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['正常/说话', '打哈欠'],
                   yticklabels=['正常/说话', '打哈欠'])
        plt.title(f'混淆矩阵 - Epoch {epoch}')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.savefig(os.path.join(OUTPUT_DIR, f'confusion_matrix_epoch_{epoch}.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def train(self, train_loader, val_loader, num_epochs=None):
        if num_epochs is None:
            num_epochs = DEBUG_EPOCHS if DEBUG_MODE else NUM_EPOCHS

        # 如果是断点训练，需要重新创建学习率调度器
        if self.start_epoch > 0:
            self.logger.info(f"断点训练：从第 {self.start_epoch} 轮开始，总共 {num_epochs} 轮")
            # 重新创建学习率调度器，考虑已经训练的轮次
            remaining_epochs = num_epochs - self.start_epoch
            if remaining_epochs <= 0:
                self.logger.warning("已达到或超过目标训练轮次")
                return

            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=LEARNING_RATE * 3,
                epochs=remaining_epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.3,
                anneal_strategy='cos'
            )
        else:
            # 正常训练，更新学习率调度器的steps_per_epoch
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=LEARNING_RATE * 3,
                epochs=num_epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.3,
                anneal_strategy='cos'
            )
        """完整训练流程"""
        if self.start_epoch > 0:
            self.logger.info(f"继续训练... 从第 {self.start_epoch} 轮开始")
        else:
            self.logger.info("开始训练...")
        start_time = time.time()

        for epoch in range(self.start_epoch, num_epochs):
            epoch_start_time = time.time()

            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)

            # 验证
            val_loss, val_acc, val_precision, val_recall, val_f1, y_true, y_pred = self.validate_epoch(val_loader)

            # 学习率调度（OneCycleLR每个epoch调用一次）
            self.scheduler.step()

            # 记录
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            # TensorBoard记录
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Val', val_acc, epoch)
            self.writer.add_scalar('Precision/Val', val_precision, epoch)
            self.writer.add_scalar('Recall/Val', val_recall, epoch)
            self.writer.add_scalar('F1/Val', val_f1, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)

            # 保存模型
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc

            self.save_model(epoch, val_acc, is_best)

            # 每10个epoch绘制混淆矩阵
            if (epoch + 1) % 10 == 0:
                self.plot_confusion_matrix(y_true, y_pred, epoch + 1)

            # 打印进度
            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f"Epoch [{epoch+1}/{num_epochs}] - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"Val F1: {val_f1:.4f}, Time: {epoch_time:.2f}s"
            )

        # 训练完成
        total_time = time.time() - start_time
        self.logger.info(f"训练完成! 总时间: {total_time/3600:.2f}小时")
        self.logger.info(f"最佳验证准确率: {self.best_val_acc:.4f}")

        # 绘制训练曲线
        self.plot_training_curves()

        # 关闭TensorBoard
        self.writer.close()

        return self.best_model_path

def main(resume_training=False):
    """主函数"""
    if resume_training:
        print("🔄 继续训练疲劳检测模型")
    else:
        print("🚀 开始训练疲劳检测模型")
    print("="*50)

    # 按优先级查找数据文件
    data_files = [
        os.path.join(PROCESSED_DATA_PATH, "balanced_debug_samples.pkl"),  # 平衡调试数据
        os.path.join(PROCESSED_DATA_PATH, "debug_samples.pkl"),           # 调试数据
        os.path.join(PROCESSED_DATA_PATH, "processed_samples.pkl")        # 正式数据
    ]

    data_path = None
    for file_path in data_files:
        if os.path.exists(file_path):
            data_path = file_path
            print(f"找到数据文件: {data_path}")
            break

    if data_path is None:
        print("❌ 没有找到任何预处理数据文件！")
        print("请先运行以下命令之一:")
        print("  python main.py --mode preprocess --debug  # 生成调试数据")
        print("  python analyze_data.py                     # 生成平衡数据")
        print("  python main.py --mode preprocess           # 生成完整数据")
        return

    print("加载预处理数据...")
    samples = load_processed_data(data_path)

    # 分析数据集
    analyze_dataset(samples)

    # 创建数据加载器
    print("创建数据加载器...")
    train_loader, val_loader, test_loader = create_data_loaders(samples)

    # 创建训练器
    trainer = YawnDetectionTrainer(resume_from_checkpoint=resume_training)

    # 开始训练
    best_model_path = trainer.train(train_loader, val_loader)

    print(f"训练完成! 最佳模型保存在: {best_model_path}")

if __name__ == "__main__":
    main()
