import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from data_preprocessing import YawDDPreprocessor

class FatigueDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor([self.labels[idx]])


class SimpleFatigueModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleFatigueModel, self).__init__()

        # 简化的LSTM模型
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        # LSTM处理
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size*2)

        # 注意力机制
        attention_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attended_output = torch.sum(lstm_out * attention_weights, dim=1)  # (batch_size, hidden_size*2)

        # 分类
        output = self.classifier(attended_output)
        return output


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=50):
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in tqdm(train_loader, desc=f"Train Epoch {epoch + 1}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()
            pred = (torch.sigmoid(outputs) > 0.5).float()
            train_total += targets.size(0)
            train_correct += (pred == targets).sum().item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Val Epoch {epoch + 1}"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                pred = (torch.sigmoid(outputs) > 0.5).float()
                val_total += targets.size(0)
                val_correct += (pred == targets).sum().item()

                all_outputs.extend(torch.sigmoid(outputs).cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # 计算指标
        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # 学习率调度
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # 打印详细训练进度
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Train - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val   - Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"  LR: {current_lr:.6f}")
        if all_outputs:
            all_outputs_array = np.array(all_outputs).flatten()
            print(f"  Output range: [{all_outputs_array.min():.3f}, {all_outputs_array.max():.3f}]")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "models/best_fatigue_model.pth")
            print(f"  *** New best model saved! Val Acc: {best_val_acc:.2f}% ***")
            patience_counter = 0
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            break

        print("-" * 60)

    return model

def save_dataset_by_fps(X, y, fps, output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    fps_int = int(round(fps))
    np.save(os.path.join(output_dir, f"X_{fps_int}.npy"), X)
    np.save(os.path.join(output_dir, f"y_{fps_int}.npy"), y)
    print(f"✅ 已保存帧率为 {fps} 的数据集到 {output_dir}/")

def load_dataset_by_fps(fps, data_dir="data"):
    fps_int = int(round(fps))
    x_path = os.path.join(data_dir, f"X_{fps_int}.npy")
    y_path = os.path.join(data_dir, f"y_{fps_int}.npy")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        return None, None

    X = np.load(x_path)
    y = np.load(y_path)
    print(f"✅ 加载了帧率为 {fps} 的数据集: X.shape={X.shape}, y.shape={y.shape}")
    return X, y


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    target_fps = 60  # 统一帧率设置
    data_dir = "F:\\College\\Pycharm_project\\PLJC\\dataset"
    predictor_path = "shape_predictor_68_face_landmarks.dat"

    # 尝试加载已有的目标帧率数据
    X, y = load_dataset_by_fps(target_fps)
    preprocessor = YawDDPreprocessor(predictor_path, seq_length=target_fps)

    if X is not None:
        print(f"📊 使用已存在的帧率 {target_fps} 的数据集进行训练")
    else:
        print(f"🔄 没有找到帧率 {target_fps} 的数据，开始预处理...")

        # 初始化预处理器并加载原始数据
        print("Processing dataset...")
        X, y = preprocessor.process_dataset(data_dir)

        print(f"Dataset info:")
        print(f"  Total samples: {len(X)}")
        print(f"  Feature shape: {X.shape}")
        print(f"  Label distribution: {np.bincount(y)}")

        # 保存为指定帧率的格式
        save_dataset_by_fps(X, y, target_fps)

        print(f"💾 帧率 {target_fps} 的数据已成功生成并加载")

    print(f"Dataset info:")
    print(f"  Total samples: {len(X)}")
    print(f"  Feature shape: {X.shape}")
    print(f"  Label distribution: {np.bincount(y)}")

    print("Splitting and scaling data...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_and_scale_data(X, y)

    print(f"Split info:")
    print(f"  Train: {len(X_train)} samples, {np.bincount(y_train)}")
    print(f"  Val: {len(X_val)} samples, {np.bincount(y_val)}")
    print(f"  Test: {len(X_test)} samples, {np.bincount(y_test)}")

    # 创建数据加载器
    train_dataset = FatigueDataset(X_train, y_train)
    val_dataset = FatigueDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # 减小batch size
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 初始化模型
    input_size = X_train.shape[2]  # 特征维度
    print(f"Model input size: {input_size}")

    model = SimpleFatigueModel(
        input_size=input_size,
        hidden_size=32,  # 更小的隐藏层
        output_size=1
    ).to(device)

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 定义损失函数和优化器
    # 使用加权损失来处理类别不平衡
    pos_weight = torch.tensor([len(y_train) / (2 * np.sum(y_train))]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 注意：SimpleFatigueModel的classifier最后一层已经是Linear(32, output_size)
    # BCEWithLogitsLoss内部包含Sigmoid，所以不需要额外修改

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # 训练模型
    print("Starting model training...")
    os.makedirs("models", exist_ok=True)
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=100)

    # 保存最终模型
    torch.save(trained_model.state_dict(), "models/final_fatigue_model.pth")
    print("Model training completed and saved!")


if __name__ == "__main__":
    main()