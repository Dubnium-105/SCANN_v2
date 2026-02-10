"""测试训练流程"""
import os
import sys
import numpy as np
from pathlib import Path

# 设置PyTorch模型下载路径到项目内（必须在导入torch之前设置）
# 获取脚本所在目录的父目录（即scann_v2根目录）
script_path = Path(__file__).resolve()
project_root = script_path.parent / "scann_v2"
model_cache_dir = project_root / "models" / "torch_cache"
model_cache_dir.mkdir(parents=True, exist_ok=True)

# 设置环境变量
os.environ['TORCH_HOME'] = str(model_cache_dir)
os.environ['TORCH_HUB_DIR'] = str(model_cache_dir)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

# 确认缓存目录
torch.hub.set_dir(str(model_cache_dir))
print(f"PyTorch模型缓存目录: {model_cache_dir}")
print(f"实际TORCH_HOME: {torch.hub.get_dir()}")

# 添加路径
sys.path.insert(0, 'g:/wksp/aikt/scann_v2/src')

from scann.ai.dataset import TripletPNGDataset
from torchvision import models, transforms

# 测试路径
pos_dir = "g:/wksp/aikt/dataset/positive"
neg_dir = "g:/wksp/aikt/dataset/negative"

# 训练参数
epochs = 2  # 只测试2个epoch
batch_size = 16
lr = 0.001

print("=== 1. 数据集加载 ===")

# 收集样本
all_samples = []
for dir_path, label in [(pos_dir, 1), (neg_dir, 0)]:
    if not os.path.isdir(dir_path):
        raise ValueError(f"目录不存在: {dir_path}")
    for fn in os.listdir(dir_path):
        if fn.lower().endswith((".png", ".fts", ".fit")):
            all_samples.append((os.path.join(dir_path, fn), label))

print(f"总计样本数: {len(all_samples)}")

# 划分训练集/验证集
n = len(all_samples)
idx = np.arange(n)
np.random.shuffle(idx)
split = int(0.8 * n)
train_idx = idx[:split].tolist()
val_idx = idx[split:].tolist()

train_samples = [all_samples[i] for i in train_idx]
val_samples = [all_samples[i] for i in val_idx]

print(f"训练集: {len(train_samples)}, 验证集: {len(val_samples)}")

# 创建数据集
train_set = TripletPNGDataset(
    samples=train_samples,
    split="train",
    resize=224,
    augment=True,
)
val_set = TripletPNGDataset(
    samples=val_samples,
    split="val",
    resize=224,
    augment=False,
)

print(f"训练数据集大小: {len(train_set)}")
print(f"验证数据集大小: {len(val_set)}")

# 类别平衡采样
train_labels = [all_samples[i][1] for i in train_idx]
count_neg = train_labels.count(0)
count_pos = train_labels.count(1)
print(f"训练集分布: 负样本={count_neg}, 正样本={count_pos}")

weight_class = [1.0 / max(count_neg, 1), 1.0 / max(count_pos, 1)]
samples_weight = [weight_class[y] for y in train_labels]
samples_weight = torch.tensor(samples_weight, dtype=torch.double)

sampler = WeightedRandomSampler(
    samples_weight, num_samples=len(train_set), replacement=True
)

train_loader = DataLoader(
    train_set, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=False
)
val_loader = DataLoader(
    val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
)

print(f"训练批次数: {len(train_loader)}")
print(f"验证批次数: {len(val_loader)}")

print("\n=== 2. 模型创建 ===")

# 创建模型
backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
backbone.fc = nn.Linear(backbone.fc.in_features, 1)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

model = backbone.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

print("\n=== 3. 训练循环 ===")

for epoch in range(epochs):
    # 训练
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    print(f"\nEpoch {epoch + 1}/{epochs} - 训练中...")

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.float().unsqueeze(1).to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        train_loss += loss.item()
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        train_total += targets.size(0)
        train_correct += (predicted == targets).sum().item()

        if batch_idx % 20 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    train_loss /= len(train_loader)
    train_acc = 100.0 * train_correct / train_total

    # 验证
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    print(f"验证中...")

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.float().unsqueeze(1).to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            val_total += targets.size(0)
            val_correct += (predicted == targets).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100.0 * val_correct / val_total

    print(f"\nEpoch {epoch + 1} 结果:")
    print(f"  训练集 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
    print(f"  验证集 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

print("\n=== 测试完成 ===")
