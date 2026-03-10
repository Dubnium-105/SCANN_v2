"""快速训练测试（使用正确的参数）"""
import os
import sys
import random
import numpy as np
from pathlib import Path

sys.path.insert(0, 'g:/wksp/aikt/scann_v2/src')

# 设置PyTorch模型下载路径
script_path = Path(__file__).resolve()
project_root = script_path.parent / "scann_v2"
model_cache_dir = project_root / "models" / "torch_cache"
model_cache_dir.mkdir(parents=True, exist_ok=True)

os.environ['TORCH_HOME'] = str(model_cache_dir)
os.environ['TORCH_HUB_DIR'] = str(model_cache_dir)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

torch.hub.set_dir(str(model_cache_dir))
print(f"PyTorch模型缓存目录: {model_cache_dir}")

from scann.ai.dataset import TripletPNGDataset
from torchvision import models

# 训练参数
epochs = 1  # 快速测试
batch_size = 16
lr = 0.001

print("=== 1. 数据集加载 ===")

# 先收集所有样本
all_samples = []
root_path = Path("g:/wksp/aikt/dataset")
for label_name, y in [("negative", 0), ("positive", 1)]:
    folder = root_path / label_name
    if not folder.is_dir():
        continue
    for fn in sorted(folder.glob("*.png")):
        all_samples.append((str(fn), y))

print(f"总计样本数: {len(all_samples)}")

# 划分训练集和验证集
np.random.seed(42)
indices = np.arange(len(all_samples))
np.random.shuffle(indices)

val_ratio = 0.2
val_size = int(len(all_samples) * val_ratio)
train_idx = indices[val_size:]
val_idx = indices[:val_size]

train_samples = [all_samples[i] for i in train_idx]
val_samples = [all_samples[i] for i in val_idx]

print(f"训练集: {len(train_samples)}, 验证集: {len(val_samples)}")

# 创建数据集
train_dataset = TripletPNGDataset(
    samples=train_samples,
    split="train",
    augment=True,
)

val_dataset = TripletPNGDataset(
    samples=val_samples,
    split="val",
    augment=False,
)

print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# === 2. 模型创建 ===
print("\n=== 2. 模型创建 ===")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

from scann.ai.trainer import FocalLoss

# 使用 ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# 损失函数和优化器 - 测试修复后的 FocalLoss
print("\n创建 FocalLoss...")
criterion = FocalLoss(gamma=2.0, alpha=[1.0, 1.5]).to(device)
print("FocalLoss 创建成功")

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

print("模型和损失函数创建成功")

# === 3. 训练循环 ===
print("\n=== 3. 训练循环 ===")

model.train()
total_loss = 0
correct = 0
total = 0

for batch_idx, (images, labels) in enumerate(train_loader):
    images = images.to(device)
    labels = labels.to(device)

    # 前向传播
    outputs = model(images)
    loss = criterion(outputs, labels)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 统计
    total_loss += loss.item()
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

    if batch_idx % 20 == 0:
        print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    # 测试一个批次后就退出
    if batch_idx >= 5:
        break

avg_loss = total_loss / (batch_idx + 1)
acc = 100 * correct / total

print(f"\n训练结果:")
print(f"  Loss: {avg_loss:.4f}, Acc: {acc:.2f}%")

print("\n=== 测试完成 ===")
print("FocalLoss 修复成功，训练流程正常!")
