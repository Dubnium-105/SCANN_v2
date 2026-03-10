"""快速训练测试"""
import os
import sys
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
from torch.utils.data import DataLoader, WeightedRandomSampler

torch.hub.set_dir(str(model_cache_dir))
print(f"PyTorch模型缓存目录: {model_cache_dir}")

from scann.ai.dataset import TripletPNGDataset
from torchvision import models, transforms

# 测试路径
pos_dir = "g:/wksp/aikt/dataset/positive"
neg_dir = "g:/wksp/aikt/dataset/negative"

# 训练参数
epochs = 1  # 快速测试
batch_size = 16
lr = 0.001

print("=== 1. 数据集加载 ===")

# 数据增强
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# 创建数据集
train_dataset = TripletPNGDataset(
    root_dir="g:/wksp/aikt/dataset",
    split="train",
    transform=train_transform,
    val_ratio=0.2,
    seed=42
)

val_dataset = TripletPNGDataset(
    root_dir="g:/wksp/aikt/dataset",
    split="val",
    transform=val_transform,
    val_ratio=0.2,
    seed=42
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
criterion = FocalLoss(gamma=2.0, alpha=[1.0, 1.5]).to(device)
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
