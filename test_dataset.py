"""测试数据集加载"""
import os
import sys

# 添加路径
sys.path.insert(0, 'g:/wksp/aikt/scann_v2/src')

from scann.ai.dataset import TripletPNGDataset
import numpy as np

# 测试路径
pos_dir = "g:/wksp/aikt/dataset/positive"
neg_dir = "g:/wksp/aikt/dataset/negative"

# 收集样本
all_samples = []
for dir_path, label in [(pos_dir, 1), (neg_dir, 0)]:
    if not os.path.isdir(dir_path):
        print(f"目录不存在: {dir_path}")
        continue
    count = 0
    for fn in os.listdir(dir_path):
        if fn.lower().endswith((".png", ".fts", ".fit")):
            all_samples.append((os.path.join(dir_path, fn), label))
            count += 1
    print(f"{dir_path}: {count} 个样本")

print(f"\n总计样本数: {len(all_samples)}")

# 划分数据集
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
print("\n创建训练数据集...")
train_set = TripletPNGDataset(
    samples=train_samples,
    split="train",
    resize=224,
    augment=False,  # 测试时不增强
)

print(f"训练数据集大小: {len(train_set)}")
print(f"训练样本列表长度: {len(train_samples)}")

# 测试访问第一个样本
print("\n测试访问第一个样本...")
try:
    x, y = train_set[0]
    print(f"成功访问第一个样本: x.shape={x.shape}, y={y}")
except IndexError as e:
    print(f"访问第一个样本失败: {e}")
    print(f"train_set.samples 长度: {len(train_set.samples)}")
    if len(train_set.samples) > 0:
        print(f"第一个样本路径: {train_set.samples[0]}")

# 测试访问最后一个样本
print("\n测试访问最后一个样本...")
try:
    x, y = train_set[len(train_set) - 1]
    print(f"成功访问最后一个样本: x.shape={x.shape}, y={y}")
except IndexError as e:
    print(f"访问最后一个样本失败: {e}")

# 测试访问一些随机样本
print("\n测试访问一些随机样本...")
test_indices = [0, len(train_set) // 2, len(train_set) - 1]
for i in test_indices:
    try:
        x, y = train_set[i]
        print(f"成功访问样本 {i}: x.shape={x.shape}, y={y}")
    except IndexError as e:
        print(f"访问样本 {i} 失败: {e}")

print("\n测试完成!")
