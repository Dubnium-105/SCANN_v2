"""测试训练流程修复"""
import sys
sys.path.insert(0, 'g:/wksp/aikt/scann_v2/src')

from scann.ai.trainer import FocalLoss
import torch
import torch.nn as nn

print("测试训练流程中的 FocalLoss 初始化...")

# 模拟 training_worker.py 中的代码
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 测试创建 FocalLoss（这是 training_worker.py 中的用法）
criterion = FocalLoss(gamma=2.0, alpha=[1.0, 1.5]).to(device)
print("FocalLoss 创建并调用 to(device) 成功")

# 测试前向传播
model = nn.Linear(10, 2).to(device)
logits = model(torch.randn(4, 10).to(device))
targets = torch.tensor([0, 1, 0, 1]).to(device)

loss = criterion(logits, targets)
print(f"Loss 计算成功: {loss.item():.4f}")

# 反向传播测试
loss.backward()
print("反向传播成功")

print("\n训练流程测试通过!")
