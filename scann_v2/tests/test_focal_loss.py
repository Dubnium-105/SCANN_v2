"""测试 FocalLoss 修复后的功能"""
import sys
sys.path.insert(0, 'g:/wksp/aikt/scann_v2/src')

from scann.ai.trainer import FocalLoss
import torch

# 测试1: 创建 FocalLoss 对象
print("测试1: 创建 FocalLoss 对象")
loss_fn = FocalLoss(gamma=2.0, alpha=[1.0, 1.5])
print("  FocalLoss创建成功")

# 测试2: 检查是否有 to 方法
print("\n测试2: 检查是否有 to 方法")
print(f"  是否有to方法: {hasattr(loss_fn, 'to')}")
print(f"  是否是nn.Module子类: {isinstance(loss_fn, torch.nn.Module)}")

# 测试3: 调用 to(device)
print("\n测试3: 调用 to(device)")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"  使用设备: {device}")
loss_fn = loss_fn.to(device)
print("  to(device)调用成功")

# 测试4: 计算 loss
print("\n测试4: 计算 loss")
logits = torch.randn(4, 2)
targets = torch.tensor([0, 1, 0, 1])
loss = loss_fn(logits, targets)
print(f"  Loss计算成功: {loss.item():.4f}")

# 测试5: 使用不同的 alpha 格式
print("\n测试5: 使用不同的 alpha 格式")
loss_fn_scalar = FocalLoss(gamma=2.0, alpha=1.5)
loss_fn_tensor = FocalLoss(gamma=2.0, alpha=torch.tensor([1.0, 2.0]))
loss_fn_none = FocalLoss(gamma=2.0, alpha=None)

loss1 = loss_fn_scalar(logits, targets)
loss2 = loss_fn_tensor(logits, targets)
loss3 = loss_fn_none(logits, targets)

print(f"  alpha=1.5 (scalar): {loss1.item():.4f}")
print(f"  alpha=[1.0, 2.0] (tensor): {loss2.item():.4f}")
print(f"  alpha=None: {loss3.item():.4f}")

print("\n所有测试通过!")
