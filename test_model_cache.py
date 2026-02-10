"""测试PyTorch模型缓存目录设置"""
import sys
from pathlib import Path
import torch

# 设置PyTorch模型下载路径到项目内
project_root = Path('g:/wksp/aikt/scann_v2')
model_cache_dir = project_root / "models" / "torch_cache"
model_cache_dir.mkdir(parents=True, exist_ok=True)
torch.hub.set_dir(str(model_cache_dir))

print(f"PyTorch模型缓存目录: {model_cache_dir}")
print(f"TORCH_HOME: {torch.hub.get_dir()}")

# 测试创建模型（会触发下载）
print("\n测试加载ResNet18模型...")
try:
    from torchvision import models
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    print(f"成功加载模型！")
    print(f"模型已缓存到: {model_cache_dir}")
except Exception as e:
    print(f"加载模型失败: {e}")

# 检查下载的文件
print(f"\n检查缓存目录内容:")
if model_cache_dir.exists():
    for item in model_cache_dir.rglob("*"):
        print(f"  {item.relative_to(model_cache_dir)}")
else:
    print("  缓存目录不存在")
