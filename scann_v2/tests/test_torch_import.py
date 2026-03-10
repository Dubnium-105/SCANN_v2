"""检查torch是否已被导入"""
import os
import sys
from pathlib import Path

# 在任何导入之前检查
print("导入torch之前检查:")
print(f"sys.modules中是否有'torch': {'torch' in sys.modules}")
print(f"sys.modules中是否有'torchvision': {'torchvision' in sys.modules}")

# 设置PyTorch模型下载路径到项目内
script_path = Path(__file__).resolve()
project_root = script_path.parent / "scann_v2"
model_cache_dir = project_root / "models" / "torch_cache"
model_cache_dir.mkdir(parents=True, exist_ok=True)

# 设置所有可能的环境变量
os.environ['TORCH_HOME'] = str(model_cache_dir)
os.environ['TORCH_HUB_DIR'] = str(model_cache_dir)

print(f"\n设置环境变量:")
print(f"TORCH_HOME = {os.environ['TORCH_HOME']}")
print(f"TORCH_HUB_DIR = {os.environ['TORCH_HUB_DIR']}")

# 现在导入torch
import torch

print(f"\n导入torch后检查:")
print(f"torch.hub.get_dir() = {torch.hub.get_dir()}")

# 查看torch.hub的实现
print(f"\ntorch.hub module: {torch.hub}")
print(f"torch.hub.get_dir function: {torch.hub.get_dir}")
