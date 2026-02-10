"""检查PyTorch环境变量"""
import os
import sys
from pathlib import Path

# 设置PyTorch模型下载路径到项目内
script_path = Path(__file__).resolve()
project_root = script_path.parent / "scann_v2"
model_cache_dir = project_root / "models" / "torch_cache"
model_cache_dir.mkdir(parents=True, exist_ok=True)

# 设置所有可能的环境变量
env_vars = {
    'TORCH_HOME': str(model_cache_dir),
    'TORCH_HUB_DIR': str(model_cache_dir),
    'XDG_CACHE_HOME': str(model_cache_dir.parent),
    'HUB_DIR': str(model_cache_dir),
}
for key, value in env_vars.items():
    os.environ[key] = value
    print(f"设置 {key} = {value}")

print(f"\n环境变量设置完成。")
print(f"目标缓存目录: {model_cache_dir}")

# 现在导入torch
import torch

print(f"\n导入torch后检查:")
print(f"torch.hub.get_dir() = {torch.hub.get_dir()}")
print(f"torch.hub.get_dir() == model_cache_dir = {torch.hub.get_dir() == str(model_cache_dir)}")

# 检查所有相关环境变量
print(f"\n检查环境变量:")
print(f"TORCH_HOME = {os.environ.get('TORCH_HOME', 'Not set')}")
print(f"TORCH_HUB_DIR = {os.environ.get('TORCH_HUB_DIR', 'Not set')}")
print(f"XDG_CACHE_HOME = {os.environ.get('XDG_CACHE_HOME', 'Not set')}")
print(f"HUB_DIR = {os.environ.get('HUB_DIR', 'Not set')}")
