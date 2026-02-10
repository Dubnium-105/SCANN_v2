"""检查model.py中路径计算是否正确"""
from pathlib import Path

# 模拟model.py中的路径计算
model_file = Path("g:/wksp/aikt/scann_v2/src/scann/ai/model.py").resolve()

print(f"model.py路径: {model_file}")
print(f".parent = {model_file.parent}")
print(f".parent.parent = {model_file.parent.parent}")
print(f".parent.parent.parent = {model_file.parent.parent.parent}")
print(f".parent.parent.parent.parent = {model_file.parent.parent.parent.parent}")

scann_v2_root = model_file.parent.parent.parent
print(f"\n当前计算的scann_v2_root (3级向上): {scann_v2_root}")

scann_v2_root_v4 = model_file.parent.parent.parent.parent
print(f"正确应该是 (4级向上): {scann_v2_root_v4}")

model_cache_dir_wrong = scann_v2_root / "models" / "torch_cache"
print(f"\n错误的缓存目录路径 (3级): {model_cache_dir_wrong}")

model_cache_dir_correct = scann_v2_root_v4 / "models" / "torch_cache"
print(f"正确的缓存目录路径 (4级): {model_cache_dir_correct}")
