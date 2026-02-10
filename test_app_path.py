"""检查app.py中路径计算是否正确"""
from pathlib import Path

# 模拟app.py中的路径计算
app_file = Path("g:/wksp/aikt/scann_v2/src/scann/app.py").resolve()

print(f"app.py路径: {app_file}")
print(f".parent = {app_file.parent}")
print(f".parent.parent = {app_file.parent.parent}")
print(f".parent.parent.parent = {app_file.parent.parent.parent}")

src_path = app_file.parent.parent
print(f"\n当前计算的src_path (2级向上): {src_path}")

project_root_correct = app_file.parent.parent.parent
print(f"正确的项目根目录 (3级向上): {project_root_correct}")

model_cache_dir_wrong = src_path / "models" / "torch_cache"
print(f"\n错误的缓存目录路径 (2级): {model_cache_dir_wrong}")

model_cache_dir_correct = project_root_correct / "models" / "torch_cache"
print(f"正确的缓存目录路径 (3级): {model_cache_dir_correct}")
