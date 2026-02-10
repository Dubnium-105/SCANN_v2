"""测试修复后的路径计算"""
from pathlib import Path

# 测试 model.py
model_file = Path("g:/wksp/aikt/scann_v2/src/scann/ai/model.py").resolve()
scann_v2_root_model = model_file.parent.parent.parent.parent  # 4级向上
model_cache_dir_model = scann_v2_root_model / "models" / "torch_cache"
print(f"model.py 计算的缓存目录: {model_cache_dir_model}")

# 测试 training_worker.py
worker_file = Path("g:/wksp/aikt/scann_v2/src/scann/ai/training_worker.py").resolve()
scann_v2_root_worker = worker_file.parent.parent.parent.parent  # 4级向上
model_cache_dir_worker = scann_v2_root_worker / "models" / "torch_cache"
print(f"training_worker.py 计算的缓存目录: {model_cache_dir_worker}")

# 测试 app.py
app_file = Path("g:/wksp/aikt/scann_v2/src/scann/app.py").resolve()
project_root_app = app_file.parent.parent.parent  # 3级向上
model_cache_dir_app = project_root_app / "models" / "torch_cache"
print(f"app.py 计算的缓存目录: {model_cache_dir_app}")

# 验证三者是否一致
if model_cache_dir_model == model_cache_dir_worker == model_cache_dir_app:
    print(f"\n✓ 所有路径一致: {model_cache_dir_model}")
else:
    print(f"\n✗ 路径不一致!")
    print(f"  model.py: {model_cache_dir_model}")
    print(f"  training_worker.py: {model_cache_dir_worker}")
    print(f"  app.py: {model_cache_dir_app}")
