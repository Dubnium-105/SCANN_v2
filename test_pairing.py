"""测试时间戳前缀配对逻辑"""
from pathlib import Path
from scann.data.file_manager import match_new_old_pairs

# 使用实际数据集路径
dataset_dir = Path(r"c:\wksp\SCANNv2\SCANN_v2\dataset")
new_dir = dataset_dir / "new"
old_dir = dataset_dir / "old"

if not new_dir.exists() or not old_dir.exists():
    print(f"数据集目录不存在: {dataset_dir}")
    exit(1)

print(f"扫描新图目录: {new_dir}")
print(f"扫描旧图目录: {old_dir}")

pairs, only_new, only_old = match_new_old_pairs(str(new_dir), str(old_dir))

print(f"\n配对结果:")
print(f"  配对数量: {len(pairs)}")
print(f"  仅新图: {len(only_new)}")
print(f"  仅旧图: {len(only_old)}")

print(f"\n配对列表:")
for i, pair in enumerate(pairs[:10], 1):  # 只显示前10个
    print(f"  {i}. {pair.name}")
    print(f"     new: {pair.new_path.name}")
    print(f"     old: {pair.old_path.name}")

if only_new:
    print(f"\n仅新图 (前5个):")
    for name in only_new[:5]:
        print(f"  - {name}")

if only_old:
    print(f"\n仅旧图 (前5个):")
    for name in only_old[:5]:
        print(f"  - {name}")

# 验证特定文件
print("\n验证特定配对:")
target_new = "20221227T214251__SAC NGC 3813.fts"
target_old = "20260308T215954__SAC NGC 3813.fts"

new_found = False
old_found = False
matched = False

for pair in pairs:
    if target_new in pair.new_path.name:
        new_found = True
        print(f"  新图找到: {pair.new_path.name}")
    if target_old in pair.old_path.name:
        old_found = True
        print(f"  旧图找到: {pair.old_path.name}")
    if target_new in pair.new_path.name and target_old in pair.old_path.name:
        matched = True
        print(f"  ✓ 配对成功! 名称: {pair.name}")

if not new_found:
    print(f"  ✗ 新图未找到: {target_new}")
if not old_found:
    print(f"  ✗ 旧图未找到: {target_old}")
if not matched:
    print(f"  ✗ 未配对成功")
