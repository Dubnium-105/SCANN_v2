"""测试CUDA状态显示功能"""
import sys
sys.path.insert(0, 'g:/wksp/aikt/scann_v2/src')

from PyQt5.QtWidgets import QApplication
from scann.gui.dialogs.training_dialog import TrainingDialog

app = QApplication(sys.argv)

print("创建训练对话框...")
dialog = TrainingDialog()

print(f"CUDA状态标签内容:")
print(dialog.lbl_cuda_status.text())
print()

print(f"设备选项:")
for i in range(dialog.combo_device.count()):
    text = dialog.combo_device.itemText(i)
    data = dialog.combo_device.itemData(i)
    print(f"  [{i}] {text} -> {data}")

print()
print("测试刷新CUDA状态...")
dialog._refresh_cuda_status()

import time
time.sleep(0.5)  # 等待异步检查完成

print(f"\n刷新后的CUDA状态:")
print(dialog.lbl_cuda_status.text())

print("\n对话框创建成功!")
print("提示: 如果CUDA不可用，请检查:")
print("  1. 是否安装了NVIDIA显卡和驱动")
print("  2. 是否安装了CUDA版本的PyTorch")
print("  3. 运行: nvidia-smi 检查驱动")
print("  4. 运行: python -c 'import torch; print(torch.cuda.is_available())' 检查PyTorch")
