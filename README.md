# SCANN (Star/Source Classification and Analysis Neural Network)

这是一个用于天文图像（三联图）分类和数据管理的工具集。它包含了数据下载、模型训练、归一化参数计算以及一个基于 PyQt5 的图形用户界面。

## 项目结构

- `SCANN.py`: 主 GUI 应用程序，支持数据下载、联动查看和分类。
- `train_triplet_resnet_augmented.py`: 使用 ResNet-18 进行三联图分类的模型训练脚本，支持数据增强和加权采样。
- `calc_triplet_mean_std.py`: 计算数据集均值和标准差的工具脚本，用于训练时的归一化。
- `dataset/`: 存放训练数据的目录。
  - `positive/`: 正样本 PNG 图像。
  - `negative/`: 负样本 PNG 图像。
- `best_model.pth`: 训练好的模型权重文件。
- `requirements.txt`: 项目依赖包列表。

## 安装指南

首先，请确保已安装 Python 3.7+。然后，使用以下命令安装依赖：

```bash
pip install -r requirements.txt
```

## 脚本使用方法

### 1. 主 GUI 程序 (`SCANN.py`)

这是项目的主要交互界面，集成了数据下载和可视化功能。

- **运行方法**:
  ```bash
  python SCANN.py
  ```
- **功能**:
  - 自动从 NADC 下载 JPG 和 FITS 天文数据。
  - 管理本地数据库（SQLite）中的数据联动信息。
  - 提供图像查看器，支持缩放和分类操作。

### 2. 模型训练 (`train_triplet_resnet_augmented.py`)

用于训练 ResNet-18 分类器。它将 80x240 的三联图切分为三个 80x80 的通道（差异图、新图、参考图）。

- **运行方法**:
  ```bash
  python train_triplet_resnet_augmented.py --data dataset --epochs 30 --batch 32
  ```
- **主要参数**:
  - `--data`: 数据集根目录（默认为 `dataset`）。
  - `--epochs`: 训练轮数。
  - `--batch`: 批次大小。
  - `--lr`: 学习率。
  - `--target_recall`: 目标召回率（用于保存模型）。

### 3. 计算归一化参数 (`calc_triplet_mean_std.py`)

在训练新数据集之前，建议先计算数据集的均值和标准差。

- **运行方法**:
  ```bash
  python calc_triplet_mean_std.py --neg dataset/negative --pos dataset/positive
  ```
- **输出**: 脚本会输出 `mean` 和 `std`，你可以将其复制到训练脚本的 `Normalize` 转换中。

## 数据说明

项目处理的是 **三联图 (Triplet Images)**，尺寸通常为 80x240 像素。这些图被水平切分为三部分：
1. **左 (Diff)**: 差异图。
2. **中 (New)**: 新发现的图像。
3. **右 (Ref)**: 参考图像。

这些图像按指定的通道顺序（如 0, 1, 2）堆叠成 3 通道张量输入神经网络。
