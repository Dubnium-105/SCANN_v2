"""AI 训练数据集"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from torchvision import transforms


class TripletDataset:
    """三联图数据集 (兼容 v1 PNG 格式)

    读取 80x240 PNG 三联图，切分为三通道输入。
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        indices: Optional[List[int]] = None,
        channel_order: Tuple[int, int, int] = (0, 1, 2),
        resize: int = 224,
        mean: Tuple[float, ...] = (0.264, 0.282, 0.284),
        std: Tuple[float, ...] = (0.089, 0.123, 0.128),
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.channel_order = channel_order
        self.resize = resize
        self.mean = mean
        self.std = std

        # 收集样本
        self.samples: List[Tuple[str, int]] = []
        for label_name, y in [("negative", 0), ("positive", 1)]:
            folder = self.root_dir / label_name
            if not folder.is_dir():
                continue
            for fn in sorted(folder.iterdir()):
                if fn.suffix.lower() == ".png":
                    self.samples.append((str(fn), y))

        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        parts = self._read_triplet(path)
        # 后续处理由 PyTorch DataLoader 端完成
        return parts, y

    def _read_triplet(self, path: str) -> List[np.ndarray]:
        """读取三联图并切分"""
        from PIL import Image

        im = Image.open(path).convert("L")
        w, h = im.size
        if w < 240 or h < 80:
            raise ValueError(f"尺寸不符: {w}x{h} for {path}")

        parts = [
            np.array(im.crop((0, 0, 80, 80)), dtype=np.float32) / 255.0,
            np.array(im.crop((80, 0, 160, 80)), dtype=np.float32) / 255.0,
            np.array(im.crop((160, 0, 240, 80)), dtype=np.float32) / 255.0,
        ]
        return [parts[i] for i in self.channel_order]

    def get_label_counts(self) -> dict:
        """统计各类别数量"""
        counts = {0: 0, 1: 0}
        for _, y in self.samples:
            counts[y] = counts.get(y, 0) + 1
        return counts


class TripletPNGDataset:
    """三联图 PyTorch 数据集 (支持 transform 和 tensor 返回)

    与 TripletDataset 类似，但返回 PyTorch tensor 格式，
    并支持数据增强和归一化。
    """

    def __init__(
        self,
        root_dir: str = "",  # 仅用于兼容，实际使用 samples
        split: str = "train",
        indices: Optional[List[int]] = None,
        channel_order: Tuple[int, int, int] = (0, 1, 2),
        resize: int = 224,
        mean: Tuple[float, ...] = (0.264, 0.282, 0.284),
        std: Tuple[float, ...] = (0.089, 0.123, 0.128),
        augment: bool = True,
        samples: Optional[List[Tuple[str, int]]] = None,  # 新增：支持传入预构建的样本列表
    ):
        self.root_dir = Path(root_dir) if root_dir else Path("")
        self.split = split
        self.channel_order = channel_order
        self.resize = resize
        self.mean = mean
        self.std = std
        self.augment = augment and (split == "train")

        # 收集样本（优先使用传入的 samples）
        self.samples: List[Tuple[str, int]] = []
        if samples is not None:
            self.samples = samples
        else:
            # 兼容旧逻辑：从目录收集
            for label_name, y in [("negative", 0), ("positive", 1)]:
                folder = self.root_dir / label_name
                if not folder.is_dir():
                    continue
                for fn in sorted(folder.iterdir()):
                    if fn.suffix.lower() == ".png":
                        self.samples.append((str(fn), y))

        if indices is not None:
            # 应用索引筛选
            self.samples = [self.samples[i] for i in indices]

        # 基础 transform
        self.base_transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
        ])
        self.normalize = transforms.Normalize(list(mean), list(std))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import torch
        from torchvision.transforms import functional as TF

        path, y = self.samples[idx]
        parts = self._read_triplet_images(path)

        # 转换为 tensor
        tensors = [self.base_transform(p) for p in parts]
        x = torch.cat(tensors, dim=0)  # [3, H, W]

        # 数据增强
        if self.augment:
            if random.random() < 0.5:
                x = TF.hflip(x)
            if random.random() < 0.5:
                x = TF.vflip(x)
            k = random.randint(0, 3)
            if k > 0:
                x = torch.rot90(x, k, dims=[1, 2])

        # 归一化
        x = self.normalize(x)

        return x, torch.tensor(y, dtype=torch.long)

    def _read_triplet_images(self, path: str) -> List:
        """读取三联图并切分（返回 PIL Image）"""
        from PIL import Image

        im = Image.open(path).convert("L")
        w, h = im.size
        if w < 240 or h < 80:
            raise ValueError(f"尺寸不符: {w}x{h} for {path}")

        parts = [
            im.crop((0, 0, 80, 80)),
            im.crop((80, 0, 160, 80)),
            im.crop((160, 0, 240, 80)),
        ]
        return [parts[i] for i in self.channel_order]

    def get_label_counts(self) -> dict:
        """统计各类别数量"""
        counts = {0: 0, 1: 0}
        for _, y in self.samples:
            counts[y] = counts.get(y, 0) + 1
        return counts


class FitsDetectionDataset:
    """FITS 全图检测数据集 (v2 新格式)

    支持从 FITS 图像中提取滑动窗口 patches 用于目标检测训练。
    """

    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        patch_size: int = 224,
        stride: int = 112,
        label_map: Optional[dict] = None,
    ):
        """
        Args:
            image_dir: FITS 图像目录
            annotation_file: JSON 格式的标注文件
            patch_size: 提取的 patch 大小
            stride: 滑动窗口步长
            label_map: 标签映射字典，如 {"real": 1, "bogus": 0}
        """
        self.image_dir = Path(image_dir)
        self.annotation_file = Path(annotation_file)
        self.patch_size = patch_size
        self.stride = stride
        self.label_map = label_map or {"real": 1, "bogus": 0}

        # 加载标注
        self.samples = self._load_annotations()

    def _load_annotations(self) -> list:
        """从 JSON 文件加载标注"""
        import json

        with open(self.annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples = []
        for img_info in data.get("images", []):
            img_path = self.image_dir / img_info["file"]
            if not img_path.exists():
                continue

            samples.append({
                "image": str(img_path),
                "width": img_info["width"],
                "height": img_info["height"],
                "annotations": img_info.get("annotations", []),
            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """获取指定索引的样本

        Returns:
            (patch, targets) tuple
            patch: (3, patch_size, patch_size) numpy array
            targets: list of [x_center, y_center, width, height, class_id]
        """
        from scann.core.fits_io import read_fits

        sample = self.samples[idx]

        # 读取 FITS 图像
        fits_image = read_fits(sample["image"])
        image = fits_image.data

        # 默认提取中心区域的 patch
        crop_box = self._get_center_crop_box(sample["width"], sample["height"])
        patch = self._extract_patch(image, crop_box)

        # 转换标注到 patch 坐标系
        targets = self._annotations_to_targets(sample["annotations"], crop_box)

        return patch, targets

    def get_crop(self, idx: int, x: int, y: int, size: int):
        """获取指定位置的 crop

        Args:
            idx: 图像索引
            x: 左上角 x 坐标
            y: 左上角 y 坐标
            size: crop 大小

        Returns:
            (patch, targets) tuple
        """
        from scann.core.fits_io import read_fits

        sample = self.samples[idx]
        fits_image = read_fits(sample["image"])
        image = fits_image.data

        crop_box = (x, y, x + size, y + size)
        patch = self._extract_patch(image, crop_box)
        targets = self._annotations_to_targets(sample["annotations"], crop_box)

        return patch, targets

    def iter_patches(self, idx: int):
        """迭代图像的所有滑动窗口 patches

        Args:
            idx: 图像索引

        Yields:
            (patch, targets) tuples
        """
        from scann.core.fits_io import read_fits

        sample = self.samples[idx]
        fits_image = read_fits(sample["image"])
        image = fits_image.data

        # 计算所有可能的 crop 位置
        for y in range(0, sample["height"] - self.patch_size + 1, self.stride):
            for x in range(0, sample["width"] - self.patch_size + 1, self.stride):
                crop_box = (x, y, x + self.patch_size, y + self.patch_size)
                patch = self._extract_patch(image, crop_box)
                targets = self._annotations_to_targets(sample["annotations"], crop_box)
                yield patch, targets

    def _get_center_crop_box(self, width: int, height: int) -> tuple:
        """获取中心区域的 crop box"""
        x0 = max(0, (width - self.patch_size) // 2)
        y0 = max(0, (height - self.patch_size) // 2)
        x1 = min(width, x0 + self.patch_size)
        y1 = min(height, y0 + self.patch_size)
        return (x0, y0, x1, y1)

    def _extract_patch(self, image: np.ndarray, crop_box: tuple) -> np.ndarray:
        """从图像中提取 patch 并归一化

        Args:
            image: (H, W) 输入图像
            crop_box: (x0, y0, x1, y1) crop 区域

        Returns:
            (3, patch_size, patch_size) 归一化的 patch
        """
        x0, y0, x1, y1 = crop_box

        # 裁剪
        patch = image[y0:y1, x0:x1].astype(np.float32)

        # 如果裁剪尺寸不足，使用 padding
        if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
            padded = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
            h, w = patch.shape
            padded[:h, :w] = patch
            patch = padded

        # 归一化到 0-1
        if patch.max() > patch.min():
            patch = (patch - patch.min()) / (patch.max() - patch.min())

        # 调整大小到目标尺寸
        if patch.shape != (self.patch_size, self.patch_size):
            from skimage.transform import resize
            patch = resize(
                patch,
                (self.patch_size, self.patch_size),
                order=1,
                preserve_range=True,
                anti_aliasing=False
            )
            patch = patch.astype(np.float32)

        # 扩展为三通道（如果需要多通道输入）
        # 这里简单复制为三通道
        patch_3ch = np.stack([patch, patch, patch], axis=0)

        return patch_3ch

    def _annotations_to_targets(
        self,
        annotations: list,
        crop_box: tuple,
    ) -> list:
        """将标注转换为训练目标格式

        Args:
            annotations: 原始标注列表
            crop_box: (x0, y0, x1, y1) crop 区域

        Returns:
            list of [x_center, y_center, width, height, class_id]
            坐标已归一化到 [0, 1]
        """
        x0, y0, x1, y1 = crop_box
        crop_width = x1 - x0
        crop_height = y1 - y0

        targets = []
        for ann in annotations:
            # 检查标注是否在 crop 区域内
            ann_x = ann["x"]
            ann_y = ann["y"]
            ann_w = ann["width"]
            ann_h = ann["height"]

            # 计算重叠
            # 简化处理：如果中心点在 crop 区域内
            center_x = ann_x + ann_w / 2
            center_y = ann_y + ann_h / 2

            if center_x < x0 or center_x > x1 or center_y < y0 or center_y > y1:
                continue  # 不在 crop 区域内

            # 转换到 crop 坐标系
            rel_x = (center_x - x0) / crop_width
            rel_y = (center_y - y0) / crop_height
            rel_w = ann_w / crop_width
            rel_h = ann_h / crop_height

            # 映射标签
            label = ann.get("label", "real")
            class_id = self.label_map.get(label, 0)

            targets.append([rel_x, rel_y, rel_w, rel_h, class_id])

        return targets

    def get_label_counts(self) -> dict:
        """统计各类别的标注数量"""
        counts = {}
        for sample in self.samples:
            for ann in sample["annotations"]:
                label = ann.get("label", "real")
                counts[label] = counts.get(label, 0) + 1
        return counts
