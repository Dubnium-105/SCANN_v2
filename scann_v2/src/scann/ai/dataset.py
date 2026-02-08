"""AI 训练数据集"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


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


class FitsDetectionDataset:
    """FITS 全图检测数据集 (v2 新格式)

    TODO: 实现 FITS 格式的标注数据集
    """
    pass
