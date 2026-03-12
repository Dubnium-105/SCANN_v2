"""图像配对服务。

职责:
- 扫描新旧图文件夹
- 执行图像配对
- 读取配对对应的 FITS 图像
- 提供配对图像路径解析边界
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import numpy as np

from scann.core.fits_io import read_fits
from scann.core.models import ImagePair
from scann.data.file_manager import FitsFileInfo, FitsImagePair, match_new_old_pairs, scan_fits_folder


class PairService:
    """封装图像配对相关的应用服务。

    当前提交仅建立服务边界，不持有 UI 或会话状态。
    """

    def __init__(
        self,
        scan_folder_fn: Callable[[str], list[FitsFileInfo]] = scan_fits_folder,
        match_pairs_fn: Callable[[str, str], tuple[list[FitsImagePair], list[str], list[str]]] = match_new_old_pairs,
        read_fits_fn: Callable[[str | Path], object] = read_fits,
    ) -> None:
        self._scan_folder = scan_folder_fn
        self._match_pairs = match_pairs_fn
        self._read_fits = read_fits_fn

    def scan_new_folder(self, folder: str | Path) -> list[FitsFileInfo]:
        """扫描新图文件夹。"""
        return self._scan_folder(str(folder))

    def scan_old_folder(self, folder: str | Path) -> list[FitsFileInfo]:
        """扫描旧图文件夹。"""
        return self._scan_folder(str(folder))

    def match_pairs(
        self,
        new_folder: str | Path,
        old_folder: str | Path,
    ) -> tuple[list[FitsImagePair], list[str], list[str]]:
        """匹配新旧图像配对。"""
        return self._match_pairs(str(new_folder), str(old_folder))

    def read_image(self, path: str | Path):
        """读取单张 FITS 图像。"""
        return self._read_fits(path)

    def aligned_artifact_paths(self, pair: FitsImagePair) -> tuple[Path, Path, Path, Path]:
        """返回配对图像的对齐裁剪产物路径。"""
        new_path = Path(pair.new_path)
        old_path = Path(pair.old_path)
        return (
            new_path.with_name(f"{new_path.stem}__aligned_crop.fts"),
            old_path.with_name(f"{old_path.stem}__aligned_crop.fts"),
            new_path.with_name(f"{new_path.stem}__aligned.marker"),
            old_path.with_name(f"{old_path.stem}__aligned.marker"),
        )

    def pair_has_aligned_artifacts(self, pair: FitsImagePair) -> bool:
        """配对是否已有可复用的对齐裁剪结果。"""
        new_aligned_path, old_aligned_path, new_marker_path, old_marker_path = (
            self.aligned_artifact_paths(pair)
        )
        return (
            new_aligned_path.is_file()
            and old_aligned_path.is_file()
            and new_marker_path.is_file()
            and old_marker_path.is_file()
        )

    def resolve_pair_image_paths(self, pair: FitsImagePair) -> tuple[Path, Path, bool]:
        """解析配对应使用的图像路径，优先复用对齐裁剪产物。"""
        if self.pair_has_aligned_artifacts(pair):
            new_aligned_path, old_aligned_path, _, _ = self.aligned_artifact_paths(pair)
            return new_aligned_path, old_aligned_path, True
        return Path(pair.new_path), Path(pair.old_path), False

    def calc_nonzero_valid_bounds(
        self,
        image: np.ndarray | None,
    ) -> tuple[int, int, int, int] | None:
        """估计图像有效区域边界，用于移除对齐黑边。"""
        if image is None or image.size == 0:
            return None

        arr = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        mask = np.abs(arr) > 1e-6
        if not np.any(mask):
            return None

        row_ratio = np.mean(mask, axis=1)
        col_ratio = np.mean(mask, axis=0)

        row_valid = row_ratio > 0.98
        col_valid = col_ratio > 0.98
        if not np.any(row_valid):
            row_valid = np.any(mask, axis=1)
        if not np.any(col_valid):
            col_valid = np.any(mask, axis=0)
        if not np.any(row_valid) or not np.any(col_valid):
            return None

        ys = np.where(row_valid)[0]
        xs = np.where(col_valid)[0]
        y0, y1 = int(ys[0]), int(ys[-1] + 1)
        x0, x1 = int(xs[0]), int(xs[-1] + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, x1, y0, y1

    def calc_overlap_crop_bounds(
        self,
        w: int,
        h: int,
        dx: float,
        dy: float,
        aligned_old: np.ndarray | None = None,
    ) -> tuple[int, int, int, int] | None:
        """根据平移量和旧图有效区域，计算重叠裁剪区域。"""
        x0 = max(0, int(math.ceil(dx)))
        x1 = min(w, int(math.floor(w + dx)))
        y0 = max(0, int(math.ceil(dy)))
        y1 = min(h, int(math.floor(h + dy)))

        if aligned_old is not None:
            valid = self.calc_nonzero_valid_bounds(aligned_old)
            if valid is not None:
                vx0, vx1, vy0, vy1 = valid
                x0 = max(x0, vx0)
                x1 = min(x1, vx1)
                y0 = max(y0, vy0)
                y1 = min(y1, vy1)

        if x1 <= x0 or y1 <= y0:
            return None
        return x0, x1, y0, y1

    def load_pair(self, pair: FitsImagePair) -> ImagePair:
        """读取一对 FITS 图像并转换为内存模型。"""
        new_path, old_path, using_aligned = self.resolve_pair_image_paths(pair)
        new_image = self.read_image(new_path)
        old_image = self.read_image(old_path)
        return ImagePair(
            name=pair.name,
            new_image=new_image,
            old_image=old_image,
            aligned=using_aligned,
        )