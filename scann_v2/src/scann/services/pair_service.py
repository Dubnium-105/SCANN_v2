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

    def resolve_marked_image_path(self, new_image_path: str | Path) -> Path | None:
        """根据新图路径推断同名带标记新图。"""
        new_path = Path(new_image_path)
        if new_path.parent.name != "new":
            return None

        marked_path = new_path.parent.parent / "new_marked" / new_path.name
        if marked_path.is_file():
            return marked_path
        return None

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
        # 使用更敏感的阈值来检测有效信号
        mask = np.abs(arr) > 1e-4
        if not np.any(mask):
            return None

        row_ratio = np.mean(mask, axis=1)
        col_ratio = np.mean(mask, axis=0)

        # 优先保留“绝大多数像素有效”的行列，便于去掉对齐后的黑边
        row_valid = row_ratio > 0.995
        col_valid = col_ratio > 0.995
        if not np.any(row_valid):
            row_valid = row_ratio > 0.98
        if not np.any(col_valid):
            col_valid = col_ratio > 0.98
        if not np.any(row_valid):
            row_valid = row_ratio > 0.90
        if not np.any(col_valid):
            col_valid = col_ratio > 0.90
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

    def calc_overlap_crop_bounds_from_aligned_images(
        self,
        new_image: np.ndarray | None,
        aligned_old: np.ndarray | None,
    ) -> tuple[int, int, int, int] | None:
        """根据已在同一坐标系的新旧图直接计算重叠有效区域。"""
        if new_image is None or aligned_old is None:
            return None
        if new_image.size == 0 or aligned_old.size == 0:
            return None
        if new_image.shape[:2] != aligned_old.shape[:2]:
            return None

        new_arr = np.nan_to_num(new_image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        old_arr = np.nan_to_num(aligned_old.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        eps_new = max(float(np.percentile(np.abs(new_arr), 99)) * 1e-6, 1e-6)
        eps_old = max(float(np.percentile(np.abs(old_arr), 99)) * 1e-6, 1e-6)
        mask_new = np.abs(new_arr) > eps_new
        mask_old = np.abs(old_arr) > eps_old

        def _remove_edge_floor(mask: np.ndarray, arr: np.ndarray) -> np.ndarray:
            """去除边缘低值填充带（常见于仿射插值后的无效区）。"""
            a_min = float(np.min(arr))
            a_max = float(np.max(arr))
            a_rng = a_max - a_min
            if a_rng <= 1e-6:
                return mask

            floor_thr = a_min + max(a_rng * 1e-3, 1e-6)
            low_mask = arr <= floor_thr
            if not np.any(low_mask):
                return mask

            edge = np.zeros_like(low_mask, dtype=bool)
            edge[0, :] = True
            edge[-1, :] = True
            edge[:, 0] = True
            edge[:, -1] = True

            edge_low_ratio = float(np.mean(low_mask[edge]))
            global_low_ratio = float(np.mean(low_mask))
            if edge_low_ratio > 0.10 and edge_low_ratio > global_low_ratio * 1.5:
                return mask & (~low_mask)
            return mask

        mask_new = _remove_edge_floor(mask_new, new_arr)
        mask_old = _remove_edge_floor(mask_old, old_arr)
        common = mask_new & mask_old
        if not np.any(common):
            return None

        row_ratio = np.mean(common, axis=1)
        col_ratio = np.mean(common, axis=0)

        row_valid = None
        col_valid = None
        for thr in (0.999, 0.995, 0.98, 0.95, 0.90):
            rv = row_ratio > thr
            cv = col_ratio > thr
            if np.any(rv) and np.any(cv):
                row_valid = rv
                col_valid = cv
                break

        if row_valid is None or col_valid is None:
            row_valid = np.any(common, axis=1)
            col_valid = np.any(common, axis=0)
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
        new_image: np.ndarray | None = None,
    ) -> tuple[int, int, int, int] | None:
        """根据平移量和新旧图有效区域，计算重叠裁剪区域（取交集以移除L型黑边）。"""
        direct_bounds = self.calc_overlap_crop_bounds_from_aligned_images(new_image, aligned_old)
        if direct_bounds is not None:
            return direct_bounds

        # 计算几何重叠区域
        x0 = max(0, int(math.ceil(dx)))
        x1 = min(w, int(math.floor(w + dx)))
        y0 = max(0, int(math.ceil(dy)))
        y1 = min(h, int(math.floor(h + dy)))

        # 获取旧图的有效区域边界
        if aligned_old is not None:
            valid_old = self.calc_nonzero_valid_bounds(aligned_old)
            if valid_old is not None:
                vx0, vx1, vy0, vy1 = valid_old
                x0 = max(x0, vx0)
                x1 = min(x1, vx1)
                y0 = max(y0, vy0)
                y1 = min(y1, vy1)

        # 获取新图的有效区域边界
        if new_image is not None:
            valid_new = self.calc_nonzero_valid_bounds(new_image)
            if valid_new is not None:
                # 将新图的有效区域坐标转换到旧图坐标系
                # 新图坐标 (x, y) 对应对齐后的坐标 (x - dx, y - dy)
                # 新图有效区域 [vx0, vx1) 转换到旧图坐标系为 [vx0 - dx, vx1 - dx)
                # 然后再与几何重叠区域取交集
                vx0, vx1, vy0, vy1 = valid_new
                # 计算新图有效区域在旧图坐标系中的边界
                new_x0 = int(math.ceil(dx + vx0))  # 新图左边框在旧图坐标系中的位置
                new_x1 = int(math.floor(dx + vx1))  # 新图右边框在旧图坐标系中的位置
                new_y0 = int(math.ceil(dy + vy0))
                new_y1 = int(math.floor(dy + vy1))
                # 与当前边界取交集
                x0 = max(x0, new_x0)
                x1 = min(x1, new_x1)
                y0 = max(y0, new_y0)
                y1 = min(y1, new_y1)

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
