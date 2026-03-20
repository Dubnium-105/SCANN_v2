from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from PIL import Image


class FITSEngine:
    """FITS 读取与 PNG 渲染引擎（含简单内存缓存）。"""

    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = dataset_root.resolve()
        self._cache: Dict[Tuple[str, float], bytes] = {}

    def _resolve_file(self, relative_path: str) -> Path:
        file_path = (self.dataset_root / relative_path).resolve()
        file_path.relative_to(self.dataset_root)
        return file_path

    @staticmethod
    def _normalize_to_uint8(data: np.ndarray, method: str = "zscale") -> np.ndarray:
        image = np.asarray(data, dtype=np.float32)
        finite_mask = np.isfinite(image)
        if not finite_mask.any():
            return np.zeros_like(image, dtype=np.uint8)

        finite_vals = image[finite_mask]
        if method == "zscale":
            interval = ZScaleInterval()
            vmin, vmax = interval.get_limits(finite_vals)
        else:
            vmin, vmax = float(np.min(finite_vals)), float(np.max(finite_vals))

        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = float(np.min(finite_vals)), float(np.max(finite_vals))
            if vmax <= vmin:
                return np.zeros_like(image, dtype=np.uint8)

        scaled = (image - vmin) / (vmax - vmin)
        scaled = np.clip(scaled, 0.0, 1.0)
        scaled[~finite_mask] = 0.0
        return (scaled * 255.0).astype(np.uint8)

    def render_png(self, relative_path: str, method: str = "zscale") -> bytes:
        file_path = self._resolve_file(relative_path)
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(str(file_path))

        cache_key = (str(file_path), file_path.stat().st_mtime)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        with fits.open(file_path, memmap=False) as hdul:
            data = hdul[0].data

        if data is None:
            raise ValueError(f"FITS data is empty: {file_path}")

        if data.ndim > 2:
            data = np.squeeze(data)
        if data.ndim != 2:
            raise ValueError(f"Unsupported FITS shape: {data.shape}")

        image_u8 = self._normalize_to_uint8(data, method=method)
        image = Image.fromarray(image_u8)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()
        self._cache[cache_key] = png_bytes
        return png_bytes

    def get_fits_binary(self, relative_path: str) -> bytes:
        """返回FITS文件的原始二进制数据。"""
        file_path = self._resolve_file(relative_path)
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(str(file_path))

        return file_path.read_bytes()
