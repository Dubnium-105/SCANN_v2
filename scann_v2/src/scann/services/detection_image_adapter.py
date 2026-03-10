"""Detection image adapter helpers."""

from __future__ import annotations

import numpy as np


def img_brief_stats(name: str, img: np.ndarray) -> str:
    """生成轻量图像统计字符串，用于日志诊断。"""
    arr = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        return f"{name}:empty"
    nz = float(np.mean(np.abs(arr) > 1e-6))
    p1 = float(np.percentile(arr, 1.0))
    p50 = float(np.percentile(arr, 50.0))
    p99 = float(np.percentile(arr, 99.0))
    return (
        f"{name}:shape={arr.shape},dtype={img.dtype},"
        f"nz={nz:.3f},p1={p1:.3f},p50={p50:.3f},p99={p99:.3f}"
    )


def robust_to_uint8(image: np.ndarray) -> np.ndarray:
    """将任意数值范围图像稳健映射到 uint8。"""
    if image.dtype == np.uint8:
        return image

    img = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if img.size == 0:
        return np.zeros_like(img, dtype=np.uint8)

    p_low = float(np.percentile(img, 1.0))
    p_high = float(np.percentile(img, 99.0))
    if not np.isfinite(p_low) or not np.isfinite(p_high) or p_high <= p_low:
        p_low = float(np.min(img))
        p_high = float(np.max(img))
        if p_high <= p_low:
            return np.zeros_like(img, dtype=np.uint8)

    scaled = (img - p_low) / (p_high - p_low)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)