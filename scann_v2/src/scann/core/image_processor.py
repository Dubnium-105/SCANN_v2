"""图像处理模块

职责:
- 直方图/屏幕拉伸 (仅显示，不保存)
- 反色 (仅显示)
- 去噪点
- 伪平场
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def histogram_stretch(
    data: np.ndarray,
    black_point: Optional[float] = None,
    white_point: Optional[float] = None,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> np.ndarray:
    """直方图拉伸 - 临时调整显示亮度，不改变原始数据

    Args:
        data: 原始像素数据
        black_point: 手动指定黑点 (None=自动)
        white_point: 手动指定白点 (None=自动)
        percentile_low: 自动模式下低百分位
        percentile_high: 自动模式下高百分位

    Returns:
        拉伸后的显示用数据 (float32, 0~1 范围)
    """
    if data.size == 0:
        return data.astype(np.float32)

    data_f = data.astype(np.float32)

    if black_point is None:
        black_point = float(np.percentile(data_f, percentile_low))
    if white_point is None:
        white_point = float(np.percentile(data_f, percentile_high))

    # 防止除零
    span = white_point - black_point
    if span <= 0:
        span = 1.0

    stretched = (data_f - black_point) / span
    stretched = np.clip(stretched, 0.0, 1.0)
    return stretched


def invert(data: np.ndarray) -> np.ndarray:
    """反色显示 - 仅临时处理，不保存

    Args:
        data: 输入数据 (float32, 0~1 或 uint 类型)

    Returns:
        反色后的数据（与输入同类型）
    """
    if data.dtype == np.float32 or data.dtype == np.float64:
        return 1.0 - data
    else:
        # 整数类型
        info = np.iinfo(data.dtype)
        return (info.max - data).astype(data.dtype)


def denoise(
    data: np.ndarray,
    method: str = "median",
    kernel_size: int = 3,
) -> np.ndarray:
    """去噪点

    Args:
        data: 原始像素数据
        method: 去噪方法 ("median", "gaussian", "bilateral")
        kernel_size: 滤波核大小

    Returns:
        去噪后的数据
    """
    import cv2

    if method == "median":
        if data.dtype != np.uint8 and data.dtype != np.uint16:
            # OpenCV median filter 需要特定类型
            data_work = data.astype(np.float32)
            result = cv2.medianBlur(data_work, kernel_size)
            return result.astype(data.dtype)
        return cv2.medianBlur(data, kernel_size)
    elif method == "gaussian":
        return cv2.GaussianBlur(data, (kernel_size, kernel_size), 0)
    elif method == "bilateral":
        data_f = data.astype(np.float32)
        return cv2.bilateralFilter(data_f, kernel_size, 75, 75).astype(data.dtype)
    else:
        raise ValueError(f"不支持的去噪方法: {method}")


def pseudo_flat_field(
    data: np.ndarray,
    kernel_size: int = 127,
) -> np.ndarray:
    """伪平场校正

    通过大核中值/高斯滤波估算背景，然后用原始数据除以背景。

    Args:
        data: 原始像素数据
        kernel_size: 平场估算核大小 (建议大奇数, 如 127)

    Returns:
        平场校正后的数据
    """
    import cv2

    data_f = data.astype(np.float64)

    # 用大核高斯滤波估算背景
    background = cv2.GaussianBlur(data_f, (kernel_size, kernel_size), 0)

    # 防止除零
    background = np.maximum(background, 1.0)

    # 平均背景值，用于保持亮度量级
    mean_bg = np.mean(background)

    # 校正: data * mean_bg / background
    corrected = data_f * mean_bg / background

    # 裁剪到原始数据范围
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        corrected = np.clip(corrected, info.min, info.max)

    return corrected.astype(data.dtype)


def compute_statistics(data: np.ndarray) -> dict:
    """计算图像统计信息（用于直方图显示）"""
    return {
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
    }
