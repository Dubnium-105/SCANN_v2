"""目标标记模块

职责:
- 在图像上绘制方框/十字线标记
- 保存标记后的 FITS 图像 (含观测日期时间文件名)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from scann.core.models import BitDepth, FitsHeader, MarkerType


def mark_on_image(
    data: np.ndarray,
    x: int,
    y: int,
    marker_type: MarkerType = MarkerType.CROSSHAIR,
    size: int = 20,
    value: Optional[float] = None,
) -> np.ndarray:
    """在图像上绘制标记（返回副本，不修改原图）

    Args:
        data: 原始图像数据
        x, y: 标记中心坐标
        marker_type: 标记类型
        size: 标记大小 (像素)
        value: 标记像素值 (None=自动取最大值)

    Returns:
        标记后的图像副本
    """
    marked = data.copy()
    h, w = data.shape[:2]

    if value is None:
        if np.issubdtype(data.dtype, np.integer):
            value = float(np.iinfo(data.dtype).max)
        else:
            img_max = float(np.max(data))
            value = img_max if img_max > 0 else 1.0

    half = size // 2

    if marker_type == MarkerType.CROSSHAIR:
        # 水平线
        y_start, y_end = max(0, y), min(h, y + 1)
        x_start, x_end = max(0, x - half), min(w, x + half + 1)
        if y_start < y_end:
            marked[y_start:y_end, x_start:x_end] = value

        # 垂直线
        y_start, y_end = max(0, y - half), min(h, y + half + 1)
        x_start, x_end = max(0, x), min(w, x + 1)
        if x_start < x_end:
            marked[y_start:y_end, x_start:x_end] = value

    elif marker_type == MarkerType.BOUNDING_BOX:
        # 上边
        y_top = max(0, y - half)
        marked[y_top, max(0, x - half):min(w, x + half + 1)] = value
        # 下边
        y_bot = min(h - 1, y + half)
        marked[y_bot, max(0, x - half):min(w, x + half + 1)] = value
        # 左边
        x_left = max(0, x - half)
        marked[max(0, y - half):min(h, y + half + 1), x_left] = value
        # 右边
        x_right = min(w - 1, x + half)
        marked[max(0, y - half):min(h, y + half + 1), x_right] = value

    return marked


def generate_marked_filename(
    original_name: str,
    header: Optional[FitsHeader] = None,
    suffix: str = "_marked",
) -> str:
    """生成带观测日期的标记文件名

    避免重名，在文件名后追加 FITS 头中的拍摄日期时间。

    Args:
        original_name: 原始文件名
        header: FITS 文件头
        suffix: 文件名后缀

    Returns:
        生成的文件名
    """
    stem = Path(original_name).stem
    ext = Path(original_name).suffix or ".fits"

    datetime_str = ""
    if header is not None and header.observation_datetime is not None:
        dt = header.observation_datetime
        datetime_str = f"_{dt.strftime('%Y%m%d_%H%M%S')}"

    return f"{stem}{suffix}{datetime_str}{ext}"
