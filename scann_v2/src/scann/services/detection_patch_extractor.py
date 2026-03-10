"""Detection patch extraction helpers."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from skimage.transform import resize


MODEL_INPUT_SIZE = 224


def extract_patch(
    image: np.ndarray,
    x: int,
    y: int,
    size: int,
) -> np.ndarray:
    """从图像中提取 patch（带 padding）。"""
    half = size // 2

    y0 = max(0, y - half)
    y1 = min(image.shape[0], y + half)
    x0 = max(0, x - half)
    x1 = min(image.shape[1], x + half)

    patch = np.zeros((size, size), dtype=image.dtype)
    patch_height = y1 - y0
    patch_width = x1 - x0

    patch_y0 = half - (y - y0)
    patch_x0 = half - (x - x0)

    patch[patch_y0:patch_y0 + patch_height, patch_x0:patch_x0 + patch_width] = image[y0:y1, x0:x1]

    return patch


def prepare_triplet_patch(
    new_data: np.ndarray,
    old_data: np.ndarray,
    x: int,
    y: int,
    size: int,
    *,
    is_v1_model: bool,
    channel_order: Sequence[int] = (0, 1, 2),
    model_input_size: int = MODEL_INPUT_SIZE,
    extract_patch_fn: Callable[[np.ndarray, int, int, int], np.ndarray] = extract_patch,
) -> np.ndarray:
    """准备三元组 patch。"""
    patch_new = extract_patch_fn(new_data, x, y, size)
    patch_old = extract_patch_fn(old_data, x, y, size)

    if is_v1_model:
        patch_diff = np.clip(
            patch_new.astype(np.float32) - patch_old.astype(np.float32),
            0,
            255,
        ).astype(np.uint8)

        p_diff = patch_diff.astype(np.float32) / 255.0
        p_new = np.clip(patch_new.astype(np.float32), 0.0, 255.0) / 255.0
        p_old = np.clip(patch_old.astype(np.float32), 0.0, 255.0) / 255.0

        if (
            not isinstance(channel_order, (tuple, list))
            or len(channel_order) != 3
            or sorted(channel_order) != [0, 1, 2]
        ):
            channel_order = (0, 1, 2)

        channels = [p_diff, p_new, p_old]
        ordered_channels = [channels[i] for i in channel_order]
        patch_3ch = np.stack(ordered_channels, axis=0).astype(np.float32)
    else:
        patch_diff = patch_new.astype(np.float32) - patch_old.astype(np.float32)

        def normalize(img: np.ndarray) -> np.ndarray:
            if img.max() > img.min():
                return (img - img.min()) / (img.max() - img.min())
            return img - img.min()

        patch_new_norm = normalize(patch_new)
        patch_old_norm = normalize(patch_old)
        patch_diff_norm = normalize(patch_diff)

        channels = [patch_new_norm, patch_old_norm, patch_diff_norm]
        ordered_channels = [channels[i] for i in channel_order]
        patch_3ch = np.stack(ordered_channels, axis=0).astype(np.float32)

    if size != model_input_size:
        patch_3ch = resize(
            patch_3ch,
            (3, model_input_size, model_input_size),
            order=1,
            preserve_range=True,
            anti_aliasing=False,
        )
        patch_3ch = patch_3ch.astype(np.float32)

    return patch_3ch