"""Sliding-window detection helpers."""

from __future__ import annotations

import logging
from typing import Callable, List

import numpy as np

from scann.core.models import Candidate, CandidateFeatures

logger = logging.getLogger(__name__)


def sliding_window_detect(
    new_data: np.ndarray,
    old_data: np.ndarray,
    *,
    inference_engine,
    patch_size: int,
    is_v1_model: bool,
    extract_patch_fn: Callable[[np.ndarray, int, int, int], np.ndarray],
    prepare_triplet_patch_fn: Callable[[np.ndarray, np.ndarray, int, int, int], np.ndarray],
    nms_candidates_fn: Callable[[List[Candidate], int], List[Candidate]],
) -> List[Candidate]:
    """使用滑动窗口 + AI 在全图上检测候选体。"""
    if not inference_engine or not inference_engine.is_ready:
        return []

    h, w = new_data.shape[:2]
    size = patch_size
    stride = max(size // 2, 1)
    threshold = inference_engine.threshold
    channel_order = getattr(inference_engine, "_channel_order", (0, 1, 2))
    logger.info(
        "滑窗参数: model=%s, image=%dx%d, patch=%d, stride=%d, threshold=%.4f, channel_order=%s",
        "v1" if is_v1_model else "v2",
        w,
        h,
        size,
        stride,
        float(threshold),
        channel_order,
    )

    centers = []
    patches = []
    half = size // 2

    if h <= size:
        y_positions = [max(0, h // 2)]
    else:
        y_positions = list(range(half, h - half + 1, stride))

    if w <= size:
        x_positions = [max(0, w // 2)]
    else:
        x_positions = list(range(half, w - half + 1, stride))

    total_window_count = len(x_positions) * len(y_positions)
    logger.info(
        "滑窗网格: x_positions=%d, y_positions=%d, total_windows=%d",
        len(x_positions),
        len(y_positions),
        total_window_count,
    )

    skipped_by_valid = 0

    def collect_windows(apply_valid_filter: bool) -> None:
        nonlocal skipped_by_valid
        for cy in y_positions:
            for cx in x_positions:
                if is_v1_model and apply_valid_filter and h > size and w > size:
                    old_patch = extract_patch_fn(old_data, cx, cy, size)
                    valid_ratio = float(np.mean(np.abs(old_patch.astype(np.float32)) > 1e-6))
                    if valid_ratio < 0.85:
                        skipped_by_valid += 1
                        continue

                patch_3ch = prepare_triplet_patch_fn(
                    new_data,
                    old_data,
                    cx,
                    cy,
                    size,
                )
                patches.append(patch_3ch)
                centers.append((cx, cy))

    collect_windows(apply_valid_filter=True)
    logger.info(
        "滑窗采样结果(首轮): kept=%d, skipped_by_valid=%d",
        len(patches),
        skipped_by_valid,
    )

    if not patches and is_v1_model and skipped_by_valid > 0:
        logger.info("v1 滑窗有效区过滤后无候选窗口，回退为不过滤模式重试")
        collect_windows(apply_valid_filter=False)
        logger.info(
            "滑窗采样结果(回退): kept=%d, skipped_by_valid=%d",
            len(patches),
            skipped_by_valid,
        )

    if not patches:
        logger.warning(
            "滑窗采样后无可推理窗口: image=%dx%d, patch=%d, stride=%d, is_v1=%s",
            w,
            h,
            size,
            stride,
            is_v1_model,
        )
        return []

    try:
        scores = inference_engine.classify_patches(patches)
    except Exception as exc:
        logger.warning("滑动窗口推理失败: %s", exc)
        return []

    if len(scores) > 0:
        score_array = np.asarray(scores, dtype=np.float32)
        logger.info(
            "滑窗推理分数统计: n=%d, min=%.4f, p50=%.4f, p95=%.4f, max=%.4f",
            score_array.size,
            float(np.min(score_array)),
            float(np.percentile(score_array, 50.0)),
            float(np.percentile(score_array, 95.0)),
            float(np.max(score_array)),
        )

    candidates = []
    for (cx, cy), score in zip(centers, scores):
        if score >= threshold:
            candidates.append(
                Candidate(
                    x=cx,
                    y=cy,
                    features=CandidateFeatures(),
                    ai_score=float(score),
                )
            )
    logger.info(
        "滑窗阈值过滤: pass=%d/%d (threshold=%.4f)",
        len(candidates),
        len(scores),
        float(threshold),
    )

    if len(candidates) > 1:
        candidates = nms_candidates_fn(candidates, min_dist=size // 2)

    return candidates