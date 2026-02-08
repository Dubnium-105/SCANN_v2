"""候选目标检测模块

职责:
- 从新旧差异图中检测可疑目标
- 计算候选体特征
- Top-K 过滤
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from scann.core.models import Candidate, CandidateFeatures


@dataclass
class DetectionParams:
    """检测参数"""
    thresh: int = 80
    min_area: int = 6
    max_area: int = 600
    sharpness_min: float = 1.2
    sharpness_max: float = 5.0
    contrast_min: int = 15
    edge_margin: int = 10
    dynamic_thresh: bool = False
    kill_flat: bool = True
    kill_dipole: bool = True
    aspect_ratio_max: float = 3.0
    extent_max: float = 0.90
    topk: int = 20


def detect_candidates(
    new_data: np.ndarray,
    old_data: np.ndarray,
    diff_data: Optional[np.ndarray] = None,
    params: Optional[DetectionParams] = None,
) -> List[Candidate]:
    """从新旧图像中检测候选目标

    Args:
        new_data: 新图像素数据 (灰度)
        old_data: 旧图像素数据 (灰度, 已对齐)
        diff_data: 差异图 (可选, None则自动计算)
        params: 检测参数

    Returns:
        候选目标列表
    """
    import cv2

    if params is None:
        params = DetectionParams()

    # 计算差异图
    if diff_data is None:
        new_f = new_data.astype(np.float32)
        old_f = old_data.astype(np.float32)
        diff_data = np.clip(new_f - old_f, 0, 255).astype(np.uint8)

    h_img, w_img = diff_data.shape[:2]

    # 阈值
    actual_thresh = params.thresh
    if params.dynamic_thresh:
        actual_thresh = int(np.median(diff_data)) + params.thresh

    # 二值化 + 轮廓
    blurred = cv2.GaussianBlur(diff_data, (3, 3), 0)
    _, bin_img = cv2.threshold(blurred, actual_thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < params.min_area or area > params.max_area:
            continue

        bx, by, bw, bh = cv2.boundingRect(c)

        # 边缘排除
        if (bx < params.edge_margin or by < params.edge_margin or
                bx + bw > w_img - params.edge_margin or
                by + bh > h_img - params.edge_margin):
            continue

        # 重心
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # 计算特征
        features = _compute_features(
            diff_data, new_data, old_data,
            cx, cy, bx, by, bw, bh, area, params
        )
        if features is None:
            continue

        candidates.append(Candidate(x=cx, y=cy, features=features))

    # Top-K 排序（按 cheap_score 降序）
    candidates.sort(key=lambda c: _cheap_score(c.features), reverse=True)
    return candidates[:params.topk]


def _compute_features(
    diff_data: np.ndarray,
    new_data: np.ndarray,
    old_data: np.ndarray,
    cx: int, cy: int,
    bx: int, by: int, bw: int, bh: int,
    area: float,
    params: DetectionParams,
) -> Optional[CandidateFeatures]:
    """计算单个候选体的特征"""
    h_img, w_img = diff_data.shape[:2]

    # 局部 ROI
    roi_spot = diff_data[by:by + bh, bx:bx + bw]
    if roi_spot.size == 0:
        return None

    peak = float(np.max(roi_spot))
    mean = float(np.mean(roi_spot))
    median_spot = float(np.median(roi_spot))
    sharpness = peak / (mean + 1e-6)
    contrast = peak - median_spot

    # 平坦过滤
    if params.kill_flat:
        if sharpness < params.sharpness_min or sharpness > params.sharpness_max:
            return None
        if contrast < params.contrast_min:
            return None

    # 形态过滤
    extent = float(area) / (bw * bh) if (bw * bh) > 0 else 0
    aspect = float(bw) / bh if bh > 0 else 0
    if area > 20 and extent > params.extent_max:
        return None
    if aspect > params.aspect_ratio_max or aspect < 1.0 / params.aspect_ratio_max:
        return None

    # 偶极子过滤
    if params.kill_dipole:
        pad_d = 4
        dy0 = max(0, by - pad_d)
        dy1 = min(h_img, by + bh + pad_d)
        dx0 = max(0, bx - pad_d)
        dx1 = min(w_img, bx + bw + pad_d)
        local_min = float(np.min(diff_data[dy0:dy1, dx0:dx1]))
        if local_min < 15:
            return None

    # 增亮
    check_r = 3
    y0_r = max(0, cy - check_r)
    y1_r = min(h_img, cy + check_r + 1)
    x0_r = max(0, cx - check_r)
    x1_r = min(w_img, cx + check_r + 1)

    roi_new = new_data[y0_r:y1_r, x0_r:x1_r]
    roi_old = old_data[y0_r:y1_r, x0_r:x1_r]
    val_new = float(np.max(roi_new)) if roi_new.size > 0 else 0
    val_old = float(np.max(roi_old)) if roi_old.size > 0 else 0
    rise = val_new - val_old

    return CandidateFeatures(
        peak=peak,
        mean=mean,
        sharpness=sharpness,
        contrast=contrast,
        area=area,
        rise=rise,
        val_new=val_new,
        val_old=val_old,
        extent=extent,
        aspect_ratio=aspect,
    )


def _cheap_score(features: CandidateFeatures) -> float:
    """快速评分 (用于 Top-K 排序)"""
    return 2.0 * features.rise + 1.0 * features.contrast + 0.5 * features.sharpness
