"""Robust local features and significance-based candidate generation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from scann.core.models import Candidate, CandidateFeatures


SIGNIFICANCE_DETECTOR_VERSION = "significance_v1"


def robust_background(image: np.ndarray) -> tuple[float, float]:
    values = np.asarray(image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    sigma = max(1.4826 * mad, float(np.std(finite)) * 0.05, 1e-6)
    return median, sigma


def significance_map(
    new_data: np.ndarray,
    old_data: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    new_values = np.nan_to_num(
        np.asarray(new_data, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    old_values = np.nan_to_num(
        np.asarray(old_data, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    height = min(new_values.shape[0], old_values.shape[0])
    width = min(new_values.shape[1], old_values.shape[1])
    difference = new_values[:height, :width] - old_values[:height, :width]
    background, sigma = robust_background(difference)
    normalized = (difference - background) / max(sigma, 1e-6)
    return normalized.astype(np.float32), {
        "difference_background": background,
        "difference_sigma": sigma,
    }


def _weighted_centroid(values: np.ndarray) -> tuple[float, float] | None:
    weights = np.clip(
        np.nan_to_num(values.astype(np.float64), nan=0.0),
        0.0,
        None,
    )
    total = float(np.sum(weights))
    if total <= 0.0:
        return None
    yy, xx = np.indices(weights.shape, dtype=np.float64)
    return (
        float(np.sum(xx * weights) / total),
        float(np.sum(yy * weights) / total),
    )


def extract_candidate_features(
    *,
    new_data: np.ndarray,
    old_data: np.ndarray,
    significance: np.ndarray,
    cx: int,
    cy: int,
    bbox: tuple[int, int, int, int],
    polarity: int,
    saturation_level: float | None = None,
) -> CandidateFeatures:
    bx, by, bw, bh = bbox
    height, width = significance.shape[:2]
    pad = max(3, int(round(max(bw, bh))))
    x0 = max(0, bx - pad)
    x1 = min(width, bx + bw + pad)
    y0 = max(0, by - pad)
    y1 = min(height, by + bh + pad)

    local_sig = significance[y0:y1, x0:x1]
    local_new = np.asarray(new_data, dtype=np.float32)[y0:y1, x0:x1]
    local_old = np.asarray(old_data, dtype=np.float32)[y0:y1, x0:x1]
    signed = float(1 if polarity >= 0 else -1)
    response = signed * local_sig
    positive = int(np.count_nonzero(local_sig > 0.0))
    negative = int(np.count_nonzero(local_sig < 0.0))
    pixel_count = max(1, int(local_sig.size))

    peak = float(np.max(response)) if response.size else 0.0
    mean = float(np.mean(response)) if response.size else 0.0
    contrast = peak - float(np.median(response)) if response.size else 0.0
    area = float(max(1, bw * bh))
    extent = min(1.0, max(0.0, float(np.count_nonzero(response > 0.0)) / area))
    aspect = float(bw) / max(1.0, float(bh))

    weights = np.clip(response, 0.0, None)
    if weights.size and float(np.sum(weights)) > 0.0:
        yy, xx = np.indices(weights.shape, dtype=np.float64)
        total = float(np.sum(weights))
        centroid_x = float(np.sum(xx * weights) / total)
        centroid_y = float(np.sum(yy * weights) / total)
        var_x = float(np.sum(((xx - centroid_x) ** 2) * weights) / total)
        var_y = float(np.sum(((yy - centroid_y) ** 2) * weights) / total)
        fwhm = 2.355 * math.sqrt(max(0.0, 0.5 * (var_x + var_y)))
        ellipticity = abs(var_x - var_y) / max(var_x + var_y, 1e-6)
    else:
        fwhm = 0.0
        ellipticity = 0.0

    new_centroid = _weighted_centroid(local_new - np.median(local_new))
    old_centroid = _weighted_centroid(local_old - np.median(local_old))
    if new_centroid is not None and old_centroid is not None:
        centroid_shift = math.hypot(
            new_centroid[0] - old_centroid[0],
            new_centroid[1] - old_centroid[1],
        )
    else:
        centroid_shift = 0.0

    gradient_y, gradient_x = (
        np.gradient(local_sig)
        if min(local_sig.shape, default=0) >= 2
        else (np.zeros_like(local_sig), np.zeros_like(local_sig))
    )
    background_gradient = float(
        np.median(np.hypot(gradient_x, gradient_y))
    ) if local_sig.size else 0.0
    edge_distance = float(min(cx, cy, max(0, width - 1 - cx), max(0, height - 1 - cy)))

    if saturation_level is None or not math.isfinite(float(saturation_level)):
        saturation_level = float(np.percentile(local_new, 99.9)) if local_new.size else 0.0
    saturated_fraction = (
        float(np.count_nonzero(local_new >= float(saturation_level))) / pixel_count
        if local_new.size and float(saturation_level) > 0.0
        else 0.0
    )
    flux_difference = float(np.sum(local_new - local_old)) if local_new.size else 0.0
    val_new = float(np.max(local_new)) if local_new.size else 0.0
    val_old = float(np.max(local_old)) if local_old.size else 0.0
    dipole_score = float(
        min(positive, negative) / max(1, max(positive, negative))
    )

    return CandidateFeatures(
        peak=peak,
        mean=mean,
        sharpness=peak / max(abs(mean), 1e-6),
        contrast=contrast,
        area=area,
        rise=val_new - val_old,
        val_new=val_new,
        val_old=val_old,
        extent=extent,
        aspect_ratio=aspect,
        snr=peak,
        flux_difference=flux_difference,
        fwhm=fwhm,
        ellipticity=ellipticity,
        positive_fraction=float(positive / pixel_count),
        negative_fraction=float(negative / pixel_count),
        dipole_score=dipole_score,
        background_gradient=background_gradient,
        edge_distance=edge_distance,
        saturated_fraction=saturated_fraction,
        centroid_shift=centroid_shift,
        polarity=1 if polarity >= 0 else -1,
    )


def detect_significance_candidates(
    new_data: np.ndarray,
    old_data: np.ndarray,
    *,
    params: Any,
) -> list[Candidate]:
    """Detect positive and negative residuals with a resource safety cap."""

    import cv2

    significance, _statistics = significance_map(new_data, old_data)
    threshold_sigma = max(
        1.0,
        float(getattr(params, "significance_sigma", 5.0)),
    )
    raw_limit = max(
        1,
        int(getattr(params, "raw_candidate_limit", 500)),
    )
    candidates: list[Candidate] = []

    for polarity in (1, -1):
        response = significance * float(polarity)
        mask = (response >= threshold_sigma).astype(np.uint8) * 255
        if bool(getattr(params, "significance_morphology", True)):
            kernel = np.ones((3, 3), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < float(params.min_area) or area > float(params.max_area):
                continue
            bx, by, bw, bh = cv2.boundingRect(contour)
            height, width = significance.shape[:2]
            if (
                bx < int(params.edge_margin)
                or by < int(params.edge_margin)
                or bx + bw > width - int(params.edge_margin)
                or by + bh > height - int(params.edge_margin)
            ):
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            cx = int(round(moments["m10"] / moments["m00"]))
            cy = int(round(moments["m01"] / moments["m00"]))
            features = extract_candidate_features(
                new_data=new_data,
                old_data=old_data,
                significance=significance,
                cx=cx,
                cy=cy,
                bbox=(int(bx), int(by), int(bw), int(bh)),
                polarity=polarity,
            )
            if features.aspect_ratio > float(params.aspect_ratio_max):
                continue
            if features.aspect_ratio < 1.0 / max(float(params.aspect_ratio_max), 1e-6):
                continue
            candidates.append(
                Candidate(
                    x=cx,
                    y=cy,
                    features=features,
                    detector_score=max(0.0, float(features.snr)),
                    polarity=polarity,
                    bbox_x=int(bx),
                    bbox_y=int(by),
                    bbox_width=int(bw),
                    bbox_height=int(bh),
                )
            )
            if len(candidates) >= raw_limit:
                break
        if len(candidates) >= raw_limit:
            break

    candidates.sort(
        key=lambda candidate: (
            float(candidate.detector_score),
            -float(candidate.features.dipole_score),
        ),
        reverse=True,
    )
    return candidates[: max(1, int(params.topk))]
