from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.stats import sigma_clipped_stats


@dataclass(frozen=True)
class IntervalMatchResult:
    method: str
    display_min: float
    display_max: float
    background_anchor: float
    highlight_anchor: float
    median: float | None = None
    std: float | None = None
    high_percentile_value: float | None = None
    effective_high_percentile: float | None = None
    tail_ratio_99_9_to_99_5: float | None = None
    background_position: float = 0.10
    highlight_position: float = 0.98


def finite_values(data: np.ndarray) -> np.ndarray:
    values = np.asarray(data, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("image contains no finite pixels")
    return finite


def sample_values(values: np.ndarray, max_samples: int) -> np.ndarray:
    if values.size <= max_samples:
        return values
    indices = np.linspace(0, values.size - 1, max_samples, dtype=np.int64)
    return values[indices]


def compute_zscale_stats(values: np.ndarray, max_samples: int = 200000) -> tuple[float, float]:
    sampled = sample_values(values, max_samples)
    _, median, std = sigma_clipped_stats(sampled, sigma=3.0, maxiters=5)
    return float(median), float(std)


def ensure_valid_cut_range(low: float, high: float, values: np.ndarray) -> tuple[float, float]:
    data_min = float(np.min(values))
    data_max = float(np.max(values))

    if not np.isfinite(low):
        low = data_min
    if not np.isfinite(high):
        high = data_max
    if high <= low:
        if data_max > data_min:
            low, high = data_min, data_max
        else:
            low, high = data_min, data_min + 1.0
    return float(low), float(high)


def adaptive_high_percentile_from_tail_ratio(
    values: np.ndarray,
    *,
    tail_base_percentile: float = 99.5,
    tail_probe_percentile: float = 99.9,
    coeff_a: float = -0.25261547,
    coeff_b: float = 0.53000827,
    coeff_c: float = 99.48221607,
    percentile_min: float = 99.2,
    percentile_max: float = 99.8,
) -> tuple[float, float]:
    base_value = float(np.percentile(values, tail_base_percentile))
    probe_value = float(np.percentile(values, tail_probe_percentile))
    if base_value <= 0:
        tail_ratio = 1.0
    else:
        tail_ratio = probe_value / base_value

    effective_percentile = coeff_a * (tail_ratio**2) + coeff_b * tail_ratio + coeff_c
    effective_percentile = float(np.clip(effective_percentile, percentile_min, percentile_max))
    return effective_percentile, float(tail_ratio)


def brightness_match_anchors(
    data: np.ndarray,
    *,
    max_samples: int = 200000,
    high_percentile: float = 99.9,
    highlight_sigma: float = 5.0,
    adaptive_high_percentile: bool = False,
) -> tuple[float, float, float, float, float, float | None, float]:
    values = finite_values(data)
    median, std = compute_zscale_stats(values, max_samples=max_samples)

    background_anchor = float(median)
    effective_high_percentile = float(high_percentile)
    tail_ratio = None
    if adaptive_high_percentile:
        effective_high_percentile, tail_ratio = adaptive_high_percentile_from_tail_ratio(values)

    percentile_high = float(np.percentile(values, effective_high_percentile))
    if np.isfinite(std) and std > 0:
        sigma_high = float(median + highlight_sigma * std)
        highlight_anchor = max(percentile_high, sigma_high)
    else:
        highlight_anchor = percentile_high

    return (
        background_anchor,
        highlight_anchor,
        median,
        std,
        percentile_high,
        tail_ratio,
        effective_high_percentile,
    )


def compute_brightness_match_interval(
    data: np.ndarray,
    *,
    max_samples: int = 200000,
    high_percentile: float = 99.9,
    highlight_sigma: float = 5.0,
    background_position: float = 0.10,
    highlight_position: float = 0.98,
    adaptive_high_percentile: bool = False,
    method_name: str = "brightness_matched_linear_interval",
) -> IntervalMatchResult:
    values = finite_values(data)
    (
        background_anchor,
        highlight_anchor,
        median,
        std,
        percentile_high,
        tail_ratio,
        effective_high_percentile,
    ) = brightness_match_anchors(
        values,
        max_samples=max_samples,
        high_percentile=high_percentile,
        highlight_sigma=highlight_sigma,
        adaptive_high_percentile=adaptive_high_percentile,
    )

    if highlight_position <= background_position:
        raise ValueError("highlight_position must be greater than background_position")

    display_min = (
        highlight_position * background_anchor - background_position * highlight_anchor
    ) / (highlight_position - background_position)
    display_max = display_min + (highlight_anchor - background_anchor) / (
        highlight_position - background_position
    )
    display_min, display_max = ensure_valid_cut_range(display_min, display_max, values)

    return IntervalMatchResult(
        method=method_name,
        display_min=display_min,
        display_max=display_max,
        background_anchor=background_anchor,
        highlight_anchor=highlight_anchor,
        median=median,
        std=std,
        high_percentile_value=percentile_high,
        effective_high_percentile=effective_high_percentile,
        tail_ratio_99_9_to_99_5=tail_ratio,
        background_position=background_position,
        highlight_position=highlight_position,
    )


def infer_match_positions_from_target_interval(
    data: np.ndarray,
    *,
    target_min: float,
    target_max: float,
    max_samples: int = 200000,
    high_percentile: float = 99.9,
    highlight_sigma: float = 5.0,
    adaptive_high_percentile: bool = False,
    method_name: str = "inferred_match_positions",
) -> IntervalMatchResult:
    if target_max <= target_min:
        raise ValueError("target_max must be greater than target_min")

    (
        background_anchor,
        highlight_anchor,
        median,
        std,
        percentile_high,
        tail_ratio,
        effective_high_percentile,
    ) = brightness_match_anchors(
        data,
        max_samples=max_samples,
        high_percentile=high_percentile,
        highlight_sigma=highlight_sigma,
        adaptive_high_percentile=adaptive_high_percentile,
    )

    width = float(target_max - target_min)
    background_position = (background_anchor - target_min) / width
    highlight_position = (highlight_anchor - target_min) / width

    return IntervalMatchResult(
        method=method_name,
        display_min=float(target_min),
        display_max=float(target_max),
        background_anchor=background_anchor,
        highlight_anchor=highlight_anchor,
        median=median,
        std=std,
        high_percentile_value=percentile_high,
        effective_high_percentile=effective_high_percentile,
        tail_ratio_99_9_to_99_5=tail_ratio,
        background_position=float(background_position),
        highlight_position=float(highlight_position),
    )


__all__ = [
    "IntervalMatchResult",
    "adaptive_high_percentile_from_tail_ratio",
    "brightness_match_anchors",
    "compute_brightness_match_interval",
    "compute_zscale_stats",
    "ensure_valid_cut_range",
    "finite_values",
    "infer_match_positions_from_target_interval",
    "sample_values",
]
