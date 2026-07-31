"""Deterministic synthetic point-source injection for offline evaluation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from scann.services.candidate_feature_extractor import robust_background


SOURCE_INJECTION_VERSION = "source-injection-v1"


@dataclass(frozen=True)
class InjectedSource:
    source_id: str
    x: float
    y: float
    snr: float
    fwhm: float
    amplitude: float
    polarity: int
    edge_region: bool


@dataclass(frozen=True)
class InjectionResult:
    image: np.ndarray
    sources: tuple[InjectedSource, ...]
    metadata: dict[str, Any]

    def truth_boxes(self, scale: float = 2.5) -> list[dict[str, Any]]:
        boxes: list[dict[str, Any]] = []
        for source in self.sources:
            size = max(3.0, float(source.fwhm) * float(scale))
            boxes.append(
                {
                    "x": float(source.x - size / 2.0),
                    "y": float(source.y - size / 2.0),
                    "width": size,
                    "height": size,
                    "detail_type": "injected_source",
                    "source_id": source.source_id,
                    "snr": source.snr,
                    "fwhm": source.fwhm,
                    "polarity": source.polarity,
                }
            )
        return boxes


def _overlaps_forbidden(
    x: float,
    y: float,
    forbidden_boxes: Sequence[Mapping[str, Any]],
    margin: float,
) -> bool:
    for box in forbidden_boxes:
        left = float(box.get("x") or box.get("left") or 0.0) - margin
        top = float(box.get("y") or box.get("top") or 0.0) - margin
        width = max(0.0, float(box.get("width") or 0.0)) + 2.0 * margin
        height = max(0.0, float(box.get("height") or 0.0)) + 2.0 * margin
        if left <= x <= left + width and top <= y <= top + height:
            return True
    return False


def inject_point_sources(
    image: np.ndarray,
    *,
    count: int,
    seed: int,
    snr_values: Sequence[float] = (3.0, 5.0, 8.0, 12.0),
    fwhm_values: Sequence[float] = (2.0, 3.0, 4.5),
    polarity_values: Sequence[int] = (1,),
    forbidden_boxes: Sequence[Mapping[str, Any]] = (),
    edge_margin: int = 12,
) -> InjectionResult:
    """Return an injected copy. The original array is never modified."""

    source = np.asarray(image, dtype=np.float32)
    if source.ndim != 2 or source.size == 0:
        raise ValueError("source injection requires a non-empty 2D image")
    if count <= 0:
        raise ValueError("source injection count must be positive")
    if not snr_values or not fwhm_values or not polarity_values:
        raise ValueError("source injection parameter grids cannot be empty")

    output = np.array(source, dtype=np.float32, copy=True)
    background, noise_sigma = robust_background(source)
    rng = np.random.default_rng(int(seed))
    height, width = output.shape
    safe_margin = max(
        int(edge_margin),
        int(math.ceil(max(float(value) for value in fwhm_values) * 2.0)),
    )
    if width <= 2 * safe_margin or height <= 2 * safe_margin:
        raise ValueError("image is too small for requested injection margin")

    yy, xx = np.indices(output.shape, dtype=np.float32)
    injected: list[InjectedSource] = []
    attempts = 0
    max_attempts = max(100, int(count) * 100)
    while len(injected) < int(count) and attempts < max_attempts:
        attempts += 1
        x = float(rng.uniform(safe_margin, width - safe_margin))
        y = float(rng.uniform(safe_margin, height - safe_margin))
        fwhm = float(rng.choice(np.asarray(fwhm_values, dtype=np.float64)))
        if _overlaps_forbidden(
            x,
            y,
            forbidden_boxes,
            margin=max(2.0, fwhm),
        ):
            continue
        if any(
            math.hypot(existing.x - x, existing.y - y)
            < max(existing.fwhm, fwhm) * 2.5
            for existing in injected
        ):
            continue
        snr = float(rng.choice(np.asarray(snr_values, dtype=np.float64)))
        polarity = int(rng.choice(np.asarray(polarity_values, dtype=np.int64)))
        polarity = 1 if polarity >= 0 else -1
        amplitude = snr * max(noise_sigma, 1e-6)
        sigma = max(fwhm / 2.355, 0.5)
        gaussian = amplitude * np.exp(
            -((xx - x) ** 2 + (yy - y) ** 2)
            / (2.0 * sigma * sigma)
        )
        output += gaussian.astype(np.float32) * float(polarity)
        injected.append(
            InjectedSource(
                source_id=f"injected-{len(injected) + 1:05d}",
                x=x,
                y=y,
                snr=snr,
                fwhm=fwhm,
                amplitude=amplitude,
                polarity=polarity,
                edge_region=(
                    min(x, y, width - x, height - y)
                    < max(32.0, safe_margin * 2.0)
                ),
            )
        )
    if len(injected) != int(count):
        raise ValueError(
            f"could only place {len(injected)} of {count} requested sources"
        )
    return InjectionResult(
        image=output,
        sources=tuple(injected),
        metadata={
            "version": SOURCE_INJECTION_VERSION,
            "seed": int(seed),
            "requested_count": int(count),
            "injected_count": len(injected),
            "background": background,
            "noise_sigma": noise_sigma,
            "snr_values": [float(value) for value in snr_values],
            "fwhm_values": [float(value) for value in fwhm_values],
            "polarity_values": [
                1 if int(value) >= 0 else -1
                for value in polarity_values
            ],
            "attempt_count": attempts,
        },
    )


def recovery_curve(
    injected_sources: Iterable[InjectedSource],
    matched_source_ids: Iterable[str],
) -> dict[str, Any]:
    matched = {str(item) for item in matched_source_ids}
    sources = list(injected_sources)
    by_snr: dict[str, dict[str, Any]] = {}
    by_fwhm: dict[str, dict[str, Any]] = {}
    for key_name, values in (
        ("snr", by_snr),
        ("fwhm", by_fwhm),
    ):
        grouped: dict[float, list[InjectedSource]] = {}
        for source in sources:
            grouped.setdefault(float(getattr(source, key_name)), []).append(source)
        for value, grouped_sources in sorted(grouped.items()):
            recovered = sum(
                1
                for source in grouped_sources
                if source.source_id in matched
            )
            values[str(value)] = {
                "support": len(grouped_sources),
                "recovered": recovered,
                "recall": recovered / len(grouped_sources),
            }
    edge_sources = [source for source in sources if source.edge_region]
    center_sources = [source for source in sources if not source.edge_region]

    def summarize(items: Sequence[InjectedSource]) -> dict[str, Any]:
        recovered = sum(1 for item in items if item.source_id in matched)
        return {
            "support": len(items),
            "recovered": recovered,
            "recall": recovered / len(items) if items else None,
        }

    return {
        "version": SOURCE_INJECTION_VERSION,
        "total": summarize(sources),
        "by_snr": by_snr,
        "by_fwhm": by_fwhm,
        "center": summarize(center_sources),
        "edge": summarize(edge_sources),
    }
