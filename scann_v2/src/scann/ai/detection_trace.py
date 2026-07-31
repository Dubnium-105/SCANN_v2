"""Versioned observability records for candidate detection."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np


DETECTION_TRACE_VERSION = "scann-detection-trace-v1"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return default
    return normalized if math.isfinite(normalized) else default


def image_statistics(image: np.ndarray) -> dict[str, Any]:
    """Return compact robust statistics without modifying the input array."""

    values = np.asarray(image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "shape": list(values.shape),
            "finite_fraction": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
            "median": 0.0,
            "mad": 0.0,
            "p01": 0.0,
            "p99": 0.0,
        }
    median = float(np.median(finite))
    return {
        "shape": list(values.shape),
        "finite_fraction": float(finite.size / max(1, values.size)),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
        "median": median,
        "mad": float(np.median(np.abs(finite - median))),
        "p01": float(np.percentile(finite, 1.0)),
        "p99": float(np.percentile(finite, 99.0)),
    }


def score_summary(scores: Sequence[float]) -> dict[str, Any]:
    normalized = np.asarray(
        [_finite_float(value) for value in scores],
        dtype=np.float64,
    )
    if normalized.size == 0:
        return {
            "count": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "p50": None,
            "p95": None,
        }
    return {
        "count": int(normalized.size),
        "minimum": float(np.min(normalized)),
        "maximum": float(np.max(normalized)),
        "mean": float(np.mean(normalized)),
        "p50": float(np.percentile(normalized, 50.0)),
        "p95": float(np.percentile(normalized, 95.0)),
    }


@dataclass
class DetectionTrace:
    """One immutable-on-completion detection execution trace."""

    pair_name: str
    detector_version: str
    detection_mode: str
    trace_version: str = DETECTION_TRACE_VERSION
    started_at: str = field(default_factory=_utc_now_iso)
    finished_at: str | None = None
    duration_ms: float = 0.0
    image_stats: dict[str, Any] = field(default_factory=dict)
    alignment: dict[str, Any] = field(default_factory=dict)
    stage_counts: dict[str, int] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    fallback_reasons: list[str] = field(default_factory=list)
    score_distribution: dict[str, Any] = field(default_factory=dict)
    final_candidate_count: int = 0
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    _started_monotonic: float = field(
        default_factory=time.perf_counter,
        repr=False,
    )

    def record_stage(
        self,
        name: str,
        count: int,
        *,
        duration_ms: float | None = None,
    ) -> None:
        normalized_name = str(name or "").strip()
        if not normalized_name:
            return
        self.stage_counts[normalized_name] = max(0, int(count))
        if duration_ms is not None:
            self.timings_ms[normalized_name] = max(
                0.0,
                _finite_float(duration_ms),
            )

    def record_fallback(self, reason: Any) -> None:
        normalized = str(reason or "").strip()
        if normalized and normalized not in self.fallback_reasons:
            self.fallback_reasons.append(normalized)

    def finish(
        self,
        *,
        candidates: Sequence[Any] = (),
        error: str = "",
    ) -> None:
        self.finished_at = _utc_now_iso()
        self.duration_ms = max(
            0.0,
            (time.perf_counter() - self._started_monotonic) * 1000.0,
        )
        self.error = str(error or "")
        self.final_candidate_count = len(candidates)
        self.stage_counts["final"] = len(candidates)
        self.score_distribution = score_summary(
            [
                _finite_float(getattr(candidate, "ai_score", 0.0))
                for candidate in candidates
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("_started_monotonic", None)
        return payload


def trace_from_mapping(payload: Mapping[str, Any]) -> DetectionTrace:
    """Load the stable public fields of a persisted trace."""

    trace = DetectionTrace(
        pair_name=str(payload.get("pair_name") or ""),
        detector_version=str(payload.get("detector_version") or "unknown"),
        detection_mode=str(payload.get("detection_mode") or "unknown"),
    )
    for field_name in (
        "trace_version",
        "started_at",
        "finished_at",
        "duration_ms",
        "image_stats",
        "alignment",
        "stage_counts",
        "thresholds",
        "timings_ms",
        "fallback_reasons",
        "score_distribution",
        "final_candidate_count",
        "error",
        "metadata",
    ):
        if field_name in payload:
            setattr(trace, field_name, payload[field_name])
    return trace
