"""Dependency-light aggregation for discovery pipeline observability."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence


MONITORING_SCHEMA_VERSION = "discovery-monitoring-v1"


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(item) for item in values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def aggregate_detection_metrics(
    traces: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    items = [dict(item) for item in traces]
    durations = [
        float(item.get("duration_ms") or 0.0)
        for item in items
    ]
    final_counts = [
        int(
            item.get("final_candidate_count")
            or (item.get("stage_counts") or {}).get("final")
            or 0
        )
        for item in items
    ]
    stage_counts: dict[str, list[int]] = {}
    errors: Counter[str] = Counter()
    fallback_reasons: Counter[str] = Counter()
    for item in items:
        for stage, count in (item.get("stage_counts") or {}).items():
            stage_counts.setdefault(str(stage), []).append(int(count))
        if str(item.get("error") or "").strip():
            errors[str(item.get("error"))] += 1
        fallback_reasons.update(
            str(reason)
            for reason in item.get("fallback_reasons") or []
        )
    return {
        "version": MONITORING_SCHEMA_VERSION,
        "trace_count": len(items),
        "empty_result_rate": (
            sum(1 for count in final_counts if count == 0) / len(items)
            if items
            else None
        ),
        "duration_ms": {
            "p50": _percentile(durations, 50.0),
            "p95": _percentile(durations, 95.0),
            "p99": _percentile(durations, 99.0),
        },
        "final_candidates": {
            "p50": _percentile(final_counts, 50.0),
            "p95": _percentile(final_counts, 95.0),
            "p99": _percentile(final_counts, 99.0),
        },
        "stage_candidates": {
            stage: {
                "p50": _percentile(counts, 50.0),
                "p95": _percentile(counts, 95.0),
                "p99": _percentile(counts, 99.0),
            }
            for stage, counts in sorted(stage_counts.items())
        },
        "errors": dict(errors),
        "fallback_reasons": dict(fallback_reasons),
    }
