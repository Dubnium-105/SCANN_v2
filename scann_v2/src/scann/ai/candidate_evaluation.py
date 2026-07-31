"""Offline candidate-stage evaluation and immutable artifact writing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CANDIDATE_EVALUATOR_VERSION = "candidate-evaluator-v1"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return default
    return normalized if math.isfinite(normalized) else default


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(percentile) / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass(frozen=True)
class EvaluationBox:
    x: float
    y: float
    width: float
    height: float
    detail_type: str | None = None
    score: float | None = None
    stage: str | None = None
    candidate_id: str | None = None

    @property
    def center(self) -> tuple[float, float]:
        return (
            self.x + self.width / 2.0,
            self.y + self.height / 2.0,
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EvaluationBox":
        width = max(1.0, _safe_float(payload.get("width"), 1.0))
        height = max(1.0, _safe_float(payload.get("height"), 1.0))
        if payload.get("left") is not None:
            x = _safe_float(payload.get("left"))
        else:
            x = _safe_float(payload.get("x"))
        if payload.get("top") is not None:
            y = _safe_float(payload.get("top"))
        else:
            y = _safe_float(payload.get("y"))
        if bool(payload.get("coordinates_are_center")):
            x -= width / 2.0
            y -= height / 2.0
        return cls(
            x=x,
            y=y,
            width=width,
            height=height,
            detail_type=(
                str(payload.get("detail_type") or "").strip().lower()
                or None
            ),
            score=(
                _safe_float(payload.get("score"))
                if payload.get("score") is not None
                else None
            ),
            stage=str(payload.get("stage") or "").strip() or None,
            candidate_id=(
                str(payload.get("candidate_id") or "").strip()
                or None
            ),
        )


def intersection_over_union(
    left: EvaluationBox,
    right: EvaluationBox,
) -> float:
    x0 = max(left.x, right.x)
    y0 = max(left.y, right.y)
    x1 = min(left.x + left.width, right.x + right.width)
    y1 = min(left.y + left.height, right.y + right.height)
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (
        left.width * left.height
        + right.width * right.height
        - intersection
    )
    return intersection / union if union > 0.0 else 0.0


def center_distance(
    left: EvaluationBox,
    right: EvaluationBox,
) -> float:
    left_x, left_y = left.center
    right_x, right_y = right.center
    return math.hypot(left_x - right_x, left_y - right_y)


def match_boxes(
    truth: Sequence[EvaluationBox],
    candidates: Sequence[EvaluationBox],
    *,
    iou_threshold: float = 0.1,
    center_distance_threshold: float = 8.0,
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    """Greedy one-to-one matching with deterministic tie breaking."""

    possible: list[tuple[float, float, int, int]] = []
    for truth_index, truth_box in enumerate(truth):
        for candidate_index, candidate_box in enumerate(candidates):
            iou = intersection_over_union(truth_box, candidate_box)
            distance = center_distance(truth_box, candidate_box)
            if iou >= iou_threshold or distance <= center_distance_threshold:
                possible.append(
                    (
                        -iou,
                        distance,
                        truth_index,
                        candidate_index,
                    )
                )
    possible.sort()
    used_truth: set[int] = set()
    used_candidates: set[int] = set()
    matches: list[dict[str, Any]] = []
    for negative_iou, distance, truth_index, candidate_index in possible:
        if truth_index in used_truth or candidate_index in used_candidates:
            continue
        used_truth.add(truth_index)
        used_candidates.add(candidate_index)
        matches.append(
            {
                "truth_index": truth_index,
                "candidate_index": candidate_index,
                "iou": -negative_iou,
                "center_distance": distance,
            }
        )
    return (
        matches,
        sorted(set(range(len(truth))) - used_truth),
        sorted(set(range(len(candidates))) - used_candidates),
    )


def evaluate_task(
    payload: Mapping[str, Any],
    *,
    iou_threshold: float = 0.1,
    center_distance_threshold: float = 8.0,
) -> dict[str, Any]:
    truth = [
        EvaluationBox.from_mapping(item)
        for item in payload.get("truth") or []
        if isinstance(item, Mapping)
    ]
    candidates = [
        EvaluationBox.from_mapping(item)
        for item in payload.get("candidates") or []
        if isinstance(item, Mapping)
    ]
    matches, missed_truth, unmatched_candidates = match_boxes(
        truth,
        candidates,
        iou_threshold=iou_threshold,
        center_distance_threshold=center_distance_threshold,
    )
    matched_truth = {
        int(item["truth_index"]): int(item["candidate_index"])
        for item in matches
    }
    missed_by_detail: Counter[str] = Counter()
    matched_by_detail: Counter[str] = Counter()
    for index, box in enumerate(truth):
        detail_type = box.detail_type or "unknown"
        if index in matched_truth:
            matched_by_detail[detail_type] += 1
        else:
            missed_by_detail[detail_type] += 1

    trace = payload.get("trace")
    stage_counts = (
        dict(trace.get("stage_counts") or {})
        if isinstance(trace, Mapping)
        else {}
    )
    duration_ms = (
        _safe_float(trace.get("duration_ms"))
        if isinstance(trace, Mapping)
        else _safe_float(payload.get("duration_ms"))
    )
    post_filter_count = int(
        payload.get("post_filter_candidate_count")
        if payload.get("post_filter_candidate_count") is not None
        else len(candidates)
    )
    raw_count = int(
        payload.get("raw_candidate_count")
        if payload.get("raw_candidate_count") is not None
        else max(
            [int(value) for value in stage_counts.values()]
            + [len(candidates)]
        )
    )
    return {
        "task_id": str(payload.get("task_id") or ""),
        "truth_count": len(truth),
        "matched_truth_count": len(matches),
        "missed_truth_count": len(missed_truth),
        "candidate_count": len(candidates),
        "raw_candidate_count": raw_count,
        "post_filter_candidate_count": post_filter_count,
        "false_positive_count": len(unmatched_candidates),
        "recall": (
            len(matches) / len(truth)
            if truth
            else None
        ),
        "precision": (
            len(matches) / len(candidates)
            if candidates
            else None
        ),
        "duration_ms": duration_ms,
        "stage_counts": stage_counts,
        "matches": matches,
        "missed_truth_indices": missed_truth,
        "unmatched_candidate_indices": unmatched_candidates,
        "matched_detail_type_counts": dict(sorted(matched_by_detail.items())),
        "missed_detail_type_counts": dict(sorted(missed_by_detail.items())),
    }


def evaluate_candidate_records(
    tasks: Iterable[Mapping[str, Any]],
    *,
    iou_threshold: float = 0.1,
    center_distance_threshold: float = 8.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_task = [
        evaluate_task(
            task,
            iou_threshold=iou_threshold,
            center_distance_threshold=center_distance_threshold,
        )
        for task in tasks
    ]
    truth_count = sum(int(item["truth_count"]) for item in per_task)
    matched_count = sum(
        int(item["matched_truth_count"])
        for item in per_task
    )
    candidate_count = sum(
        int(item["candidate_count"])
        for item in per_task
    )
    matched_details: Counter[str] = Counter()
    missed_details: Counter[str] = Counter()
    for item in per_task:
        matched_details.update(item["matched_detail_type_counts"])
        missed_details.update(item["missed_detail_type_counts"])
    detail_metrics: dict[str, Any] = {}
    for detail_type in sorted(set(matched_details) | set(missed_details)):
        matched = int(matched_details[detail_type])
        missed = int(missed_details[detail_type])
        support = matched + missed
        detail_metrics[detail_type] = {
            "support": support,
            "matched": matched,
            "missed": missed,
            "recall": matched / support if support else None,
            "supported": support > 0,
        }
    durations = [
        float(item["duration_ms"])
        for item in per_task
        if item["duration_ms"] is not None
    ]
    raw_counts = [
        float(item["raw_candidate_count"])
        for item in per_task
    ]
    post_counts = [
        float(item["post_filter_candidate_count"])
        for item in per_task
    ]
    metrics = {
        "evaluator_version": CANDIDATE_EVALUATOR_VERSION,
        "task_count": len(per_task),
        "truth_count": truth_count,
        "matched_truth_count": matched_count,
        "candidate_count": candidate_count,
        "recall": matched_count / truth_count if truth_count else None,
        "precision": matched_count / candidate_count if candidate_count else None,
        "iou_threshold": float(iou_threshold),
        "center_distance_threshold": float(center_distance_threshold),
        "raw_candidates_per_task": {
            "p50": _percentile(raw_counts, 50.0),
            "p95": _percentile(raw_counts, 95.0),
        },
        "post_filter_candidates_per_task": {
            "p50": _percentile(post_counts, 50.0),
            "p95": _percentile(post_counts, 95.0),
        },
        "duration_ms": {
            "p50": _percentile(durations, 50.0),
            "p95": _percentile(durations, 95.0),
        },
        "detail_type_metrics": detail_metrics,
    }
    return metrics, per_task


def write_evaluation_artifact(
    output_root: Path,
    *,
    run_id: str,
    run_type: str,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    per_task: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    run_path = (output_root / str(run_id)).resolve()
    try:
        run_path.relative_to(output_root)
    except ValueError as exc:
        raise ValueError("evaluation run path escapes output root") from exc
    run_path.mkdir(parents=True, exist_ok=False)

    metrics_bytes = (
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    per_task_bytes = b"".join(
        (
            json.dumps(item, ensure_ascii=False, sort_keys=True)
            + "\n"
        ).encode("utf-8")
        for item in per_task
    )
    (run_path / "metrics.json").write_bytes(metrics_bytes)
    (run_path / "per_task.jsonl").write_bytes(per_task_bytes)
    base_manifest = {
        "version": "1.0",
        "run_id": str(run_id),
        "run_type": str(run_type),
        "created_at": _utc_now_iso(),
        "evaluator_version": CANDIDATE_EVALUATOR_VERSION,
        "config": dict(config),
        "provenance": dict(provenance or {}),
        "files": {
            "metrics.json": {
                "sha256": _sha256_bytes(metrics_bytes),
                "size_bytes": len(metrics_bytes),
            },
            "per_task.jsonl": {
                "sha256": _sha256_bytes(per_task_bytes),
                "size_bytes": len(per_task_bytes),
            },
        },
    }
    manifest_sha256 = _sha256_bytes(
        _canonical_json(base_manifest).encode("utf-8")
    )
    manifest = {
        **base_manifest,
        "manifest_sha256": manifest_sha256,
    }
    manifest_path = run_path / "manifest.json"
    temp_path = run_path / f".manifest.{uuid.uuid4().hex}.tmp"
    temp_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, manifest_path)
    return {
        "run_path": str(run_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "metrics": dict(metrics),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate candidate-stage recall and candidate volume",
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--run-type", default="candidate")
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    parser.add_argument("--center-distance", type=float, default=8.0)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    document = json.loads(args.input.read_text(encoding="utf-8"))
    tasks = document.get("tasks") if isinstance(document, Mapping) else None
    if not isinstance(tasks, list):
        raise ValueError("evaluation input must contain a tasks list")
    metrics, per_task = evaluate_candidate_records(
        tasks,
        iou_threshold=args.iou_threshold,
        center_distance_threshold=args.center_distance,
    )
    run_id = str(args.run_id or f"evaluation-{uuid.uuid4().hex[:16]}")
    result = write_evaluation_artifact(
        args.output_root,
        run_id=run_id,
        run_type=args.run_type,
        config={
            "iou_threshold": args.iou_threshold,
            "center_distance_threshold": args.center_distance,
        },
        metrics=metrics,
        per_task=per_task,
        provenance=document.get("provenance") or {},
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
