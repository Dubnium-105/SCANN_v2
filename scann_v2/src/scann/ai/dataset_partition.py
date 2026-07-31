"""Deterministic, group-isolated train/validation/gold-test partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from scann.ai.taxonomy import TAXONOMY_VERSION, normalize_detail_type


PARTITION_MANIFEST_VERSION = "1.0"
PARTITION_STRATEGY = "group_stratified_v1"
SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "test")
_TASK_DATE_RE = re.compile(r"^(?P<date>\d{8})T\d{6}__")


@dataclass(frozen=True)
class PartitionTask:
    task_id: str
    field_key: str
    capture_key: str
    date_obs: str | None
    night_key: str
    group_key: str
    annotation_count: int
    detail_type_counts: dict[str, int] = field(default_factory=dict)

    def to_manifest_entry(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "field_key": self.field_key,
            "capture_key": self.capture_key,
            "date_obs": self.date_obs,
            "night_key": self.night_key,
            "group_key": self.group_key,
            "annotation_count": int(self.annotation_count),
            "detail_type_counts": dict(sorted(self.detail_type_counts.items())),
        }


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def derive_night_key(date_obs: Any, task_id: Any) -> str:
    raw_date_obs = str(date_obs or "").strip()
    if raw_date_obs:
        normalized = raw_date_obs.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized).date().isoformat()
        except ValueError:
            if len(raw_date_obs) >= 10:
                prefix = raw_date_obs[:10]
                try:
                    return datetime.strptime(prefix, "%Y-%m-%d").date().isoformat()
                except ValueError:
                    pass

    match = _TASK_DATE_RE.match(str(task_id or "").strip())
    if match is not None:
        return datetime.strptime(match.group("date"), "%Y%m%d").date().isoformat()
    return "unknown"


def derive_group_key(
    *,
    task_id: Any,
    field_key: Any,
    capture_key: Any,
    night_key: Any,
) -> str:
    normalized_task_id = str(task_id or "").strip()
    normalized_field = str(field_key or "").strip()
    normalized_capture = str(capture_key or "").strip()
    normalized_night = str(night_key or "").strip()
    if normalized_night and normalized_night != "unknown" and normalized_field:
        return f"night:{normalized_night}|field:{normalized_field}"
    if normalized_capture:
        return f"capture:{normalized_capture}"
    if normalized_field:
        return f"field:{normalized_field}"
    return f"task:{normalized_task_id}"


def partition_task_from_image(image: Mapping[str, Any]) -> PartitionTask | None:
    task_id = str(
        image.get("task_id")
        or image.get("id")
        or image.get("file_name")
        or ""
    ).strip()
    if not task_id:
        return None
    field_key = str(image.get("field_key") or "").strip()
    capture_key = str(image.get("capture_key") or "").strip()
    date_obs = str(image.get("date_obs") or "").strip() or None
    night_key = derive_night_key(date_obs, task_id)
    group_key = derive_group_key(
        task_id=task_id,
        field_key=field_key,
        capture_key=capture_key,
        night_key=night_key,
    )
    counts: Counter[str] = Counter()
    annotations = image.get("annotations") or []
    if not isinstance(annotations, list):
        annotations = []
    for annotation in annotations:
        if not isinstance(annotation, Mapping):
            continue
        detail_type = normalize_detail_type(annotation.get("detail_type"))
        if detail_type is not None:
            counts[detail_type.value] += 1
    return PartitionTask(
        task_id=task_id,
        field_key=field_key,
        capture_key=capture_key,
        date_obs=date_obs,
        night_key=night_key,
        group_key=group_key,
        annotation_count=int(sum(counts.values())),
        detail_type_counts=dict(counts),
    )


def _normalize_ratios(
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> dict[str, float]:
    raw = {
        "train": float(train_ratio),
        "validation": float(validation_ratio),
        "test": float(test_ratio),
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in raw.values()):
        raise ValueError("partition ratios must be finite and greater than zero")
    total = sum(raw.values())
    return {name: value / total for name, value in raw.items()}


def _group_tasks(tasks: list[PartitionTask]) -> list[dict[str, Any]]:
    grouped: dict[str, list[PartitionTask]] = defaultdict(list)
    for task in tasks:
        grouped[task.group_key].append(task)
    result: list[dict[str, Any]] = []
    for group_key, group_tasks in grouped.items():
        class_counts: Counter[str] = Counter()
        for task in group_tasks:
            class_counts.update(task.detail_type_counts)
        result.append(
            {
                "group_key": group_key,
                "tasks": sorted(group_tasks, key=lambda item: item.task_id),
                "task_count": len(group_tasks),
                "annotation_count": sum(
                    int(item.annotation_count)
                    for item in group_tasks
                ),
                "class_counts": dict(class_counts),
            }
        )
    return result


def _group_tiebreaker(seed: int, group_key: str) -> str:
    return hashlib.sha256(f"{int(seed)}:{group_key}".encode("utf-8")).hexdigest()


def _assignment_cost(
    *,
    split_name: str,
    group: Mapping[str, Any],
    ratios: Mapping[str, float],
    total_tasks: int,
    total_class_counts: Mapping[str, int],
    split_task_counts: Mapping[str, int],
    split_class_counts: Mapping[str, Counter[str]],
) -> float:
    target_tasks = max(float(total_tasks) * float(ratios[split_name]), 1.0)
    projected_tasks = (
        float(split_task_counts[split_name])
        + float(group["task_count"])
    )
    size_fill = projected_tasks / target_tasks

    group_class_counts = group["class_counts"]
    class_fills: list[float] = []
    for detail_type, group_count in group_class_counts.items():
        target_class = max(
            float(total_class_counts.get(detail_type, 0))
            * float(ratios[split_name]),
            1.0,
        )
        projected_class = (
            float(split_class_counts[split_name].get(detail_type, 0))
            + float(group_count)
        )
        class_fills.append(projected_class / target_class)
    class_fill = (
        sum(value * value for value in class_fills) / len(class_fills)
        if class_fills
        else size_fill * size_fill
    )
    overfill = max(0.0, size_fill - 1.0)
    return (
        0.45 * size_fill * size_fill
        + 0.55 * class_fill
        + 4.0 * overfill * overfill
    )


def _split_summary(
    groups: list[Mapping[str, Any]],
) -> dict[str, Any]:
    class_counts: Counter[str] = Counter()
    task_count = 0
    annotation_count = 0
    for group in groups:
        task_count += int(group["task_count"])
        annotation_count += int(group["annotation_count"])
        class_counts.update(group["class_counts"])
    return {
        "task_count": task_count,
        "group_count": len(groups),
        "annotation_count": annotation_count,
        "detail_type_counts": dict(sorted(class_counts.items())),
    }


def _ensure_nonempty_splits(
    assignments: dict[str, list[dict[str, Any]]],
    *,
    forced_train_groups: set[str],
) -> None:
    for split_name in ("validation", "test"):
        if assignments[split_name]:
            continue
        candidates = [
            group
            for group in assignments["train"]
            if str(group["group_key"]) not in forced_train_groups
        ]
        if not candidates:
            continue
        moved = min(
            candidates,
            key=lambda group: (
                int(group["task_count"]),
                int(group["annotation_count"]),
                str(group["group_key"]),
            ),
        )
        assignments["train"].remove(moved)
        assignments[split_name].append(moved)


def build_partition_manifest(
    images: Iterable[Mapping[str, Any]],
    *,
    partition_name: str,
    seed: int = 42,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, Any]:
    normalized_name = str(partition_name or "").strip()
    if not normalized_name:
        raise ValueError("partition name is required")
    ratios = _normalize_ratios(train_ratio, validation_ratio, test_ratio)

    tasks_by_id: dict[str, PartitionTask] = {}
    for image in images:
        if not isinstance(image, Mapping):
            continue
        task = partition_task_from_image(image)
        if task is None or task.annotation_count <= 0:
            continue
        if task.task_id in tasks_by_id:
            raise ValueError(f"duplicate task in partition input: {task.task_id}")
        tasks_by_id[task.task_id] = task
    tasks = sorted(tasks_by_id.values(), key=lambda item: item.task_id)
    if len(tasks) < 3:
        raise ValueError("partition requires at least 3 annotated tasks")

    groups = _group_tasks(tasks)
    if len(groups) < 3:
        raise ValueError("partition requires at least 3 independent groups")

    total_class_counts: Counter[str] = Counter()
    class_group_counts: Counter[str] = Counter()
    class_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        total_class_counts.update(group["class_counts"])
        for detail_type in group["class_counts"]:
            class_group_counts[detail_type] += 1
            class_groups[detail_type].append(group)

    forced_train_groups: set[str] = set()
    for detail_type in sorted(class_groups):
        representative = max(
            class_groups[detail_type],
            key=lambda group: (
                int(group["class_counts"].get(detail_type, 0)),
                -int(group["task_count"]),
                str(group["group_key"]),
            ),
        )
        forced_train_groups.add(str(representative["group_key"]))

    rng = random.Random(int(seed))
    random_noise = {
        str(group["group_key"]): rng.random()
        for group in sorted(groups, key=lambda item: str(item["group_key"]))
    }

    def rarity_score(group: Mapping[str, Any]) -> float:
        return sum(
            float(count) / max(1.0, float(total_class_counts[detail_type]))
            for detail_type, count in group["class_counts"].items()
        )

    ordered_groups = sorted(
        groups,
        key=lambda group: (
            0 if str(group["group_key"]) in forced_train_groups else 1,
            -rarity_score(group),
            -int(group["annotation_count"]),
            random_noise[str(group["group_key"])],
            _group_tiebreaker(int(seed), str(group["group_key"])),
        ),
    )

    assignments: dict[str, list[dict[str, Any]]] = {
        split_name: []
        for split_name in SPLIT_NAMES
    }
    split_task_counts = {split_name: 0 for split_name in SPLIT_NAMES}
    split_class_counts = {
        split_name: Counter()
        for split_name in SPLIT_NAMES
    }

    for group in ordered_groups:
        group_key = str(group["group_key"])
        if group_key in forced_train_groups:
            split_name = "train"
        else:
            split_name = min(
                SPLIT_NAMES,
                key=lambda candidate: (
                    _assignment_cost(
                        split_name=candidate,
                        group=group,
                        ratios=ratios,
                        total_tasks=len(tasks),
                        total_class_counts=total_class_counts,
                        split_task_counts=split_task_counts,
                        split_class_counts=split_class_counts,
                    ),
                    SPLIT_NAMES.index(candidate),
                ),
            )
        assignments[split_name].append(group)
        split_task_counts[split_name] += int(group["task_count"])
        split_class_counts[split_name].update(group["class_counts"])

    _ensure_nonempty_splits(
        assignments,
        forced_train_groups=forced_train_groups,
    )

    split_payload: dict[str, list[dict[str, Any]]] = {}
    split_summaries: dict[str, dict[str, Any]] = {}
    for split_name in SPLIT_NAMES:
        split_groups = sorted(
            assignments[split_name],
            key=lambda group: str(group["group_key"]),
        )
        split_payload[split_name] = sorted(
            [
                task.to_manifest_entry()
                for group in split_groups
                for task in group["tasks"]
            ],
            key=lambda item: item["task_id"],
        )
        split_summaries[split_name] = _split_summary(split_groups)

    group_sets = {
        split_name: {
            item["group_key"]
            for item in split_payload[split_name]
        }
        for split_name in SPLIT_NAMES
    }
    overlap = sorted(
        (group_sets["train"] & group_sets["validation"])
        | (group_sets["train"] & group_sets["test"])
        | (group_sets["validation"] & group_sets["test"])
    )
    if overlap:
        raise RuntimeError("partition group overlap detected")

    warnings: list[str] = []
    limited_group_classes = {
        detail_type: int(count)
        for detail_type, count in sorted(class_group_counts.items())
        if int(count) < 3
    }
    if limited_group_classes:
        warnings.append("classes_with_fewer_than_three_independent_groups")

    base_manifest: dict[str, Any] = {
        "version": PARTITION_MANIFEST_VERSION,
        "partition_name": normalized_name,
        "strategy": PARTITION_STRATEGY,
        "seed": int(seed),
        "taxonomy_version": TAXONOMY_VERSION,
        "ratios": ratios,
        "splits": split_payload,
        "audit": {
            "task_count": len(tasks),
            "group_count": len(groups),
            "annotation_count": sum(
                int(task.annotation_count)
                for task in tasks
            ),
            "detail_type_counts": dict(sorted(total_class_counts.items())),
            "class_independent_group_counts": dict(
                sorted(class_group_counts.items())
            ),
            "limited_group_classes": limited_group_classes,
            "split_summaries": split_summaries,
            "group_overlap": overlap,
            "warnings": warnings,
        },
    }
    content_hash = _sha256_payload(base_manifest)
    manifest = {
        "partition_id": f"partition-{content_hash[:16]}",
        **base_manifest,
    }
    manifest["manifest_sha256"] = _sha256_payload(manifest)
    return manifest


def verify_partition_manifest(manifest: Mapping[str, Any]) -> bool:
    expected = str(manifest.get("manifest_sha256") or "").strip().lower()
    if not expected:
        return False
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    return _sha256_payload(payload) == expected


def task_ids_for_splits(
    manifest: Mapping[str, Any],
    split_names: Iterable[str],
) -> list[str]:
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("partition manifest does not contain splits")
    result: list[str] = []
    seen: set[str] = set()
    for raw_split_name in split_names:
        split_name = str(raw_split_name).strip().lower()
        if split_name not in SPLIT_NAMES:
            raise ValueError(f"unknown partition split: {split_name}")
        items = splits.get(split_name)
        if not isinstance(items, list):
            raise ValueError(f"partition split is invalid: {split_name}")
        for item in items:
            if not isinstance(item, Mapping):
                continue
            task_id = str(item.get("task_id") or "").strip()
            if task_id and task_id not in seen:
                seen.add(task_id)
                result.append(task_id)
    return result


def plan_dataset_partition(
    dataset_root: Path,
    *,
    partition_name: str,
    db_path: Path | None = None,
    seed: int = 42,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, Any]:
    from scann.core.dataset_storage import DatasetStorage

    storage = DatasetStorage(dataset_root, db_path=db_path)
    images = list(storage.list_current_annotations().values())
    return build_partition_manifest(
        images,
        partition_name=partition_name,
        seed=seed,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a deterministic SCANN dataset partition manifest",
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--db-path", type=Path)
    parser.add_argument("--partition-name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    manifest = plan_dataset_partition(
        args.dataset_root,
        partition_name=args.partition_name,
        db_path=args.db_path,
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
    )
    payload = json.dumps(manifest, ensure_ascii=False, indent=2)
    if args.output is not None:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    if not args.quiet:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
