"""Class-balance helpers for SCANN detail-type classification."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np

from scann.core.annotation_models import AnnotationLabel, DETAIL_TYPE_TO_LABEL, DetailType


DETAIL_TYPE_CLASS_ORDER: tuple[str, ...] = (
    DetailType.ASTEROID.value,
    DetailType.SUPERNOVA.value,
    DetailType.VARIABLE_STAR.value,
    DetailType.SATELLITE_TRAIL.value,
    DetailType.NOISE.value,
    DetailType.DIFFRACTION_SPIKE.value,
    DetailType.CMOS_CONDENSATION.value,
    DetailType.CORRESPONDING.value,
    DetailType.DISAPPEARED_ASTEROID.value,
    DetailType.DISAPPEARED_STAR.value,
    DetailType.DISAPPEARED_GALAXY.value,
)

DETAIL_TYPE_TO_CLASS_INDEX: dict[str, int] = {
    detail_type: index
    for index, detail_type in enumerate(DETAIL_TYPE_CLASS_ORDER)
}

DETAIL_TYPE_TO_BUCKET: dict[str, str] = {
    detail_type.value: DETAIL_TYPE_TO_LABEL[detail_type].value
    for detail_type in DetailType
}

DEFAULT_IMBALANCE_CONFIG: dict[str, Any] = {
    "imbalance_strategy": "class_balanced_focal",
    "class_weight_beta": 0.999,
    "class_weight_clip": [0.25, 5.0],
    "sampler_power": 0.5,
    "sampler_max_ratio": 4.0,
    "min_train_support_warning": 5,
    "min_val_support_warning": 1,
    "selection_metric": "macro_f1_supported",
}


def normalize_detail_type(value: Any) -> str | None:
    if isinstance(value, DetailType):
        return value.value
    normalized = str(value or "").strip().lower()
    if not normalized:
        return None
    try:
        return DetailType(normalized).value
    except Exception:
        return normalized if normalized in DETAIL_TYPE_TO_CLASS_INDEX else None


def label_index_for_detail_type(value: Any) -> int | None:
    detail_type = normalize_detail_type(value)
    if detail_type is None:
        return None
    return DETAIL_TYPE_TO_CLASS_INDEX.get(detail_type)


def bucket_for_detail_type(value: Any) -> str | None:
    detail_type = normalize_detail_type(value)
    if detail_type is None:
        return None
    return DETAIL_TYPE_TO_BUCKET.get(detail_type)


def merge_imbalance_config(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(DEFAULT_IMBALANCE_CONFIG)
    if isinstance(raw, dict):
        for key in DEFAULT_IMBALANCE_CONFIG:
            if key in raw:
                merged[key] = raw[key]
    clip = merged.get("class_weight_clip")
    if not isinstance(clip, (list, tuple)) or len(clip) != 2:
        clip = DEFAULT_IMBALANCE_CONFIG["class_weight_clip"]
    lo, hi = float(clip[0]), float(clip[1])
    if not math.isfinite(lo) or not math.isfinite(hi) or lo <= 0 or hi < lo:
        lo, hi = DEFAULT_IMBALANCE_CONFIG["class_weight_clip"]
    merged["class_weight_clip"] = [float(lo), float(hi)]
    merged["class_weight_beta"] = float(merged.get("class_weight_beta") or 0.999)
    merged["sampler_power"] = max(0.0, float(merged.get("sampler_power") or 0.5))
    merged["sampler_max_ratio"] = max(1.0, float(merged.get("sampler_max_ratio") or 4.0))
    merged["min_train_support_warning"] = max(0, int(merged.get("min_train_support_warning") or 0))
    merged["min_val_support_warning"] = max(0, int(merged.get("min_val_support_warning") or 0))
    return merged


def sample_records_from_snapshot_document(document: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    images = document.get("images", []) if isinstance(document, dict) else []
    if not isinstance(images, list):
        return records

    for image_info in images:
        if not isinstance(image_info, dict):
            continue
        task_id = str(image_info.get("id") or image_info.get("file_name") or image_info.get("file") or "").strip()
        image_detail_type = normalize_detail_type(image_info.get("detail_type"))
        annotations = image_info.get("annotations") or []
        if not isinstance(annotations, list):
            continue
        for ann_index, ann in enumerate(annotations):
            if not isinstance(ann, dict):
                continue
            detail_type = normalize_detail_type(ann.get("detail_type")) or image_detail_type
            label_index = label_index_for_detail_type(detail_type)
            if detail_type is None or label_index is None:
                continue
            records.append(
                {
                    "task_id": task_id or f"image-{len(records)}",
                    "annotation_index": int(ann_index),
                    "detail_type": detail_type,
                    "label": int(label_index),
                    "bucket": bucket_for_detail_type(detail_type),
                }
            )
    return records


def _counts_by_label(records: Iterable[dict[str, Any]], *, key: str = "label") -> list[int]:
    counts = [0] * len(DETAIL_TYPE_CLASS_ORDER)
    for record in records:
        try:
            label = int(record.get(key))
        except Exception:
            continue
        if 0 <= label < len(counts):
            counts[label] += 1
    return counts


def _bucket_counts_from_class_counts(class_counts: list[int]) -> dict[str, int]:
    counts = {AnnotationLabel.REAL.value: 0, AnnotationLabel.BOGUS.value: 0}
    for index, count in enumerate(class_counts):
        detail_type = DETAIL_TYPE_CLASS_ORDER[index]
        bucket = DETAIL_TYPE_TO_BUCKET.get(detail_type)
        if bucket:
            counts[bucket] = counts.get(bucket, 0) + int(count)
    return counts


def build_class_audit(
    records: Iterable[dict[str, Any]],
    *,
    split_support: dict[str, list[int]] | None = None,
    min_train_support: int = 5,
    min_val_support: int = 1,
) -> dict[str, Any]:
    record_list = list(records)
    class_counts = _counts_by_label(record_list)
    supported_counts = [count for count in class_counts if count > 0]
    imbalance_ratio = (
        float(max(supported_counts)) / float(min(supported_counts))
        if supported_counts
        else 0.0
    )
    per_class = {
        detail_type: {
            "index": index,
            "total": int(class_counts[index]),
            "bucket": DETAIL_TYPE_TO_BUCKET.get(detail_type),
        }
        for index, detail_type in enumerate(DETAIL_TYPE_CLASS_ORDER)
    }
    for split_name, support in (split_support or {}).items():
        for index, detail_type in enumerate(DETAIL_TYPE_CLASS_ORDER):
            per_class[detail_type][split_name] = int(support[index]) if index < len(support) else 0

    missing_classes = [
        detail_type
        for index, detail_type in enumerate(DETAIL_TYPE_CLASS_ORDER)
        if class_counts[index] <= 0
    ]
    low_sample_classes = [
        detail_type
        for index, detail_type in enumerate(DETAIL_TYPE_CLASS_ORDER)
        if 0 < class_counts[index] < int(min_train_support)
    ]

    train_support = (split_support or {}).get("train")
    val_support = (split_support or {}).get("val")
    promotion_warnings: list[str] = []
    if train_support is not None:
        low_train = [
            DETAIL_TYPE_CLASS_ORDER[index]
            for index, count in enumerate(train_support)
            if int(count) < int(min_train_support)
        ]
        if low_train:
            promotion_warnings.append(
                "train_support_below_minimum: " + ", ".join(low_train)
            )
    if val_support is not None:
        low_val = [
            DETAIL_TYPE_CLASS_ORDER[index]
            for index, count in enumerate(val_support)
            if int(count) < int(min_val_support)
        ]
        if low_val:
            promotion_warnings.append(
                "val_support_below_minimum: " + ", ".join(low_val)
            )
    untrained_classes = (
        [
            DETAIL_TYPE_CLASS_ORDER[index]
            for index, count in enumerate(train_support)
            if int(count) <= 0
        ]
        if train_support is not None
        else []
    )
    unverifiable_classes = (
        [
            DETAIL_TYPE_CLASS_ORDER[index]
            for index, count in enumerate(val_support)
            if int(count) <= 0
        ]
        if val_support is not None
        else []
    )

    return {
        "class_names": list(DETAIL_TYPE_CLASS_ORDER),
        "total_samples": int(sum(class_counts)),
        "class_counts": {
            detail_type: int(class_counts[index])
            for index, detail_type in enumerate(DETAIL_TYPE_CLASS_ORDER)
        },
        "bucket_counts": _bucket_counts_from_class_counts(class_counts),
        "per_class": per_class,
        "missing_classes": missing_classes,
        "low_sample_classes": low_sample_classes,
        "untrained_classes": untrained_classes,
        "unverifiable_classes": unverifiable_classes,
        "imbalance_ratio": float(imbalance_ratio),
        "split_support": {
            split_name: {
                DETAIL_TYPE_CLASS_ORDER[index]: int(count)
                for index, count in enumerate(support)
            }
            for split_name, support in (split_support or {}).items()
        },
        "promotion_warnings": promotion_warnings,
    }


def stratified_group_train_val_split(
    records: list[dict[str, Any]],
    *,
    val_split: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int], dict[str, list[int]]]:
    if len(records) < 2:
        raise ValueError("training requires at least 2 samples")

    class_totals = _counts_by_label(records)
    target_val = [
        0 if count < 2 else max(1, int(round(float(count) * float(val_split))))
        for count in class_totals
    ]
    target_val = [min(target, max(0, class_totals[index] - 1)) for index, target in enumerate(target_val)]

    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        group_key = str(record.get("task_id") or f"sample-{index}")
        groups[group_key].append(index)

    group_items = []
    for group_key, indices in groups.items():
        counts = Counter(int(records[index]["label"]) for index in indices)
        group_items.append((group_key, indices, counts))

    rng = random.Random(int(seed))
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: (len(item[1]), len(item[2]), item[0]))

    selected_groups: set[str] = set()
    val_counts = [0] * len(DETAIL_TYPE_CLASS_ORDER)

    def can_select(counts: Counter[int]) -> bool:
        for label, count in counts.items():
            if class_totals[label] < 2:
                return False
            if val_counts[label] + int(count) > class_totals[label] - 1:
                return False
        return True

    def select_group(group_key: str, counts: Counter[int]) -> None:
        selected_groups.add(group_key)
        for label, count in counts.items():
            val_counts[label] += int(count)

    for label in sorted(range(len(DETAIL_TYPE_CLASS_ORDER)), key=lambda item: class_totals[item]):
        while val_counts[label] < target_val[label]:
            candidates = [
                item
                for item in group_items
                if item[0] not in selected_groups and item[2].get(label, 0) > 0 and can_select(item[2])
            ]
            if not candidates:
                break
            best = min(
                candidates,
                key=lambda item: (
                    sum(max(0, val_counts[k] + item[2].get(k, 0) - target_val[k]) for k in item[2]),
                    len(item[1]),
                    item[0],
                ),
            )
            select_group(best[0], best[2])

    target_total = max(1, int(round(len(records) * float(val_split))))
    while sum(val_counts) < target_total:
        candidates = [
            item
            for item in group_items
            if item[0] not in selected_groups and can_select(item[2])
        ]
        if not candidates:
            break
        best = min(
            candidates,
            key=lambda item: (
                sum(max(0, val_counts[k] + item[2].get(k, 0) - target_val[k]) for k in item[2]),
                len(item[1]),
                item[0],
            ),
        )
        select_group(best[0], best[2])

    val_indices = sorted(
        index
        for group_key, indices, _counts in group_items
        if group_key in selected_groups
        for index in indices
    )
    train_indices = sorted(set(range(len(records))) - set(val_indices))
    if not train_indices:
        train_indices = [val_indices.pop()]

    split_support = {
        "train": _counts_by_label(records[index] for index in train_indices),
        "val": _counts_by_label(records[index] for index in val_indices),
    }
    return train_indices, val_indices, split_support


def compute_class_balanced_weights(
    labels: Iterable[int],
    *,
    beta: float = 0.999,
    clip: tuple[float, float] | list[float] = (0.25, 5.0),
    class_count: int = len(DETAIL_TYPE_CLASS_ORDER),
) -> list[float]:
    counts = [0] * int(class_count)
    for raw_label in labels:
        label = int(raw_label)
        if 0 <= label < class_count:
            counts[label] += 1

    beta = min(max(float(beta), 0.0), 0.999999)
    raw_weights = [0.0] * class_count
    for index, count in enumerate(counts):
        if count <= 0:
            continue
        effective_num = 1.0 - (beta ** count)
        raw_weights[index] = (1.0 - beta) / max(effective_num, 1e-12)

    supported = [value for value in raw_weights if value > 0.0]
    if not supported:
        return [0.0] * class_count
    mean_weight = float(sum(supported)) / float(len(supported))
    min_clip, max_clip = float(clip[0]), float(clip[1])
    weights = []
    for value in raw_weights:
        if value <= 0.0:
            weights.append(0.0)
        else:
            weights.append(float(min(max(value / mean_weight, min_clip), max_clip)))
    return weights


def sampler_weights_from_class_weights(
    labels: Iterable[int],
    class_weights: list[float],
    *,
    power: float = 0.5,
    max_ratio: float = 4.0,
) -> list[float]:
    weights = []
    for raw_label in labels:
        label = int(raw_label)
        base = class_weights[label] if 0 <= label < len(class_weights) else 1.0
        weights.append(max(float(base), 1e-12) ** float(power))
    if not weights:
        return weights
    positive = [value for value in weights if value > 0.0]
    if not positive:
        return [1.0] * len(weights)
    low = min(positive)
    high = low * max(1.0, float(max_ratio))
    return [float(min(max(value, low), high)) for value in weights]


def compute_multiclass_metrics(
    labels: Iterable[int],
    preds: Iterable[int],
    *,
    class_count: int = len(DETAIL_TYPE_CLASS_ORDER),
) -> dict[str, Any]:
    label_arr = np.asarray(list(labels), dtype=np.int64)
    pred_arr = np.asarray(list(preds), dtype=np.int64)
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    for true_label, pred_label in zip(label_arr, pred_arr):
        if 0 <= int(true_label) < class_count and 0 <= int(pred_label) < class_count:
            confusion[int(true_label), int(pred_label)] += 1

    per_class: dict[str, dict[str, float | int | str | None]] = {}
    f1_values: list[float] = []
    for index, detail_type in enumerate(DETAIL_TYPE_CLASS_ORDER):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index, :].sum() - tp)
        support = int(confusion[index, :].sum())
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = (2.0 * precision * recall) / (precision + recall + 1e-12)
        if support > 0:
            f1_values.append(float(f1))
        per_class[detail_type] = {
            "index": int(index),
            "bucket": DETAIL_TYPE_TO_BUCKET.get(detail_type),
            "support": support,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    accuracy = float((label_arr == pred_arr).mean()) if label_arr.size else 0.0
    bucket_recalls: dict[str, float] = {}
    for bucket in (AnnotationLabel.REAL.value, AnnotationLabel.BOGUS.value):
        bucket_indices = {
            index
            for index, detail_type in enumerate(DETAIL_TYPE_CLASS_ORDER)
            if DETAIL_TYPE_TO_BUCKET.get(detail_type) == bucket
        }
        mask = np.asarray([int(label) in bucket_indices for label in label_arr], dtype=bool)
        if not mask.any():
            bucket_recalls[bucket] = 0.0
            continue
        pred_bucket = np.asarray([int(pred) in bucket_indices for pred in pred_arr], dtype=bool)
        bucket_recalls[bucket] = float((pred_bucket[mask]).mean())

    return {
        "accuracy": accuracy,
        "macro_f1_supported": float(sum(f1_values) / len(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion.astype(int).tolist(),
        "real_bucket_recall": bucket_recalls.get(AnnotationLabel.REAL.value, 0.0),
        "bogus_bucket_recall": bucket_recalls.get(AnnotationLabel.BOGUS.value, 0.0),
    }
