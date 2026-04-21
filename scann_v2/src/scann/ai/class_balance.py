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
    "training_mode": "frozen_feature_classifier",
    "feature_encoder": "auto",
    "feature_cache_enabled": True,
    "class_weight_beta": 0.999,
    "class_weight_clip": [0.25, 5.0],
    "sampler_power": 0.5,
    "sampler_max_ratio": 4.0,
    "min_train_support_warning": 5,
    "min_val_support_warning": 1,
    "selection_metric": "macro_f1_supported",
    "selection_metric_weights": {
        "macro_f1_supported": 0.5,
        "tail_recall@1": 0.3,
        "macro_ap": 0.2,
    },
    "selection_constraints": {},
    "head_min_support": 100,
    "mid_min_support": 21,
    "tail_max_support": 20,
    "tail_recall_max_support": 20,
    "variance_transfer": {
        "enabled": True,
        "synthetic_per_tail": 500,
        "tail_max_support": 20,
        "donor_min_support": 100,
        "shrinkage": 0.2,
        "covariance_mode": "full",
    },
    "dbl": {
        "enabled": True,
        "quality_min_weight": 0.35,
        "quality_max_weight": 1.0,
        "focal_gamma": 2.0,
    },
    "prior_logit_correction": {
        "enabled": True,
        "tau": 1.0,
        "smoothing": 1.0,
    },
    "expert_distillation": {
        "enabled": False,
    },
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
                default_value = DEFAULT_IMBALANCE_CONFIG[key]
                raw_value = raw[key]
                if isinstance(default_value, dict) and isinstance(raw_value, dict):
                    nested = dict(default_value)
                    nested.update(raw_value)
                    merged[key] = nested
                else:
                    merged[key] = raw_value
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
    merged["head_min_support"] = max(1, int(merged.get("head_min_support") or 100))
    merged["mid_min_support"] = max(1, int(merged.get("mid_min_support") or 21))
    merged["tail_max_support"] = max(1, int(merged.get("tail_max_support") or 20))
    merged["tail_recall_max_support"] = max(1, int(merged.get("tail_recall_max_support") or 20))
    weights = dict(DEFAULT_IMBALANCE_CONFIG["selection_metric_weights"])
    if isinstance(merged.get("selection_metric_weights"), dict):
        weights.update(merged["selection_metric_weights"])
    clean_weights: dict[str, float] = {}
    for key in ("macro_f1_supported", "tail_recall@1", "macro_ap"):
        try:
            value = float(weights.get(key, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        clean_weights[key] = value if math.isfinite(value) else 0.0
    if sum(max(0.0, value) for value in clean_weights.values()) <= 0.0:
        clean_weights = dict(DEFAULT_IMBALANCE_CONFIG["selection_metric_weights"])
    merged["selection_metric_weights"] = clean_weights
    constraints: dict[str, float] = {}
    if isinstance(merged.get("selection_constraints"), dict):
        for key, raw_value in merged["selection_constraints"].items():
            normalized_key = str(key or "").strip()
            if normalized_key not in {"macro_f1_supported", "tail_recall@1", "macro_ap", "long_tail_score"}:
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                constraints[normalized_key] = max(0.0, value)
    merged["selection_constraints"] = constraints

    vt = dict(DEFAULT_IMBALANCE_CONFIG["variance_transfer"])
    if isinstance(merged.get("variance_transfer"), dict):
        vt.update(merged["variance_transfer"])
    vt["enabled"] = bool(vt.get("enabled", True))
    vt["synthetic_per_tail"] = max(0, int(vt.get("synthetic_per_tail") or 0))
    vt["tail_max_support"] = max(1, int(vt.get("tail_max_support") or merged["tail_max_support"]))
    vt["donor_min_support"] = max(1, int(vt.get("donor_min_support") or merged["head_min_support"]))
    vt["shrinkage"] = max(0.0, min(1.0, float(vt.get("shrinkage") or 0.0)))
    vt["covariance_mode"] = str(vt.get("covariance_mode") or "full").strip().lower()
    if vt["covariance_mode"] not in {"full", "diagonal"}:
        vt["covariance_mode"] = "full"
    merged["variance_transfer"] = vt

    dbl = dict(DEFAULT_IMBALANCE_CONFIG["dbl"])
    if isinstance(merged.get("dbl"), dict):
        dbl.update(merged["dbl"])
    dbl["enabled"] = bool(dbl.get("enabled", True))
    dbl["quality_min_weight"] = max(0.0, float(dbl.get("quality_min_weight") or 0.0))
    dbl["quality_max_weight"] = max(dbl["quality_min_weight"], float(dbl.get("quality_max_weight") or 1.0))
    dbl["focal_gamma"] = max(0.0, float(dbl.get("focal_gamma") or 0.0))
    merged["dbl"] = dbl

    prior = dict(DEFAULT_IMBALANCE_CONFIG["prior_logit_correction"])
    if isinstance(merged.get("prior_logit_correction"), dict):
        prior.update(merged["prior_logit_correction"])
    prior["enabled"] = bool(prior.get("enabled", True))
    prior["tau"] = max(0.0, float(prior.get("tau") or 0.0))
    prior["smoothing"] = max(0.0, float(prior.get("smoothing") or 0.0))
    merged["prior_logit_correction"] = prior

    distill = dict(DEFAULT_IMBALANCE_CONFIG["expert_distillation"])
    if isinstance(merged.get("expert_distillation"), dict):
        distill.update(merged["expert_distillation"])
    distill["enabled"] = bool(distill.get("enabled", False))
    merged["expert_distillation"] = distill
    return merged


def classify_long_tail_support(
    class_counts: list[int],
    *,
    head_min_support: int = 100,
    mid_min_support: int = 21,
    tail_max_support: int = 20,
) -> dict[str, list[str]]:
    head_classes: list[str] = []
    mid_classes: list[str] = []
    tail_classes: list[str] = []
    zero_shot_classes: list[str] = []
    tail_recall_eligible_classes: list[str] = []

    for index, count in enumerate(class_counts):
        detail_type = DETAIL_TYPE_CLASS_ORDER[index]
        support = int(count)
        if support <= 0:
            zero_shot_classes.append(detail_type)
        elif support >= int(head_min_support):
            head_classes.append(detail_type)
        elif support >= int(mid_min_support):
            mid_classes.append(detail_type)
        else:
            tail_classes.append(detail_type)
        if 0 < support <= int(tail_max_support):
            tail_recall_eligible_classes.append(detail_type)

    return {
        "head_classes": head_classes,
        "mid_classes": mid_classes,
        "tail_classes": tail_classes,
        "zero_shot_classes": zero_shot_classes,
        "tail_recall_eligible_classes": tail_recall_eligible_classes,
    }


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
    head_min_support: int = 100,
    mid_min_support: int = 21,
    tail_max_support: int = 20,
) -> dict[str, Any]:
    record_list = list(records)
    class_counts = _counts_by_label(record_list)
    support_groups = classify_long_tail_support(
        class_counts,
        head_min_support=head_min_support,
        mid_min_support=mid_min_support,
        tail_max_support=tail_max_support,
    )
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
    if missing_classes:
        promotion_warnings.append(
            "zero_shot_classes_present: " + ", ".join(missing_classes)
        )
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
        **support_groups,
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


def compute_sample_quality(record: dict[str, Any]) -> float:
    """Estimate how much a sample should influence DBL instance weighting.

    The score intentionally uses only data already present in SCANN training records.
    It is not a truth label; it keeps obvious low-quality crops from dominating the
    tiny tail-class gradient budget.
    """
    data = record.get("data")
    quality_terms: list[float] = []

    try:
        width = float(record.get("bbox_width", 0.0) or 0.0)
        height = float(record.get("bbox_height", 0.0) or 0.0)
        area = max(0.0, width * height)
        quality_terms.append(float(min(1.0, max(0.15, area / 16.0))))
    except Exception:
        pass

    try:
        edge_margin = float(record.get("edge_margin", 999.0) or 0.0)
        patch_size = float(record.get("patch_size", 80.0) or 80.0)
        quality_terms.append(float(min(1.0, max(0.35, edge_margin / max(1.0, patch_size * 0.25)))))
    except Exception:
        pass

    if isinstance(data, np.ndarray) and data.size > 0:
        arr = np.asarray(data, dtype=np.float32)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            quality_terms.append(float(min(1.0, max(0.2, np.std(finite) / 0.12))))
        if arr.ndim == 3 and arr.shape[0] > 0:
            diff = np.abs(arr[0].astype(np.float32) - 0.5)
            noise = float(np.std(diff) + 1e-6)
            peak_snr = float(np.percentile(diff, 99.0) / noise) if diff.size else 0.0
            quality_terms.append(float(min(1.0, max(0.2, peak_snr / 8.0))))

    try:
        confidence = float(record.get("confidence", 1.0) or 1.0)
        quality_terms.append(float(min(1.0, max(0.5, confidence))))
    except Exception:
        pass

    if not quality_terms:
        return 1.0
    return float(min(1.0, max(0.05, sum(quality_terms) / len(quality_terms))))


def build_class_log_prior(
    labels: Iterable[int],
    *,
    class_count: int = len(DETAIL_TYPE_CLASS_ORDER),
    smoothing: float = 1.0,
) -> list[float]:
    counts = [0] * int(class_count)
    for raw_label in labels:
        label = int(raw_label)
        if 0 <= label < class_count:
            counts[label] += 1

    supported = [index for index, count in enumerate(counts) if count > 0]
    if not supported:
        return [0.0] * int(class_count)

    smooth = max(0.0, float(smoothing))
    denom = float(sum(counts[index] + smooth for index in supported))
    priors = [0.0] * int(class_count)
    for index in supported:
        freq = float(counts[index] + smooth) / max(denom, 1e-12)
        priors[index] = float(math.log(max(freq, 1e-12)))
    return priors


def generate_variance_transfer_features(
    features: np.ndarray,
    labels: Iterable[int],
    *,
    class_count: int = len(DETAIL_TYPE_CLASS_ORDER),
    config: dict[str, Any] | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    labels_arr = np.asarray(list(labels), dtype=np.int64)
    features_arr = np.asarray(features, dtype=np.float32)
    if features_arr.ndim != 2 or labels_arr.ndim != 1 or features_arr.shape[0] != labels_arr.shape[0]:
        raise ValueError("features and labels must be aligned 2D/1D arrays")

    cfg = dict(DEFAULT_IMBALANCE_CONFIG["variance_transfer"])
    if isinstance(config, dict):
        cfg.update(config)
    enabled = bool(cfg.get("enabled", True))
    synthetic_per_tail = max(0, int(cfg.get("synthetic_per_tail") or 0))
    if not enabled or synthetic_per_tail <= 0 or features_arr.size == 0:
        return (
            np.empty((0, features_arr.shape[1] if features_arr.ndim == 2 else 0), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            {"enabled": enabled, "synthetic_counts": {}, "donor_classes": {}},
        )

    tail_max_support = max(1, int(cfg.get("tail_max_support") or 20))
    donor_min_support = max(1, int(cfg.get("donor_min_support") or 100))
    shrinkage = max(0.0, min(1.0, float(cfg.get("shrinkage") or 0.0)))
    covariance_mode = str(cfg.get("covariance_mode") or "full").strip().lower()
    rng = np.random.default_rng(int(seed))

    counts = [int((labels_arr == index).sum()) for index in range(int(class_count))]
    tail_indices = [index for index, count in enumerate(counts) if 0 < count <= tail_max_support]
    donor_indices = [index for index, count in enumerate(counts) if count >= donor_min_support]
    if not donor_indices:
        donor_indices = [index for index, count in enumerate(counts) if count > tail_max_support]
    if not donor_indices:
        return (
            np.empty((0, features_arr.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            {"enabled": enabled, "synthetic_counts": {}, "donor_classes": {}},
        )

    centroids: dict[int, np.ndarray] = {}
    for index, count in enumerate(counts):
        if count > 0:
            centroids[index] = features_arr[labels_arr == index].mean(axis=0)

    synthetic_features: list[np.ndarray] = []
    synthetic_labels: list[np.ndarray] = []
    synthetic_counts: dict[str, int] = {}
    donor_classes: dict[str, str] = {}

    for tail_index in tail_indices:
        tail_center = centroids.get(tail_index)
        if tail_center is None:
            continue
        donor_index = min(
            donor_indices,
            key=lambda item: float(np.linalg.norm(centroids[item] - tail_center)),
        )
        donor = features_arr[labels_arr == donor_index].astype(np.float64)
        if donor.shape[0] < 2:
            variances = np.var(features_arr, axis=0).astype(np.float64)
            cov = np.diag(np.maximum(variances, 1e-6))
        else:
            centered = donor - donor.mean(axis=0, keepdims=True)
            cov = np.cov(centered, rowvar=False)
            if cov.ndim == 0:
                cov = np.asarray([[float(cov)]], dtype=np.float64)
            diag = np.diag(np.maximum(np.diag(cov), 1e-6))
            if covariance_mode == "diagonal":
                cov = diag
            else:
                cov = (1.0 - shrinkage) * cov + shrinkage * diag
            cov = cov + np.eye(cov.shape[0], dtype=np.float64) * 1e-6

        try:
            sampled = rng.multivariate_normal(
                mean=tail_center.astype(np.float64),
                cov=cov,
                size=synthetic_per_tail,
                check_valid="ignore",
            ).astype(np.float32)
        except Exception:
            std = np.sqrt(np.maximum(np.diag(cov), 1e-6)).astype(np.float32)
            sampled = (
                tail_center.astype(np.float32)[None, :]
                + rng.normal(0.0, 1.0, size=(synthetic_per_tail, features_arr.shape[1])).astype(np.float32) * std[None, :]
            )

        synthetic_features.append(sampled)
        synthetic_labels.append(np.full((synthetic_per_tail,), tail_index, dtype=np.int64))
        synthetic_counts[DETAIL_TYPE_CLASS_ORDER[tail_index]] = int(synthetic_per_tail)
        donor_classes[DETAIL_TYPE_CLASS_ORDER[tail_index]] = DETAIL_TYPE_CLASS_ORDER[donor_index]

    if not synthetic_features:
        return (
            np.empty((0, features_arr.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            {"enabled": enabled, "synthetic_counts": {}, "donor_classes": {}},
        )

    return (
        np.concatenate(synthetic_features, axis=0).astype(np.float32),
        np.concatenate(synthetic_labels, axis=0).astype(np.int64),
        {
            "enabled": enabled,
            "synthetic_counts": synthetic_counts,
            "donor_classes": donor_classes,
            "synthetic_total": int(sum(synthetic_counts.values())),
        },
    )


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
    probs: Iterable[Iterable[float]] | np.ndarray | None = None,
    class_count: int = len(DETAIL_TYPE_CLASS_ORDER),
    tail_max_support: int = 20,
) -> dict[str, Any]:
    label_arr = np.asarray(list(labels), dtype=np.int64)
    pred_arr = np.asarray(list(preds), dtype=np.int64)
    prob_arr = None
    if probs is not None:
        prob_arr = np.asarray(list(probs) if not isinstance(probs, np.ndarray) else probs, dtype=np.float64)
        if prob_arr.ndim != 2 or prob_arr.shape[0] != label_arr.shape[0] or prob_arr.shape[1] < class_count:
            prob_arr = None
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    for true_label, pred_label in zip(label_arr, pred_arr):
        if 0 <= int(true_label) < class_count and 0 <= int(pred_label) < class_count:
            confusion[int(true_label), int(pred_label)] += 1

    per_class: dict[str, dict[str, float | int | str | None]] = {}
    f1_values: list[float] = []
    ap_values: list[float] = []
    tail_recalls: list[float] = []
    for index, detail_type in enumerate(DETAIL_TYPE_CLASS_ORDER):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index, :].sum() - tp)
        support = int(confusion[index, :].sum())
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = (2.0 * precision * recall) / (precision + recall + 1e-12)
        ap = 0.0
        if prob_arr is not None and support > 0:
            try:
                from sklearn.metrics import average_precision_score

                y_true = (label_arr == index).astype(np.int32)
                if y_true.max() > 0:
                    ap = float(average_precision_score(y_true, prob_arr[:, index]))
                    ap_values.append(ap)
            except Exception:
                ap = 0.0
        if support > 0:
            f1_values.append(float(f1))
            if support <= int(tail_max_support):
                tail_recalls.append(float(recall))
        per_class[detail_type] = {
            "index": int(index),
            "bucket": DETAIL_TYPE_TO_BUCKET.get(detail_type),
            "support": support,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "ap": float(ap),
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
        "macro_ap": float(sum(ap_values) / len(ap_values)) if ap_values else 0.0,
        "tail_recall@1": float(sum(tail_recalls) / len(tail_recalls)) if tail_recalls else 0.0,
        "per_class_ap": {
            detail_type: float(per_class[detail_type]["ap"])
            for detail_type in DETAIL_TYPE_CLASS_ORDER
        },
        "per_class": per_class,
        "confusion_matrix": confusion.astype(int).tolist(),
        "real_bucket_recall": bucket_recalls.get(AnnotationLabel.REAL.value, 0.0),
        "bogus_bucket_recall": bucket_recalls.get(AnnotationLabel.BOGUS.value, 0.0),
    }
