from __future__ import annotations

from scann.ai.class_balance import (
    DETAIL_TYPE_CLASS_ORDER,
    build_class_audit,
    build_class_log_prior,
    compute_class_balanced_weights,
    compute_multiclass_metrics,
    generate_variance_transfer_features,
    sampler_weights_from_class_weights,
    stratified_group_train_val_split,
)


def _record(task_id: str, label: int) -> dict:
    return {
        "task_id": task_id,
        "annotation_index": 0,
        "detail_type": DETAIL_TYPE_CLASS_ORDER[label],
        "label": label,
    }


def test_stratified_group_split_keeps_task_groups_together_and_singletons_train_only() -> None:
    records = [
        _record("rare-singleton", 0),
        *[_record(f"noise-{index}", 4) for index in range(8)],
        *[_record(f"corresponding-{index}", 7) for index in range(8)],
    ]

    train_idx, val_idx, support = stratified_group_train_val_split(records, val_split=0.25, seed=7)

    train_tasks = {records[index]["task_id"] for index in train_idx}
    val_tasks = {records[index]["task_id"] for index in val_idx}
    assert train_tasks.isdisjoint(val_tasks)
    assert "rare-singleton" in train_tasks
    assert "rare-singleton" not in val_tasks
    assert support["train"][0] == 1
    assert support["val"][4] >= 1
    assert support["val"][7] >= 1


def test_stratified_group_split_keeps_all_singletons_out_of_validation() -> None:
    records = [_record(f"singleton-{index}", index) for index in range(4)]

    train_idx, val_idx, support = stratified_group_train_val_split(records, val_split=0.5, seed=3)

    assert sorted(train_idx) == list(range(4))
    assert val_idx == []
    assert sum(support["train"]) == 4
    assert sum(support["val"]) == 0


def test_class_balanced_weights_and_sampler_are_clipped() -> None:
    labels = [0] + ([7] * 100)

    class_weights = compute_class_balanced_weights(
        labels,
        beta=0.999,
        clip=(0.25, 5.0),
        class_count=len(DETAIL_TYPE_CLASS_ORDER),
    )
    sample_weights = sampler_weights_from_class_weights(
        labels,
        class_weights,
        power=0.5,
        max_ratio=4.0,
    )

    supported = [value for value in class_weights if value > 0]
    assert max(supported) <= 5.0 + 1e-9
    assert min(supported) >= 0.25 - 1e-9
    assert max(sample_weights) / min(sample_weights) <= 4.0 + 1e-9


def test_class_audit_reports_split_support_and_promotion_warnings() -> None:
    records = [_record("one", 0), _record("two", 7)]
    split_support = {
        "train": [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "val": [0] * len(DETAIL_TYPE_CLASS_ORDER),
    }

    audit = build_class_audit(records, split_support=split_support, min_train_support=5, min_val_support=1)

    assert audit["total_samples"] == 2
    assert audit["class_counts"]["asteroid"] == 1
    assert audit["bucket_counts"]["real"] == 1
    assert audit["bucket_counts"]["bogus"] == 1
    assert audit["missing_classes"]
    assert audit["unverifiable_classes"]
    assert "zero_shot_classes" in audit
    assert "tail_classes" in audit
    assert "asteroid" in audit["tail_classes"]
    assert any("train_support_below_minimum" in item for item in audit["promotion_warnings"])
    assert any("val_support_below_minimum" in item for item in audit["promotion_warnings"])
    assert any("zero_shot_classes_present" in item for item in audit["promotion_warnings"])


def test_multiclass_metrics_include_tail_recall_and_ap() -> None:
    labels = [0, 0, 7, 7]
    preds = [0, 7, 7, 7]
    probs = [
        [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
        [0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0],
    ]

    metrics = compute_multiclass_metrics(labels, preds, probs=probs)

    assert "macro_ap" in metrics
    assert "tail_recall@1" in metrics
    assert metrics["per_class"]["asteroid"]["ap"] > 0.0


def test_variance_transfer_generates_tail_features_without_zero_shot_labels() -> None:
    import numpy as np

    head = np.stack([np.array([float(index), float(index % 3)]) for index in range(20)]).astype(np.float32)
    tail = np.asarray([[100.0, 100.0], [101.0, 100.0]], dtype=np.float32)
    features = np.concatenate([tail, head], axis=0)
    labels = [0, 0] + [7] * 20

    synthetic, synthetic_labels, summary = generate_variance_transfer_features(
        features,
        labels,
        config={
            "enabled": True,
            "synthetic_per_tail": 5,
            "tail_max_support": 2,
            "donor_min_support": 10,
            "shrinkage": 0.2,
            "covariance_mode": "full",
        },
        seed=1,
    )

    assert synthetic.shape == (5, 2)
    assert synthetic_labels.tolist() == [0] * 5
    assert summary["synthetic_counts"]["asteroid"] == 5


def test_class_log_prior_does_not_boost_zero_shot_classes() -> None:
    priors = build_class_log_prior([0, 0, 7], smoothing=1.0)

    assert priors[0] < 0.0
    assert priors[7] < 0.0
    assert priors[1] == 0.0
