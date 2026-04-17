from __future__ import annotations

from scann.ai.class_balance import (
    DETAIL_TYPE_CLASS_ORDER,
    build_class_audit,
    compute_class_balanced_weights,
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
    assert any("train_support_below_minimum" in item for item in audit["promotion_warnings"])
    assert any("val_support_below_minimum" in item for item in audit["promotion_warnings"])
