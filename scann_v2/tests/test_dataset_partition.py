from __future__ import annotations

from scann.ai.dataset_partition import (
    build_partition_manifest,
    derive_night_key,
    task_ids_for_splits,
    verify_partition_manifest,
)


def _image(
    task_id: str,
    *,
    field_key: str,
    detail_type: str,
    date_obs: str | None = None,
):
    return {
        "id": task_id,
        "field_key": field_key,
        "capture_key": task_id,
        "date_obs": date_obs,
        "annotations": [
            {
                "x": 1,
                "y": 2,
                "width": 3,
                "height": 4,
                "detail_type": detail_type,
            }
        ],
    }


def test_night_key_uses_fits_date_then_task_id():
    assert derive_night_key("2026-02-04T12:30:00Z", "task") == "2026-02-04"
    assert derive_night_key(None, "20260205T123000__FIELD") == "2026-02-05"
    assert derive_night_key(None, "task") == "unknown"


def test_partition_is_deterministic_and_group_isolated():
    images = []
    for night in range(1, 7):
        date = f"2026-02-{night:02d}"
        for field_index in range(2):
            field_key = f"field-{field_index}"
            detail_type = "asteroid" if (night + field_index) % 2 else "noise"
            task_id = f"202602{night:02d}T120000__{field_key}"
            images.append(
                _image(
                    task_id,
                    field_key=field_key,
                    detail_type=detail_type,
                    date_obs=f"{date}T12:00:00Z",
                )
            )

    first = build_partition_manifest(
        images,
        partition_name="gold-v1",
        seed=17,
    )
    second = build_partition_manifest(
        list(reversed(images)),
        partition_name="gold-v1",
        seed=17,
    )

    assert first == second
    assert verify_partition_manifest(first) is True
    split_task_ids = {
        split_name: set(task_ids_for_splits(first, [split_name]))
        for split_name in ("train", "validation", "test")
    }
    assert split_task_ids["train"]
    assert split_task_ids["validation"]
    assert split_task_ids["test"]
    assert not (split_task_ids["train"] & split_task_ids["validation"])
    assert not (split_task_ids["train"] & split_task_ids["test"])
    assert not (split_task_ids["validation"] & split_task_ids["test"])
    assert first["audit"]["group_overlap"] == []


def test_single_independent_rare_group_is_kept_in_training():
    images = [
        _image(
            "20260201T120000__rare",
            field_key="rare",
            detail_type="supernova",
            date_obs="2026-02-01T12:00:00Z",
        )
    ]
    for index in range(2, 9):
        images.append(
            _image(
                f"202602{index:02d}T120000__common",
                field_key="common",
                detail_type="noise",
                date_obs=f"2026-02-{index:02d}T12:00:00Z",
            )
        )

    manifest = build_partition_manifest(
        images,
        partition_name="rare-v1",
    )

    train_ids = task_ids_for_splits(manifest, ["train"])
    assert "20260201T120000__rare" in train_ids
    assert manifest["audit"]["limited_group_classes"]["supernova"] == 1


def test_partition_rejects_too_few_independent_groups():
    images = [
        _image(
            f"task-{index}",
            field_key="same-field",
            detail_type="noise",
            date_obs="2026-02-01T12:00:00Z",
        )
        for index in range(3)
    ]

    try:
        build_partition_manifest(
            images,
            partition_name="invalid",
        )
    except ValueError as exc:
        assert "independent groups" in str(exc)
    else:
        raise AssertionError("expected partition construction to fail")
