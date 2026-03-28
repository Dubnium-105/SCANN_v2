from __future__ import annotations

from scann.core.dataset_storage import DatasetStorage, RawAssetRecord, TaskRecord


def test_current_annotation_status_distinguishes_local_and_online(tmp_path) -> None:
    storage = DatasetStorage(tmp_path)
    storage.ensure_schema()
    storage.upsert_raw_assets(
        [
            RawAssetRecord(
                asset_id="new-1",
                asset_role="new",
                field_key="field_001",
                field_name="field_001",
                capture_key="field_001",
                relpath="dataset_raw/new/field_001.fits",
                file_name="field_001.fits",
                file_stem="field_001",
                suffix=".fits",
            ),
            RawAssetRecord(
                asset_id="old-1",
                asset_role="old",
                field_key="field_001",
                field_name="field_001",
                capture_key="field_001",
                relpath="dataset_raw/old/field_001.fits",
                file_name="field_001.fits",
                file_stem="field_001",
                suffix=".fits",
            ),
        ]
    )
    storage.sync_tasks(
        [
            TaskRecord(
                task_id="task-1",
                field_key="field_001",
                field_name="field_001",
                capture_key="field_001",
                new_asset_id="new-1",
                old_asset_id="old-1",
            )
        ]
    )

    storage.upsert_current_annotation(
        task_id="task-1",
        source_view="new",
        label="real",
        detail_type=None,
        ai_suggestion=None,
        ai_confidence=None,
        annotations=[{"x": 1, "y": 2, "width": 3, "height": 4, "label": "real"}],
        annotation_origin="local",
    )
    task = storage.get_task_by_id("task-1")
    assert task is not None
    assert task.local_annotation_status == "annotated"
    assert task.online_annotation_status is None

    storage.upsert_current_annotation(
        task_id="task-1",
        source_view="new",
        label="bogus",
        detail_type=None,
        ai_suggestion=None,
        ai_confidence=None,
        annotations=[{"x": 1, "y": 2, "width": 3, "height": 4, "label": "bogus"}],
        annotation_origin="online",
    )
    task = storage.get_task_by_id("task-1")
    assert task is not None
    assert task.local_annotation_status == "annotated"
    assert task.online_annotation_status == "annotated"
