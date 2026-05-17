from __future__ import annotations

from scann.core.dataset_storage import DatasetStorage, RawAssetRecord, TaskArtifactRecord, TaskRecord


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


def test_release_online_annotation_claim_keeps_annotated_status(tmp_path) -> None:
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
                preprocess_status="ready",
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
        annotation_origin="online",
    )
    assert storage.try_claim_task(
        task_id="task-1",
        client_id="client-a",
        expires_at="2026-04-11T00:30:00+00:00",
        now_iso="2026-04-11T00:00:00+00:00",
    )

    assert storage.release_claim(task_id="task-1", client_id="client-a")
    task = storage.get_task_by_id("task-1")
    assert task is not None
    assert task.preprocess_status == "annotated"


def test_list_prepared_task_paths_skips_missing_files(tmp_path) -> None:
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
                preprocess_status="ready",
            )
        ]
    )

    aligned_new = tmp_path / "new" / "task-1__aligned_crop.fts"
    aligned_old = tmp_path / "old" / "task-1__aligned_crop.fts"
    aligned_new.parent.mkdir(parents=True, exist_ok=True)
    aligned_old.parent.mkdir(parents=True, exist_ok=True)
    aligned_new.write_bytes(b"new")

    storage.upsert_task_artifact(
        TaskArtifactRecord(task_id="task-1", artifact_role="aligned_new", relpath="new/task-1__aligned_crop.fts")
    )
    storage.upsert_task_artifact(
        TaskArtifactRecord(task_id="task-1", artifact_role="aligned_old", relpath="old/task-1__aligned_crop.fts")
    )

    prepared = storage.list_prepared_task_paths()
    assert prepared == [
        {
            "task_id": "task-1",
            "new_path": "new/task-1__aligned_crop.fts",
            "old_path": None,
            "new_marked_path": None,
        }
    ]

    aligned_new.unlink()
    prepared = storage.list_prepared_task_paths()
    assert prepared == []


def test_list_tasks_by_ids_handles_large_id_batches(tmp_path) -> None:
    storage = DatasetStorage(tmp_path)
    storage.ensure_schema()

    count = 1005
    storage.upsert_raw_assets(
        [
            RawAssetRecord(
                asset_id=f"new-{index}",
                asset_role="new",
                field_key=f"field_{index:04d}",
                field_name=f"field_{index:04d}",
                capture_key=f"field_{index:04d}",
                relpath=f"dataset_raw/new/field_{index:04d}.fits",
                file_name=f"field_{index:04d}.fits",
                file_stem=f"field_{index:04d}",
                suffix=".fits",
            )
            for index in range(count)
        ]
    )
    storage.sync_tasks(
        [
            TaskRecord(
                task_id=f"task-{index:04d}",
                field_key=f"field_{index:04d}",
                field_name=f"field_{index:04d}",
                capture_key=f"field_{index:04d}",
                new_asset_id=f"new-{index}",
                preprocess_status="ready",
            )
            for index in range(count)
        ]
    )

    task_ids = [f"task-{index:04d}" for index in range(count)]
    tasks = storage.list_tasks_by_ids(task_ids)

    assert [task.task_id for task in tasks] == task_ids


def test_list_task_prelabel_summaries_handles_large_id_batches(tmp_path) -> None:
    storage = DatasetStorage(tmp_path)
    storage.ensure_schema()

    count = 1005
    storage.upsert_raw_assets(
        [
            RawAssetRecord(
                asset_id=f"new-{index}",
                asset_role="new",
                field_key=f"field_{index:04d}",
                field_name=f"field_{index:04d}",
                capture_key=f"field_{index:04d}",
                relpath=f"dataset_raw/new/field_{index:04d}.fits",
                file_name=f"field_{index:04d}.fits",
                file_stem=f"field_{index:04d}",
                suffix=".fits",
            )
            for index in range(count)
        ]
    )
    storage.sync_tasks(
        [
            TaskRecord(
                task_id=f"task-{index:04d}",
                field_key=f"field_{index:04d}",
                field_name=f"field_{index:04d}",
                capture_key=f"field_{index:04d}",
                new_asset_id=f"new-{index}",
                preprocess_status="ready",
            )
            for index in range(count)
        ]
    )
    for index in range(0, count, 100):
        storage.enqueue_prelabel_job(
            task_id=f"task-{index:04d}",
            requested_by="tester",
            model_version="test-model",
            input_fingerprint=f"fingerprint-{index}",
        )

    summaries = storage.list_task_prelabel_summaries(
        f"task-{index:04d}" for index in range(count)
    )

    assert sorted(summaries) == [f"task-{index:04d}" for index in range(0, count, 100)]
    assert {summary.prelabel_status for summary in summaries.values()} == {"queued"}
