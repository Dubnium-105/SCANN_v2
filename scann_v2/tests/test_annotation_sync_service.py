from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from scann.core.dataset_storage import DatasetStorage, RawAssetRecord, TaskRecord
from scann.native_annotation.annotation_sync_service import AnnotationSyncConfig, AnnotationSyncService
from scann.native_annotation.auth_service import AuthUser
from scann.native_annotation import routes as native_routes


class FakeCursor:
    def __init__(self, fetchone_result: Any = None) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.fetchone_result = fetchone_result
        self.closed = False

    def execute(self, sql: str, params: Any = None) -> None:
        self.calls.append((sql, params))

    def fetchone(self) -> Any:
        return self.fetchone_result

    def close(self) -> None:
        self.closed = True


class FakeConnection:
    def __init__(self, fetchone_result: Any = None) -> None:
        self.cursor_obj = FakeCursor(fetchone_result=fetchone_result)
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self) -> FakeCursor:
        return self.cursor_obj

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"SIMPLE FITS PLACEHOLDER")


def _seed_annotation_storage(dataset_root: Path) -> DatasetStorage:
    storage = DatasetStorage(dataset_root)
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
            )
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
    annotation = {
        "x": 1.0,
        "y": 2.0,
        "width": 3.0,
        "height": 4.0,
        "label": "real",
        "detail_type": "asteroid",
    }
    storage.upsert_current_annotation(
        task_id="task-1",
        source_view="new",
        label="real",
        detail_type="asteroid",
        ai_suggestion=None,
        ai_confidence=None,
        annotations=[annotation],
        annotation_origin="online",
    )
    storage.append_annotation_revision(
        task_id="task-1",
        source_view="new",
        parent_revision_id=None,
        rollback_of_revision_id=None,
        submitted_by="annotator",
        origin="online",
        saved_at="2026-04-10T00:00:00+00:00",
        metadata={"review": "initial"},
        annotations=[annotation],
        revision_id="rev-1",
    )
    return storage


def test_annotation_sync_service_syncs_annotation_only_payload(tmp_path) -> None:
    storage = _seed_annotation_storage(tmp_path)
    connections: list[FakeConnection] = []

    def connect(_database_url: str) -> FakeConnection:
        fetchone_result = None if not connections else (1,)
        connection = FakeConnection(fetchone_result=fetchone_result)
        connections.append(connection)
        return connection

    service = AnnotationSyncService(
        storage=storage,
        config=AnnotationSyncConfig(
            enabled=True,
            database_url="postgresql://example/scann",
            dataset_id="test-dataset",
            schema_name="scann_backup",
        ),
        connect_factory=connect,
    )

    result = service.sync_now()

    assert result.success is True
    assert result.sync_mode == "incremental"
    assert result.previous_revision_rowid == 0
    assert result.last_revision_rowid == 1
    assert result.tasks_synced == 1
    assert result.revisions_synced == 1
    assert result.current_boxes_synced == 1
    assert result.revision_boxes_synced == 1

    assert len(connections) == 1
    connection = connections[0]
    assert connection.committed is True
    assert connection.rolled_back is False
    assert connection.closed is True

    calls = connection.cursor_obj.calls
    assert any("annotation_current_boxes" in sql for sql, _params in calls)
    assert any("annotation_revision_boxes" in sql for sql, _params in calls)
    assert any("ON CONFLICT(dataset_id, revision_id)" in sql for sql, _params in calls)

    serialized_params = "\n".join(repr(params) for _sql, params in calls if params is not None)
    assert "task-1" in serialized_params
    assert "rev-1" in serialized_params
    assert "field_001.fits" not in serialized_params
    assert "dataset_raw/new" not in serialized_params

    second_result = service.sync_now()

    assert second_result.success is True
    assert second_result.previous_revision_rowid == 1
    assert second_result.last_revision_rowid == 1
    assert second_result.tasks_synced == 0
    assert second_result.revisions_synced == 0
    assert second_result.current_boxes_synced == 0
    assert second_result.revision_boxes_synced == 0

    second_calls = connections[1].cursor_obj.calls
    assert not any("INSERT INTO scann_backup.annotation_tasks" in sql for sql, _params in second_calls)
    assert not any("INSERT INTO scann_backup.annotation_revisions" in sql for sql, _params in second_calls)
    assert not any("INSERT INTO scann_backup.annotation_sync_runs" in sql for sql, _params in second_calls)


def test_annotation_sync_routes_report_status_and_require_admin_config(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    _touch(dataset_root / "new" / "PGC 17069.fts")
    monkeypatch.setenv("SCANN_NATIVE_DATASET_ROOT", str(dataset_root))
    monkeypatch.delenv("SCANN_ANNOTATION_SYNC_DATABASE_URL", raising=False)
    monkeypatch.delenv("SCANN_ANNOTATION_SYNC_PG_DSN", raising=False)

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    admin = AuthUser(username="admin", role="admin")
    annotator = AuthUser(username="annotator", role="annotator")

    status = native_routes.get_annotation_sync_status(request=request, current_user=admin)
    assert status.configured is False
    assert status.scope == "annotations_only"

    with pytest.raises(HTTPException) as forbidden:
        native_routes.run_annotation_sync(request=request, full=False, current_user=annotator)
    assert forbidden.value.status_code == 403

    with pytest.raises(HTTPException) as not_configured:
        native_routes.run_annotation_sync(request=request, full=True, current_user=admin)
    assert not_configured.value.status_code == 400
    assert not_configured.value.detail == "PostgreSQL sync is not configured"
