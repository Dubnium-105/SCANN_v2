from __future__ import annotations

import hashlib
import json
import sqlite3

from scann.core.dataset_audit import audit_dataset, main
from scann.core.dataset_storage import DatasetStorage


def _insert_raw_asset(
    connection: sqlite3.Connection,
    *,
    asset_id: str,
    relpath: str,
) -> None:
    connection.execute(
        """
        INSERT INTO raw_assets (
            asset_id,
            asset_role,
            field_key,
            field_name,
            capture_key,
            relpath,
            file_name,
            file_stem,
            suffix,
            status,
            created_at,
            updated_at
        ) VALUES (?, 'new', 'field', 'field', 'capture', ?, 'new.fts',
                  'new', '.fts', 'active', '2026-01-01', '2026-01-01')
        """,
        (asset_id, relpath),
    )


def _insert_task(
    connection: sqlite3.Connection,
    *,
    task_id: str,
    asset_id: str,
    annotation_count: int = 0,
) -> None:
    connection.execute(
        """
        INSERT INTO tasks (
            task_id,
            field_key,
            field_name,
            capture_key,
            new_asset_id,
            current_annotation_count,
            created_at,
            updated_at
        ) VALUES (?, 'field', 'field', 'capture', ?, ?, '2026-01-01', '2026-01-01')
        """,
        (task_id, asset_id, annotation_count),
    )


def test_empty_dataset_schema_audits_cleanly(tmp_path):
    storage = DatasetStorage(tmp_path)
    storage.ensure_schema()
    before_mtime = storage.db_path.stat().st_mtime_ns

    report = audit_dataset(tmp_path)

    assert report.status == "ok"
    assert report.integrity_check == ["ok"]
    assert report.foreign_key_violation_count == 0
    assert report.counts["schema_migrations"] == 3
    assert report.counts["dataset_partitions"] == 0
    assert report.files["dataset_partitions"]["invalid"] == 0
    assert storage.db_path.stat().st_mtime_ns == before_mtime


def test_audit_reports_missing_files_and_invalid_annotations(tmp_path):
    storage = DatasetStorage(tmp_path)
    storage.ensure_schema()
    with sqlite3.connect(storage.db_path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        _insert_raw_asset(connection, asset_id="asset-1", relpath="missing.fts")
        _insert_task(
            connection,
            task_id="task-1",
            asset_id="asset-1",
            annotation_count=1,
        )
        connection.execute(
            """
            INSERT INTO task_annotation_boxes_current (
                task_id, box_index, x, y, width, height, label, detail_type
            ) VALUES ('task-1', 0, 1, 1, 5, 5, 'maybe', 'unknown-detail')
            """
        )

    report = audit_dataset(tmp_path)
    codes = {issue.code for issue in report.issues}

    assert report.status == "error"
    assert "raw_assets_missing" in codes
    assert "invalid_annotation_labels" in codes
    assert "invalid_annotation_detail_types" in codes


def test_audit_reports_annotation_cache_mismatch(tmp_path):
    storage = DatasetStorage(tmp_path)
    storage.ensure_schema()
    asset_path = tmp_path / "new.fts"
    asset_path.write_bytes(b"fits")
    with sqlite3.connect(storage.db_path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        _insert_raw_asset(connection, asset_id="asset-1", relpath="new.fts")
        _insert_task(
            connection,
            task_id="task-1",
            asset_id="asset-1",
            annotation_count=2,
        )

    report = audit_dataset(tmp_path)

    assert report.status == "error"
    assert any(
        issue.code == "task_annotation_count_mismatch"
        for issue in report.issues
    )


def test_audit_verifies_recorded_model_hash(tmp_path):
    storage = DatasetStorage(tmp_path)
    storage.ensure_schema()
    artifact = tmp_path / ".scann_control" / "models" / "model-1" / "model.pth"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"checkpoint")
    wrong_hash = hashlib.sha256(b"different").hexdigest()
    metadata = json.dumps({"artifact": {"sha256": wrong_hash}})
    with sqlite3.connect(storage.db_path) as connection:
        connection.execute(
            """
            INSERT INTO model_registry (
                model_id,
                model_version,
                model_backbone,
                task_type,
                artifact_path,
                metadata_json,
                created_at,
                updated_at
            ) VALUES (
                'model-1',
                'v1',
                'test',
                'classification',
                '.scann_control/models/model-1/model.pth',
                ?,
                '2026-01-01',
                '2026-01-01'
            )
            """,
            (metadata,),
        )

    report = audit_dataset(tmp_path)

    assert report.status == "error"
    assert report.models["hash_mismatches"] == 1
    assert any(
        issue.code == "model_artifact_hash_mismatch"
        for issue in report.issues
    )


def test_audit_cli_writes_report_without_touching_dataset(tmp_path):
    storage = DatasetStorage(tmp_path)
    storage.ensure_schema()
    output_path = tmp_path.parent / f"{tmp_path.name}-audit.json"

    exit_code = main(
        [
            str(tmp_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["database_path"] == str(storage.db_path.resolve())
