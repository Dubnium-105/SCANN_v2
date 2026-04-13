from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET_DB_FILE = "scann_dataset.db"
_CAPABILITY_WILDCARDS = {"", "auto", "any", "*"}

_EXPLORER_COPY_SUFFIX_RE = re.compile(r"^(?P<base>.*?)(?:\s+\((?P<index>\d+)\))$")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_dataset_db_path(dataset_root: Path, db_file: str = DEFAULT_DATASET_DB_FILE) -> Path:
    configured = os.getenv("SCANN_DATASET_DB_PATH", "").strip()
    if configured:
        return Path(configured).resolve()
    return Path(dataset_root) / db_file


@dataclass(frozen=True)
class RawAssetRecord:
    asset_id: str
    asset_role: str
    field_key: str
    field_name: str
    capture_key: str
    relpath: str
    file_name: str
    file_stem: str
    suffix: str
    date_obs: str | None = None
    size_bytes: int = 0
    modified_time: float = 0.0
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    field_key: str
    field_name: str
    capture_key: str
    new_asset_id: str
    old_asset_id: str | None = None
    new_marked_asset_id: str | None = None
    preprocess_status: str = "pending"
    current_source_view: str | None = None
    current_label: str | None = None
    current_detail_type: str | None = None
    current_annotation_count: int = 0
    current_ai_suggestion: str | None = None
    current_ai_confidence: float | None = None
    annotation_updated_at: str | None = None
    local_viewed_at: str | None = None
    local_annotation_status: str | None = None
    online_annotation_status: str | None = None
    claim_client_id: str | None = None
    claim_locked_at: str | None = None
    claim_expires_at: str | None = None
    crop_x0: int | None = None
    crop_x1: int | None = None
    crop_y0: int | None = None
    crop_y1: int | None = None
    align_dx: float | None = None
    align_dy: float | None = None


@dataclass(frozen=True)
class TaskArtifactRecord:
    task_id: str
    artifact_role: str
    relpath: str
    width: int | None = None
    height: int | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class PrelabelJobRecord:
    job_id: str
    task_id: str
    requested_by: str
    model_version: str
    input_fingerprint: str
    status: str
    model_id: str | None = None
    model_backbone: str | None = None
    candidate_limit: int | None = None
    confidence_threshold: float | None = None
    priority: int = 100
    claim_worker_id: str | None = None
    claimed_at: str | None = None
    claim_expires_at: str | None = None
    last_heartbeat_at: str | None = None
    attempt_count: int = 0
    error_message: str | None = None
    result_prelabel_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class TaskAIPrelabelRecord:
    prelabel_id: str
    task_id: str
    job_id: str | None = None
    source_view: str | None = None
    ai_suggestion: str | None = None
    ai_confidence: float | None = None
    model_version: str | None = None
    model_id: str | None = None
    model_backbone: str | None = None
    candidate_limit: int | None = None
    confidence_threshold: float | None = None
    input_fingerprint: str | None = None
    status: str = "available"
    box_count: int = 0
    worker_id: str | None = None
    accepted_revision_id: str | None = None
    metadata: dict[str, Any] | None = None
    created_at: str | None = None
    updated_at: str | None = None
    superseded_at: str | None = None


@dataclass(frozen=True)
class WorkerNodeRecord:
    worker_id: str
    display_name: str | None = None
    host_name: str | None = None
    device_label: str | None = None
    status: str = "online"
    capabilities: dict[str, Any] | None = None
    last_seen_at: str | None = None
    last_claimed_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class TaskPrelabelSummaryRecord:
    task_id: str
    prelabel_status: str | None = None
    prelabel_model_version: str | None = None
    prelabel_model_id: str | None = None
    prelabel_model_backbone: str | None = None
    prelabel_candidate_limit: int | None = None
    prelabel_confidence_threshold: float | None = None
    prelabel_updated_at: str | None = None
    prelabel_box_count: int = 0
    prelabel_id: str | None = None
    prelabel_job_id: str | None = None


@dataclass(frozen=True)
class DatasetSnapshotRecord:
    snapshot_id: str
    snapshot_name: str
    document_relpath: str
    task_count: int = 0
    annotation_count: int = 0
    created_by: str | None = None
    metadata: dict[str, Any] | None = None
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class TrainingJobRecord:
    job_id: str
    snapshot_id: str
    requested_by: str
    task_type: str
    model_version: str
    model_id: str
    model_backbone: str
    status: str
    train_config: dict[str, Any] | None = None
    priority: int = 100
    promote_on_success: bool = False
    enqueue_prelabels_on_success: bool = False
    prelabel_task_ids: list[str] | None = None
    force_prelabel: bool = False
    claim_worker_id: str | None = None
    claimed_at: str | None = None
    claim_expires_at: str | None = None
    last_heartbeat_at: str | None = None
    attempt_count: int = 0
    error_message: str | None = None
    run_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class TrainingRunRecord:
    run_id: str
    job_id: str
    snapshot_id: str
    task_type: str
    status: str
    worker_id: str | None = None
    model_id: str | None = None
    model_version: str | None = None
    model_backbone: str | None = None
    artifact_path: str | None = None
    metrics: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    started_at: str | None = None
    finished_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class RegisteredModelRecord:
    model_id: str
    model_version: str
    model_backbone: str
    task_type: str
    training_run_id: str | None = None
    snapshot_id: str | None = None
    artifact_path: str | None = None
    metrics: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    created_by: str | None = None
    is_promoted: bool = False
    promoted_at: str | None = None
    promoted_by: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class DatasetStorage:
    _schema_ready_paths: set[str] = set()
    _wal_ready_paths: set[str] = set()
    _schema_ready_lock = threading.Lock()
    _wal_ready_lock = threading.Lock()

    def __init__(
        self,
        dataset_root: Path,
        db_file: str = DEFAULT_DATASET_DB_FILE,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.db_path = resolve_dataset_db_path(self.dataset_root, db_file=db_file)

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.db_path), timeout=30)
        connection.row_factory = sqlite3.Row
        db_key = str(self.db_path.resolve())
        if db_key not in self._wal_ready_paths:
            with self._wal_ready_lock:
                if db_key not in self._wal_ready_paths:
                    connection.execute("PRAGMA journal_mode=WAL;")
                    self._wal_ready_paths.add(db_key)
        connection.execute("PRAGMA synchronous=NORMAL;")
        connection.execute("PRAGMA foreign_keys=ON;")
        return connection

    def ensure_schema(self) -> None:
        with self._connect() as connection:
            self._ensure_schema(connection)

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        db_key = str(self.db_path.resolve())
        if db_key in self._schema_ready_paths:
            return
        with self._schema_ready_lock:
            if db_key in self._schema_ready_paths:
                return
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_assets (
                asset_id TEXT PRIMARY KEY,
                asset_role TEXT NOT NULL CHECK(asset_role IN ('new', 'old', 'new_marked')),
                field_key TEXT NOT NULL,
                field_name TEXT NOT NULL,
                capture_key TEXT NOT NULL,
                relpath TEXT NOT NULL UNIQUE,
                file_name TEXT NOT NULL,
                file_stem TEXT NOT NULL,
                suffix TEXT NOT NULL,
                date_obs TEXT,
                size_bytes INTEGER NOT NULL DEFAULT 0,
                modified_time REAL NOT NULL DEFAULT 0,
                metadata_json TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_raw_assets_role_field
            ON raw_assets(asset_role, field_key)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_raw_assets_role_capture
            ON raw_assets(asset_role, capture_key)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                field_key TEXT NOT NULL,
                field_name TEXT NOT NULL,
                capture_key TEXT NOT NULL,
                new_asset_id TEXT NOT NULL UNIQUE,
                old_asset_id TEXT,
                new_marked_asset_id TEXT,
                preprocess_status TEXT NOT NULL DEFAULT 'pending',
                current_source_view TEXT,
                current_label TEXT,
                current_detail_type TEXT,
                current_annotation_count INTEGER NOT NULL DEFAULT 0,
                current_ai_suggestion TEXT,
                current_ai_confidence REAL,
                annotation_updated_at TEXT,
                local_viewed_at TEXT,
                local_annotation_status TEXT,
                online_annotation_status TEXT,
                claim_client_id TEXT,
                claim_locked_at TEXT,
                claim_expires_at TEXT,
                crop_x0 INTEGER,
                crop_x1 INTEGER,
                crop_y0 INTEGER,
                crop_y1 INTEGER,
                align_dx REAL,
                align_dy REAL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(new_asset_id) REFERENCES raw_assets(asset_id),
                FOREIGN KEY(old_asset_id) REFERENCES raw_assets(asset_id),
                FOREIGN KEY(new_marked_asset_id) REFERENCES raw_assets(asset_id)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tasks_field_key
            ON tasks(field_key)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tasks_claim_expires
            ON tasks(claim_expires_at)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tasks_claim_client
            ON tasks(claim_client_id)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS task_artifacts (
                task_id TEXT NOT NULL,
                artifact_role TEXT NOT NULL CHECK(
                    artifact_role IN (
                        'aligned_new',
                        'aligned_old',
                        'aligned_new_marked',
                        'new_marker',
                        'old_marker'
                    )
                ),
                relpath TEXT NOT NULL,
                width INTEGER,
                height INTEGER,
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(task_id, artifact_role),
                FOREIGN KEY(task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS task_annotation_boxes_current (
                task_id TEXT NOT NULL,
                box_index INTEGER NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                width REAL NOT NULL,
                height REAL NOT NULL,
                label TEXT,
                detail_type TEXT,
                confidence REAL,
                PRIMARY KEY(task_id, box_index),
                FOREIGN KEY(task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS annotation_revisions (
                revision_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                source_view TEXT,
                parent_revision_id TEXT,
                rollback_of_revision_id TEXT,
                submitted_by TEXT NOT NULL,
                origin TEXT NOT NULL DEFAULT 'unknown',
                saved_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                FOREIGN KEY(task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_annotation_revisions_task_saved_at
            ON annotation_revisions(task_id, saved_at)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS annotation_revision_boxes (
                revision_id TEXT NOT NULL,
                box_index INTEGER NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                width REAL NOT NULL,
                height REAL NOT NULL,
                label TEXT,
                detail_type TEXT,
                confidence REAL,
                PRIMARY KEY(revision_id, box_index),
                FOREIGN KEY(revision_id) REFERENCES annotation_revisions(revision_id) ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prelabel_jobs (
                job_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                requested_by TEXT NOT NULL,
                model_version TEXT NOT NULL,
                model_id TEXT,
                model_backbone TEXT,
                candidate_limit INTEGER,
                confidence_threshold REAL,
                input_fingerprint TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('queued', 'claimed', 'completed', 'failed', 'cancelled')),
                priority INTEGER NOT NULL DEFAULT 100,
                claim_worker_id TEXT,
                claimed_at TEXT,
                claim_expires_at TEXT,
                last_heartbeat_at TEXT,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                error_message TEXT,
                result_prelabel_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_prelabel_jobs_status_priority
            ON prelabel_jobs(status, priority DESC, created_at)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_prelabel_jobs_task_id
            ON prelabel_jobs(task_id, created_at)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS task_ai_prelabels (
                prelabel_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                job_id TEXT,
                source_view TEXT,
                ai_suggestion TEXT,
                ai_confidence REAL,
                model_version TEXT,
                model_id TEXT,
                model_backbone TEXT,
                candidate_limit INTEGER,
                confidence_threshold REAL,
                input_fingerprint TEXT,
                status TEXT NOT NULL CHECK(status IN ('available', 'superseded', 'hidden', 'accepted')),
                box_count INTEGER NOT NULL DEFAULT 0,
                worker_id TEXT,
                accepted_revision_id TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                superseded_at TEXT,
                FOREIGN KEY(task_id) REFERENCES tasks(task_id) ON DELETE CASCADE,
                FOREIGN KEY(job_id) REFERENCES prelabel_jobs(job_id) ON DELETE SET NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_task_ai_prelabels_task_status
            ON task_ai_prelabels(task_id, status, updated_at)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS task_ai_prelabel_boxes (
                prelabel_id TEXT NOT NULL,
                box_index INTEGER NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                width REAL NOT NULL,
                height REAL NOT NULL,
                label TEXT,
                detail_type TEXT,
                confidence REAL,
                PRIMARY KEY(prelabel_id, box_index),
                FOREIGN KEY(prelabel_id) REFERENCES task_ai_prelabels(prelabel_id) ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS worker_nodes (
                worker_id TEXT PRIMARY KEY,
                display_name TEXT,
                host_name TEXT,
                device_label TEXT,
                status TEXT NOT NULL DEFAULT 'online',
                capabilities_json TEXT NOT NULL DEFAULT '{}',
                last_seen_at TEXT,
                last_claimed_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS dataset_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                snapshot_name TEXT NOT NULL,
                document_relpath TEXT NOT NULL,
                task_count INTEGER NOT NULL DEFAULT 0,
                annotation_count INTEGER NOT NULL DEFAULT 0,
                created_by TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_dataset_snapshots_created_at
            ON dataset_snapshots(created_at)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS training_jobs (
                job_id TEXT PRIMARY KEY,
                snapshot_id TEXT NOT NULL,
                requested_by TEXT NOT NULL,
                task_type TEXT NOT NULL,
                model_version TEXT NOT NULL,
                model_id TEXT NOT NULL,
                model_backbone TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('queued', 'claimed', 'completed', 'failed', 'cancelled')),
                train_config_json TEXT NOT NULL DEFAULT '{}',
                priority INTEGER NOT NULL DEFAULT 100,
                promote_on_success INTEGER NOT NULL DEFAULT 0,
                enqueue_prelabels_on_success INTEGER NOT NULL DEFAULT 0,
                prelabel_task_ids_json TEXT NOT NULL DEFAULT '[]',
                force_prelabel INTEGER NOT NULL DEFAULT 0,
                claim_worker_id TEXT,
                claimed_at TEXT,
                claim_expires_at TEXT,
                last_heartbeat_at TEXT,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                error_message TEXT,
                run_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(snapshot_id) REFERENCES dataset_snapshots(snapshot_id) ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_training_jobs_status_priority
            ON training_jobs(status, priority DESC, created_at)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS training_runs (
                run_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                snapshot_id TEXT NOT NULL,
                task_type TEXT NOT NULL,
                status TEXT NOT NULL,
                worker_id TEXT,
                model_id TEXT,
                model_version TEXT,
                model_backbone TEXT,
                artifact_path TEXT,
                metrics_json TEXT NOT NULL DEFAULT '{}',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                started_at TEXT,
                finished_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(job_id) REFERENCES training_jobs(job_id) ON DELETE CASCADE,
                FOREIGN KEY(snapshot_id) REFERENCES dataset_snapshots(snapshot_id) ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_training_runs_job_id
            ON training_runs(job_id)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS model_registry (
                model_id TEXT PRIMARY KEY,
                model_version TEXT NOT NULL,
                model_backbone TEXT NOT NULL,
                task_type TEXT NOT NULL,
                training_run_id TEXT,
                snapshot_id TEXT,
                artifact_path TEXT,
                metrics_json TEXT NOT NULL DEFAULT '{}',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_by TEXT,
                is_promoted INTEGER NOT NULL DEFAULT 0,
                promoted_at TEXT,
                promoted_by TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(training_run_id) REFERENCES training_runs(run_id) ON DELETE SET NULL,
                FOREIGN KEY(snapshot_id) REFERENCES dataset_snapshots(snapshot_id) ON DELETE SET NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_model_registry_task_type_promoted
            ON model_registry(task_type, is_promoted, created_at)
            """
        )
        self._ensure_column(connection, "prelabel_jobs", "model_id", "TEXT")
        self._ensure_column(connection, "prelabel_jobs", "model_backbone", "TEXT")
        self._ensure_column(connection, "prelabel_jobs", "candidate_limit", "INTEGER")
        self._ensure_column(connection, "prelabel_jobs", "confidence_threshold", "REAL")
        self._ensure_column(connection, "task_ai_prelabels", "model_id", "TEXT")
        self._ensure_column(connection, "task_ai_prelabels", "model_backbone", "TEXT")
        self._ensure_column(connection, "task_ai_prelabels", "candidate_limit", "INTEGER")
        self._ensure_column(connection, "task_ai_prelabels", "confidence_threshold", "REAL")
        connection.commit()
        self._schema_ready_paths.add(db_key)

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection,
        table_name: str,
        column_name: str,
        column_sql: str,
    ) -> None:
        columns = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        if any(str(column["name"]) == column_name for column in columns):
            return
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}")

    @staticmethod
    def strip_explorer_copy_suffix(stem: str) -> tuple[str, int]:
        normalized = stem.strip()
        match = _EXPLORER_COPY_SUFFIX_RE.match(normalized)
        if match is None:
            return normalized, 0
        base = str(match.group("base") or "").strip()
        try:
            index = int(match.group("index") or "0")
        except ValueError:
            index = 0
        return base or normalized, max(index, 0)

    @staticmethod
    def _strip_datetime_prefix(stem: str) -> str:
        if len(stem) >= 17:
            prefix = stem[:15]
            if (
                prefix[0:8].isdigit()
                and prefix[8].lower() == "t"
                and prefix[9:15].isdigit()
                and stem[15:17] == "__"
            ):
                return stem[17:]
        return stem

    @staticmethod
    def strip_aligned_crop_suffix(stem: str) -> str:
        suffix = "__aligned_crop"
        if stem.lower().endswith(suffix):
            return stem[:-len(suffix)]
        return stem

    @classmethod
    def normalize_field_name(cls, stem: str) -> str:
        value = cls._strip_datetime_prefix(stem)
        for prefix in ("FW_", "fw_", "Fw_"):
            if value.startswith(prefix):
                value = value[len(prefix):]
                break
        value = cls.strip_aligned_crop_suffix(value)
        value, _copy_index = cls.strip_explorer_copy_suffix(value)
        return value.strip()

    @classmethod
    def normalize_field_key(cls, stem: str) -> str:
        return cls.normalize_field_name(stem).strip().lower()

    @classmethod
    def normalize_capture_key(cls, stem: str) -> str:
        value = cls._strip_datetime_prefix(stem)
        for prefix in ("FW_", "fw_", "Fw_"):
            if value.startswith(prefix):
                value = value[len(prefix):]
                break
        value = cls.strip_aligned_crop_suffix(value)
        return value.strip().lower()

    def upsert_raw_assets(self, assets: Iterable[RawAssetRecord]) -> None:
        now = _utc_now_iso()
        asset_list = list(assets)
        active_relpaths = {asset.relpath for asset in asset_list}
        with self._connect() as connection:
            self._ensure_schema(connection)
            connection.execute("UPDATE raw_assets SET status = 'missing', updated_at = ?", (now,))
            for asset in asset_list:
                existing = connection.execute(
                    "SELECT created_at FROM raw_assets WHERE relpath = ?",
                    (asset.relpath,),
                ).fetchone()
                created_at = str(existing["created_at"]) if existing is not None else now
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
                        date_obs,
                        size_bytes,
                        modified_time,
                        metadata_json,
                        status,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
                    ON CONFLICT(relpath) DO UPDATE SET
                        asset_id=excluded.asset_id,
                        asset_role=excluded.asset_role,
                        field_key=excluded.field_key,
                        field_name=excluded.field_name,
                        capture_key=excluded.capture_key,
                        file_name=excluded.file_name,
                        file_stem=excluded.file_stem,
                        suffix=excluded.suffix,
                        date_obs=excluded.date_obs,
                        size_bytes=excluded.size_bytes,
                        modified_time=excluded.modified_time,
                        metadata_json=excluded.metadata_json,
                        status='active',
                        updated_at=excluded.updated_at
                    """,
                    (
                        asset.asset_id,
                        asset.asset_role,
                        asset.field_key,
                        asset.field_name,
                        asset.capture_key,
                        asset.relpath,
                        asset.file_name,
                        asset.file_stem,
                        asset.suffix,
                        asset.date_obs,
                        int(asset.size_bytes),
                        float(asset.modified_time),
                        json.dumps(asset.metadata or {}, ensure_ascii=False),
                        created_at,
                        now,
                    ),
                )
            if active_relpaths:
                placeholders = ",".join("?" for _ in active_relpaths)
                connection.execute(
                    f"UPDATE raw_assets SET status = 'missing', updated_at = ? WHERE relpath NOT IN ({placeholders})",
                    (now, *sorted(active_relpaths)),
                )
            connection.commit()

    def list_raw_assets(self, asset_role: str, *, active_only: bool = True) -> list[RawAssetRecord]:
        query = (
            "SELECT * FROM raw_assets WHERE asset_role = ?"
            + (" AND status = 'active'" if active_only else "")
            + " ORDER BY field_name, capture_key, file_name"
        )
        with self._connect() as connection:
            self._ensure_schema(connection)
            rows = connection.execute(query, (asset_role,)).fetchall()
        return [self._row_to_raw_asset(row) for row in rows]

    def allocate_task_id(
        self,
        *,
        date_token: str | None,
        field_name: str,
        capture_key: str,
    ) -> str:
        safe_field = field_name.strip() or capture_key.strip() or "task"
        base = f"{date_token}__{safe_field}" if date_token else safe_field
        with self._connect() as connection:
            self._ensure_schema(connection)
            existing = {
                str(row["task_id"])
                for row in connection.execute("SELECT task_id FROM tasks").fetchall()
            }
        if base not in existing:
            return base
        index = 1
        while True:
            candidate = f"{base}__{index:02d}"
            if candidate not in existing:
                return candidate
            index += 1

    def sync_tasks(self, tasks: Iterable[TaskRecord]) -> None:
        task_list = list(tasks)
        active_new_asset_ids = {task.new_asset_id for task in task_list}
        now = _utc_now_iso()
        with self._connect() as connection:
            self._ensure_schema(connection)
            existing_rows = connection.execute(
                "SELECT task_id, new_asset_id, created_at FROM tasks"
            ).fetchall()
            existing_by_new_asset = {str(row["new_asset_id"]): row for row in existing_rows}
            for task in task_list:
                existing_row = existing_by_new_asset.get(task.new_asset_id)
                task_id = str(existing_row["task_id"]) if existing_row is not None else task.task_id
                created_at = str(existing_row["created_at"]) if existing_row is not None else now
                connection.execute(
                    """
                    INSERT INTO tasks (
                        task_id,
                        field_key,
                        field_name,
                        capture_key,
                        new_asset_id,
                        old_asset_id,
                        new_marked_asset_id,
                        preprocess_status,
                        current_source_view,
                        current_label,
                        current_detail_type,
                        current_annotation_count,
                        current_ai_suggestion,
                        current_ai_confidence,
                        annotation_updated_at,
                        local_viewed_at,
                        local_annotation_status,
                        online_annotation_status,
                        claim_client_id,
                        claim_locked_at,
                        claim_expires_at,
                        crop_x0,
                        crop_x1,
                        crop_y0,
                        crop_y1,
                        align_dx,
                        align_dy,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(task_id) DO UPDATE SET
                        field_key=excluded.field_key,
                        field_name=excluded.field_name,
                        capture_key=excluded.capture_key,
                        new_asset_id=excluded.new_asset_id,
                        old_asset_id=excluded.old_asset_id,
                        new_marked_asset_id=excluded.new_marked_asset_id,
                        updated_at=excluded.updated_at
                    """,
                    (
                        task_id,
                        task.field_key,
                        task.field_name,
                        task.capture_key,
                        task.new_asset_id,
                        task.old_asset_id,
                        task.new_marked_asset_id,
                        task.preprocess_status,
                        task.current_source_view,
                        task.current_label,
                        task.current_detail_type,
                        int(task.current_annotation_count),
                        task.current_ai_suggestion,
                        task.current_ai_confidence,
                        task.annotation_updated_at,
                        task.local_viewed_at,
                        task.local_annotation_status,
                        task.online_annotation_status,
                        task.claim_client_id,
                        task.claim_locked_at,
                        task.claim_expires_at,
                        task.crop_x0,
                        task.crop_x1,
                        task.crop_y0,
                        task.crop_y1,
                        task.align_dx,
                        task.align_dy,
                        created_at,
                        now,
                    ),
                )
            if active_new_asset_ids:
                placeholders = ",".join("?" for _ in active_new_asset_ids)
                connection.execute(
                    f"UPDATE tasks SET preprocess_status = 'missing', updated_at = ? WHERE new_asset_id NOT IN ({placeholders})",
                    (now, *sorted(active_new_asset_ids)),
                )
            connection.commit()

    def list_tasks(self, *, active_only: bool = True) -> list[TaskRecord]:
        query = "SELECT * FROM tasks"
        if active_only:
            query += " WHERE preprocess_status != 'missing'"
        query += " ORDER BY task_id"
        with self._connect() as connection:
            self._ensure_schema(connection)
            rows = connection.execute(query).fetchall()
        return [self._row_to_task(row) for row in rows]

    def get_task_by_new_asset_id(self, new_asset_id: str) -> TaskRecord | None:
        with self._connect() as connection:
            self._ensure_schema(connection)
            row = connection.execute("SELECT * FROM tasks WHERE new_asset_id = ?", (new_asset_id,)).fetchone()
        return self._row_to_task(row) if row is not None else None

    def get_task_by_id(self, task_id: str) -> TaskRecord | None:
        with self._connect() as connection:
            self._ensure_schema(connection)
            row = connection.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        return self._row_to_task(row) if row is not None else None

    def update_task_preprocess_state(
        self,
        task_id: str,
        *,
        preprocess_status: str,
        crop_bounds: tuple[int, int, int, int] | None = None,
        align_dx: float | None = None,
        align_dy: float | None = None,
    ) -> None:
        now = _utc_now_iso()
        crop_x0 = crop_x1 = crop_y0 = crop_y1 = None
        if crop_bounds is not None:
            crop_x0, crop_x1, crop_y0, crop_y1 = crop_bounds
        with self._connect() as connection:
            self._ensure_schema(connection)
            connection.execute(
                """
                UPDATE tasks
                SET preprocess_status = ?,
                    crop_x0 = ?,
                    crop_x1 = ?,
                    crop_y0 = ?,
                    crop_y1 = ?,
                    align_dx = ?,
                    align_dy = ?,
                    updated_at = ?
                WHERE task_id = ?
                """,
                (
                    preprocess_status,
                    crop_x0,
                    crop_x1,
                    crop_y0,
                    crop_y1,
                    align_dx,
                    align_dy,
                    now,
                    task_id,
                ),
            )
            connection.commit()

    def upsert_task_artifact(self, artifact: TaskArtifactRecord) -> None:
        now = _utc_now_iso()
        with self._connect() as connection:
            self._ensure_schema(connection)
            existing = connection.execute(
                "SELECT created_at FROM task_artifacts WHERE task_id = ? AND artifact_role = ?",
                (artifact.task_id, artifact.artifact_role),
            ).fetchone()
            created_at = str(existing["created_at"]) if existing is not None else now
            connection.execute(
                """
                INSERT INTO task_artifacts (
                    task_id,
                    artifact_role,
                    relpath,
                    width,
                    height,
                    metadata_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id, artifact_role) DO UPDATE SET
                    relpath=excluded.relpath,
                    width=excluded.width,
                    height=excluded.height,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    artifact.task_id,
                    artifact.artifact_role,
                    artifact.relpath,
                    artifact.width,
                    artifact.height,
                    json.dumps(artifact.metadata or {}, ensure_ascii=False),
                    created_at,
                    now,
                ),
            )
            connection.commit()

    def get_task_artifact_path(self, task_id: str, artifact_role: str) -> Path | None:
        with self._connect() as connection:
            self._ensure_schema(connection)
            row = connection.execute(
                "SELECT relpath FROM task_artifacts WHERE task_id = ? AND artifact_role = ?",
                (task_id, artifact_role),
            ).fetchone()
        if row is None:
            return None
        return self.dataset_root / str(row["relpath"])

    def list_prepared_task_paths(self) -> list[dict[str, str | None]]:
        with self._connect() as connection:
            self._ensure_schema(connection)
            rows = connection.execute(
                """
                SELECT
                    t.task_id AS task_id,
                    COALESCE(an.relpath, rn.relpath) AS new_path,
                    COALESCE(ao.relpath, ro.relpath) AS old_path,
                    COALESCE(am.relpath, rm.relpath) AS new_marked_path
                FROM tasks t
                LEFT JOIN raw_assets rn
                    ON rn.asset_id = t.new_asset_id AND rn.status = 'active'
                LEFT JOIN raw_assets ro
                    ON ro.asset_id = t.old_asset_id AND ro.status = 'active'
                LEFT JOIN raw_assets rm
                    ON rm.asset_id = t.new_marked_asset_id AND rm.status = 'active'
                LEFT JOIN task_artifacts an
                    ON an.task_id = t.task_id AND an.artifact_role = 'aligned_new'
                LEFT JOIN task_artifacts ao
                    ON ao.task_id = t.task_id AND ao.artifact_role = 'aligned_old'
                LEFT JOIN task_artifacts am
                    ON am.task_id = t.task_id AND am.artifact_role = 'aligned_new_marked'
                WHERE t.preprocess_status IN ('ready', 'annotated', 'claimed', 'viewed', 'align_failed')
                  AND COALESCE(an.relpath, rn.relpath) IS NOT NULL
                ORDER BY t.task_id
                """
            ).fetchall()
        prepared: list[dict[str, str | None]] = []
        for row in rows:
            new_rel = str(row["new_path"]) if row["new_path"] is not None else None
            if not new_rel or not (self.dataset_root / new_rel).is_file():
                continue

            old_rel = str(row["old_path"]) if row["old_path"] is not None else None
            if old_rel and not (self.dataset_root / old_rel).is_file():
                old_rel = None

            marked_rel = str(row["new_marked_path"]) if row["new_marked_path"] is not None else None
            if marked_rel and not (self.dataset_root / marked_rel).is_file():
                marked_rel = None

            prepared.append(
                {
                    "task_id": str(row["task_id"]),
                    "new_path": new_rel,
                    "old_path": old_rel,
                    "new_marked_path": marked_rel,
                }
            )
        return prepared

    def mark_task_viewed(self, task_id: str, viewed_at: str | None = None) -> None:
        now = viewed_at or _utc_now_iso()
        with self._connect() as connection:
            self._ensure_schema(connection)
            connection.execute(
                """
                UPDATE tasks
                SET local_viewed_at = ?,
                    preprocess_status = CASE
                        WHEN preprocess_status = 'ready' THEN 'viewed'
                        ELSE preprocess_status
                    END,
                    updated_at = ?
                WHERE task_id = ?
                """,
                (now, now, task_id),
            )
            connection.commit()

    def upsert_current_annotation(
        self,
        *,
        task_id: str,
        source_view: str | None,
        label: str | None,
        detail_type: str | None,
        ai_suggestion: str | None,
        ai_confidence: float | None,
        annotations: list[dict[str, Any]],
        annotation_origin: str = "local",
    ) -> None:
        now = _utc_now_iso()
        count = len(annotations)
        annotation_status = "annotated" if count > 0 or label or detail_type else "unlabeled"
        local_status = annotation_status if annotation_origin in {"local", "both"} else None
        online_status = annotation_status if annotation_origin in {"online", "both"} else None
        with self._connect() as connection:
            self._ensure_schema(connection)
            connection.execute(
                """
                UPDATE tasks
                SET current_source_view = ?,
                    current_label = ?,
                    current_detail_type = ?,
                    current_annotation_count = ?,
                    current_ai_suggestion = ?,
                    current_ai_confidence = ?,
                    annotation_updated_at = ?,
                    local_annotation_status = CASE
                        WHEN ? IS NOT NULL THEN ?
                        ELSE local_annotation_status
                    END,
                    online_annotation_status = CASE
                        WHEN ? IS NOT NULL THEN ?
                        ELSE online_annotation_status
                    END,
                    updated_at = ?
                WHERE task_id = ?
                """,
                (
                    source_view,
                    label,
                    detail_type,
                    count,
                    ai_suggestion,
                    ai_confidence,
                    now,
                    local_status,
                    local_status,
                    online_status,
                    online_status,
                    now,
                    task_id,
                ),
            )
            connection.execute("DELETE FROM task_annotation_boxes_current WHERE task_id = ?", (task_id,))
            for idx, ann in enumerate(annotations):
                connection.execute(
                    """
                    INSERT INTO task_annotation_boxes_current (
                        task_id, box_index, x, y, width, height, label, detail_type, confidence
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_id,
                        idx,
                        float(ann.get("x", 0)),
                        float(ann.get("y", 0)),
                        float(ann.get("width", 0)),
                        float(ann.get("height", 0)),
                        ann.get("label"),
                        ann.get("detail_type"),
                        float(ann.get("confidence", 1.0)),
                    ),
                )
            connection.commit()

    def append_annotation_revision(
        self,
        *,
        task_id: str,
        source_view: str | None,
        parent_revision_id: str | None,
        rollback_of_revision_id: str | None,
        submitted_by: str,
        origin: str,
        saved_at: str,
        metadata: dict[str, Any],
        annotations: list[dict[str, Any]],
        revision_id: str | None = None,
    ) -> str:
        revision_value = revision_id or uuid.uuid4().hex
        with self._connect() as connection:
            self._ensure_schema(connection)
            connection.execute(
                """
                INSERT INTO annotation_revisions (
                    revision_id,
                    task_id,
                    source_view,
                    parent_revision_id,
                    rollback_of_revision_id,
                    submitted_by,
                    origin,
                    saved_at,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    revision_value,
                    task_id,
                    source_view,
                    parent_revision_id,
                    rollback_of_revision_id,
                    submitted_by,
                    origin,
                    saved_at,
                    json.dumps(metadata, ensure_ascii=False),
                ),
            )
            for idx, ann in enumerate(annotations):
                connection.execute(
                    """
                    INSERT INTO annotation_revision_boxes (
                        revision_id, box_index, x, y, width, height, label, detail_type, confidence
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        revision_value,
                        idx,
                        float(ann.get("x", 0)),
                        float(ann.get("y", 0)),
                        float(ann.get("width", 0)),
                        float(ann.get("height", 0)),
                        ann.get("label"),
                        ann.get("detail_type"),
                        float(ann.get("confidence", 1.0)),
                    ),
                )
            connection.commit()
        return revision_value

    def list_annotation_revisions(self, task_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            self._ensure_schema(connection)
            rows = connection.execute(
                """
                SELECT
                    revision_id,
                    task_id,
                    source_view,
                    parent_revision_id,
                    rollback_of_revision_id,
                    submitted_by,
                    origin,
                    saved_at,
                    metadata_json
                FROM annotation_revisions
                WHERE task_id = ?
                ORDER BY rowid ASC
                """,
                (task_id,),
            ).fetchall()
            box_rows = connection.execute(
                """
                SELECT revision_id, box_index, x, y, width, height, label, detail_type, confidence
                FROM annotation_revision_boxes
                WHERE revision_id IN (
                    SELECT revision_id FROM annotation_revisions WHERE task_id = ?
                )
                ORDER BY revision_id ASC, box_index ASC
                """,
                (task_id,),
            ).fetchall()

        annotations_by_revision: dict[str, list[dict[str, Any]]] = {}
        for row in box_rows:
            revision_id = str(row["revision_id"])
            annotations_by_revision.setdefault(revision_id, []).append(
                {
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "width": float(row["width"]),
                    "height": float(row["height"]),
                    "label": row["label"],
                    "detail_type": row["detail_type"],
                    "confidence": float(row["confidence"]) if row["confidence"] is not None else 1.0,
                }
            )

        revisions: list[dict[str, Any]] = []
        for row in rows:
            try:
                metadata = json.loads(str(row["metadata_json"] or "{}"))
            except Exception:
                metadata = {}
            revision_id = str(row["revision_id"])
            revisions.append(
                {
                    "revision_id": revision_id,
                    "task_id": str(row["task_id"]),
                    "source_view": str(row["source_view"]) if row["source_view"] is not None else None,
                    "parent_revision_id": (
                        str(row["parent_revision_id"]) if row["parent_revision_id"] is not None else None
                    ),
                    "rollback_of_revision_id": (
                        str(row["rollback_of_revision_id"]) if row["rollback_of_revision_id"] is not None else None
                    ),
                    "submitted_by": str(row["submitted_by"]),
                    "origin": str(row["origin"]),
                    "saved_at": str(row["saved_at"]),
                    "metadata": metadata if isinstance(metadata, dict) else {},
                    "annotations": annotations_by_revision.get(revision_id, []),
                }
            )
        return revisions

    def list_all_annotation_revisions(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            self._ensure_schema(connection)
            rows = connection.execute(
                """
                SELECT
                    rowid AS storage_rowid,
                    revision_id,
                    task_id,
                    source_view,
                    parent_revision_id,
                    rollback_of_revision_id,
                    submitted_by,
                    origin,
                    saved_at,
                    metadata_json
                FROM annotation_revisions
                ORDER BY task_id ASC, rowid ASC
                """
            ).fetchall()
            box_rows = connection.execute(
                """
                SELECT revision_id, box_index, x, y, width, height, label, detail_type, confidence
                FROM annotation_revision_boxes
                ORDER BY revision_id ASC, box_index ASC
                """
            ).fetchall()

        annotations_by_revision: dict[str, list[dict[str, Any]]] = {}
        for row in box_rows:
            revision_id = str(row["revision_id"])
            annotations_by_revision.setdefault(revision_id, []).append(
                {
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "width": float(row["width"]),
                    "height": float(row["height"]),
                    "label": row["label"],
                    "detail_type": row["detail_type"],
                    "confidence": float(row["confidence"]) if row["confidence"] is not None else 1.0,
                }
            )

        revisions: list[dict[str, Any]] = []
        for row in rows:
            try:
                metadata = json.loads(str(row["metadata_json"] or "{}"))
            except Exception:
                metadata = {}
            revision_id = str(row["revision_id"])
            revisions.append(
                {
                    "storage_rowid": int(row["storage_rowid"]),
                    "revision_id": revision_id,
                    "task_id": str(row["task_id"]),
                    "source_view": str(row["source_view"]) if row["source_view"] is not None else None,
                    "parent_revision_id": (
                        str(row["parent_revision_id"]) if row["parent_revision_id"] is not None else None
                    ),
                    "rollback_of_revision_id": (
                        str(row["rollback_of_revision_id"]) if row["rollback_of_revision_id"] is not None else None
                    ),
                    "submitted_by": str(row["submitted_by"]),
                    "origin": str(row["origin"]),
                    "saved_at": str(row["saved_at"]),
                    "metadata": metadata if isinstance(metadata, dict) else {},
                    "annotations": annotations_by_revision.get(revision_id, []),
                }
            )
        return revisions

    def list_annotation_revisions_after_rowid(self, after_rowid: int = 0) -> list[dict[str, Any]]:
        with self._connect() as connection:
            self._ensure_schema(connection)
            rows = connection.execute(
                """
                SELECT
                    rowid AS storage_rowid,
                    revision_id,
                    task_id,
                    source_view,
                    parent_revision_id,
                    rollback_of_revision_id,
                    submitted_by,
                    origin,
                    saved_at,
                    metadata_json
                FROM annotation_revisions
                WHERE rowid > ?
                ORDER BY rowid ASC
                """,
                (int(after_rowid),),
            ).fetchall()
            revision_ids = [str(row["revision_id"]) for row in rows]
            box_rows: list[sqlite3.Row] = []
            if revision_ids:
                placeholders = ",".join("?" for _ in revision_ids)
                box_rows = connection.execute(
                    f"""
                    SELECT revision_id, box_index, x, y, width, height, label, detail_type, confidence
                    FROM annotation_revision_boxes
                    WHERE revision_id IN ({placeholders})
                    ORDER BY revision_id ASC, box_index ASC
                    """,
                    tuple(revision_ids),
                ).fetchall()

        annotations_by_revision: dict[str, list[dict[str, Any]]] = {}
        for row in box_rows:
            revision_id = str(row["revision_id"])
            annotations_by_revision.setdefault(revision_id, []).append(
                {
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "width": float(row["width"]),
                    "height": float(row["height"]),
                    "label": row["label"],
                    "detail_type": row["detail_type"],
                    "confidence": float(row["confidence"]) if row["confidence"] is not None else 1.0,
                }
            )

        revisions: list[dict[str, Any]] = []
        for row in rows:
            try:
                metadata = json.loads(str(row["metadata_json"] or "{}"))
            except Exception:
                metadata = {}
            revision_id = str(row["revision_id"])
            revisions.append(
                {
                    "storage_rowid": int(row["storage_rowid"]),
                    "revision_id": revision_id,
                    "task_id": str(row["task_id"]),
                    "source_view": str(row["source_view"]) if row["source_view"] is not None else None,
                    "parent_revision_id": (
                        str(row["parent_revision_id"]) if row["parent_revision_id"] is not None else None
                    ),
                    "rollback_of_revision_id": (
                        str(row["rollback_of_revision_id"]) if row["rollback_of_revision_id"] is not None else None
                    ),
                    "submitted_by": str(row["submitted_by"]),
                    "origin": str(row["origin"]),
                    "saved_at": str(row["saved_at"]),
                    "metadata": metadata if isinstance(metadata, dict) else {},
                    "annotations": annotations_by_revision.get(revision_id, []),
                }
            )
        return revisions

    def list_current_annotations(self) -> dict[str, dict[str, Any]]:
        with self._connect() as connection:
            self._ensure_schema(connection)
            rows = connection.execute(
                """
                SELECT
                    t.task_id,
                    t.current_source_view,
                    t.current_label,
                    t.current_detail_type,
                    t.current_ai_suggestion,
                    t.current_ai_confidence,
                    rev.metadata_json AS latest_metadata_json,
                    COALESCE(an.relpath, rn.relpath) AS new_path,
                    COALESCE(ao.relpath, ro.relpath) AS old_path,
                    COALESCE(am.relpath, rm.relpath) AS new_marked_path
                FROM tasks t
                LEFT JOIN raw_assets rn
                    ON rn.asset_id = t.new_asset_id AND rn.status = 'active'
                LEFT JOIN raw_assets ro
                    ON ro.asset_id = t.old_asset_id AND ro.status = 'active'
                LEFT JOIN raw_assets rm
                    ON rm.asset_id = t.new_marked_asset_id AND rm.status = 'active'
                LEFT JOIN task_artifacts an
                    ON an.task_id = t.task_id AND an.artifact_role = 'aligned_new'
                LEFT JOIN task_artifacts ao
                    ON ao.task_id = t.task_id AND ao.artifact_role = 'aligned_old'
                LEFT JOIN task_artifacts am
                    ON am.task_id = t.task_id AND am.artifact_role = 'aligned_new_marked'
                LEFT JOIN (
                    SELECT ar.task_id, ar.metadata_json
                    FROM annotation_revisions ar
                    INNER JOIN (
                        SELECT task_id, MAX(rowid) AS latest_rowid
                        FROM annotation_revisions
                        GROUP BY task_id
                    ) latest
                        ON latest.task_id = ar.task_id AND latest.latest_rowid = ar.rowid
                ) rev
                    ON rev.task_id = t.task_id
                WHERE t.preprocess_status != 'missing'
                """
            ).fetchall()
            box_rows = connection.execute(
                """
                SELECT task_id, box_index, x, y, width, height, label, detail_type, confidence
                FROM task_annotation_boxes_current
                ORDER BY task_id ASC, box_index ASC
                """
            ).fetchall()

        annotations_by_task: dict[str, list[dict[str, Any]]] = {}
        for row in box_rows:
            task_id = str(row["task_id"])
            annotations_by_task.setdefault(task_id, []).append(
                {
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "width": float(row["width"]),
                    "height": float(row["height"]),
                    "label": row["label"],
                    "detail_type": row["detail_type"],
                    "confidence": float(row["confidence"]) if row["confidence"] is not None else 1.0,
                }
            )

        by_id: dict[str, dict[str, Any]] = {}
        for row in rows:
            task_id = str(row["task_id"])
            paths = {
                "new": str(row["new_path"]) if row["new_path"] is not None else "",
                "old": str(row["old_path"]) if row["old_path"] is not None else "",
                "new_marked": str(row["new_marked_path"]) if row["new_marked_path"] is not None else "",
            }
            file_name = Path(paths["new"] or paths["old"] or task_id).name
            record: dict[str, Any] = {
                "id": task_id,
                "file_name": file_name,
                "file": paths["new"] or "",
                "paths": paths,
            }
            if row["current_source_view"] is not None:
                record["source_view"] = str(row["current_source_view"])
            if row["current_label"] is not None:
                record["label"] = str(row["current_label"])
            if row["current_detail_type"] is not None:
                record["detail_type"] = str(row["current_detail_type"])
            if row["current_ai_suggestion"] is not None:
                record["ai_suggestion"] = str(row["current_ai_suggestion"])
            if row["current_ai_confidence"] is not None:
                record["ai_confidence"] = float(row["current_ai_confidence"])
            if row["latest_metadata_json"] is not None:
                try:
                    metadata = json.loads(str(row["latest_metadata_json"] or "{}"))
                except Exception:
                    metadata = {}
                if isinstance(metadata, dict) and metadata:
                    record["metadata"] = metadata
            annotations = annotations_by_task.get(task_id, [])
            if annotations:
                record["annotations"] = annotations
            by_id[task_id] = record
        return by_id

    def clear_expired_claims(self, now_iso: str) -> None:
        with self._connect() as connection:
            self._ensure_schema(connection)
            connection.execute(
                """
                UPDATE tasks
                SET claim_client_id = NULL,
                    claim_locked_at = NULL,
                    claim_expires_at = NULL,
                    preprocess_status = CASE
                        WHEN preprocess_status = 'claimed'
                             AND (
                                 lower(coalesce(local_annotation_status, '')) = 'annotated'
                                 OR lower(coalesce(online_annotation_status, '')) = 'annotated'
                             )
                            THEN 'annotated'
                        WHEN preprocess_status = 'claimed'
                            THEN 'ready'
                        ELSE preprocess_status
                    END,
                    updated_at = ?
                WHERE claim_expires_at IS NOT NULL AND claim_expires_at <= ?
                """,
                (now_iso, now_iso),
            )
            connection.execute(
                """
                WITH ranked_claims AS (
                    SELECT
                        task_id,
                        ROW_NUMBER() OVER (
                            PARTITION BY claim_client_id
                            ORDER BY claim_locked_at DESC, updated_at DESC, task_id DESC
                        ) AS claim_rank
                    FROM tasks
                    WHERE claim_client_id IS NOT NULL
                )
                UPDATE tasks
                SET claim_client_id = NULL,
                    claim_locked_at = NULL,
                    claim_expires_at = NULL,
                    preprocess_status = CASE
                        WHEN preprocess_status = 'claimed'
                             AND (
                                 lower(coalesce(local_annotation_status, '')) = 'annotated'
                                 OR lower(coalesce(online_annotation_status, '')) = 'annotated'
                             )
                            THEN 'annotated'
                        WHEN preprocess_status = 'claimed'
                            THEN 'ready'
                        ELSE preprocess_status
                    END,
                    updated_at = ?
                WHERE task_id IN (
                    SELECT task_id FROM ranked_claims WHERE claim_rank > 1
                )
                """,
                (now_iso,),
            )
            connection.commit()

    @staticmethod
    def _clear_other_claims_for_client(
        connection: sqlite3.Connection,
        *,
        client_id: str,
        keep_task_id: str,
        now_iso: str,
    ) -> None:
        connection.execute(
            """
            UPDATE tasks
            SET claim_client_id = NULL,
                claim_locked_at = NULL,
                claim_expires_at = NULL,
                preprocess_status = CASE
                    WHEN preprocess_status = 'claimed'
                         AND (
                             lower(coalesce(local_annotation_status, '')) = 'annotated'
                             OR lower(coalesce(online_annotation_status, '')) = 'annotated'
                         )
                        THEN 'annotated'
                    WHEN preprocess_status = 'claimed'
                        THEN 'ready'
                    ELSE preprocess_status
                END,
                updated_at = ?
            WHERE claim_client_id = ? AND task_id != ?
            """,
            (now_iso, client_id, keep_task_id),
        )

    def get_claimed_task_by_client(self, client_id: str) -> TaskRecord | None:
        with self._connect() as connection:
            self._ensure_schema(connection)
            row = connection.execute(
                """
                SELECT * FROM tasks
                WHERE claim_client_id = ?
                ORDER BY claim_locked_at DESC, updated_at DESC
                LIMIT 1
                """,
                (client_id,),
            ).fetchone()
        return self._row_to_task(row) if row is not None else None

    def list_tasks_by_ids(self, task_ids: Iterable[str]) -> list[TaskRecord]:
        normalized_ids = [task_id for task_id in dict.fromkeys(task_ids) if task_id]
        if not normalized_ids:
            return []
        placeholders = ",".join("?" for _ in normalized_ids)
        with self._connect() as connection:
            self._ensure_schema(connection)
            rows = connection.execute(
                f"""
                SELECT * FROM tasks
                WHERE task_id IN ({placeholders})
                """,
                tuple(normalized_ids),
            ).fetchall()
        tasks_by_id = {
            task.task_id: task
            for task in (self._row_to_task(row) for row in rows)
        }
        return [tasks_by_id[task_id] for task_id in normalized_ids if task_id in tasks_by_id]

    @staticmethod
    def _load_json_dict(raw: Any) -> dict[str, Any]:
        if raw is None:
            return {}
        try:
            parsed = json.loads(str(raw))
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _load_json_list(raw: Any) -> list[Any]:
        if raw is None:
            return []
        try:
            parsed = json.loads(str(raw))
        except Exception:
            return []
        return parsed if isinstance(parsed, list) else []

    @staticmethod
    def _expires_after(now: datetime, timeout_seconds: int) -> str:
        timeout = max(1, int(timeout_seconds))
        return (now + timedelta(seconds=timeout)).isoformat(timespec="seconds")

    @staticmethod
    def _row_to_prelabel_job(row: sqlite3.Row) -> PrelabelJobRecord:
        return PrelabelJobRecord(
            job_id=str(row["job_id"]),
            task_id=str(row["task_id"]),
            requested_by=str(row["requested_by"]),
            model_version=str(row["model_version"]),
            model_id=str(row["model_id"]) if row["model_id"] is not None else None,
            model_backbone=str(row["model_backbone"]) if row["model_backbone"] is not None else None,
            candidate_limit=int(row["candidate_limit"]) if row["candidate_limit"] is not None else None,
            confidence_threshold=(
                float(row["confidence_threshold"]) if row["confidence_threshold"] is not None else None
            ),
            input_fingerprint=str(row["input_fingerprint"]),
            status=str(row["status"]),
            priority=int(row["priority"] or 100),
            claim_worker_id=(str(row["claim_worker_id"]) if row["claim_worker_id"] is not None else None),
            claimed_at=(str(row["claimed_at"]) if row["claimed_at"] is not None else None),
            claim_expires_at=(str(row["claim_expires_at"]) if row["claim_expires_at"] is not None else None),
            last_heartbeat_at=(
                str(row["last_heartbeat_at"]) if row["last_heartbeat_at"] is not None else None
            ),
            attempt_count=int(row["attempt_count"] or 0),
            error_message=(str(row["error_message"]) if row["error_message"] is not None else None),
            result_prelabel_id=(
                str(row["result_prelabel_id"]) if row["result_prelabel_id"] is not None else None
            ),
            created_at=(str(row["created_at"]) if row["created_at"] is not None else None),
            updated_at=(str(row["updated_at"]) if row["updated_at"] is not None else None),
        )

    @classmethod
    def _row_to_ai_prelabel(cls, row: sqlite3.Row) -> TaskAIPrelabelRecord:
        return TaskAIPrelabelRecord(
            prelabel_id=str(row["prelabel_id"]),
            task_id=str(row["task_id"]),
            job_id=str(row["job_id"]) if row["job_id"] is not None else None,
            source_view=str(row["source_view"]) if row["source_view"] is not None else None,
            ai_suggestion=str(row["ai_suggestion"]) if row["ai_suggestion"] is not None else None,
            ai_confidence=float(row["ai_confidence"]) if row["ai_confidence"] is not None else None,
            model_version=str(row["model_version"]) if row["model_version"] is not None else None,
            model_id=str(row["model_id"]) if row["model_id"] is not None else None,
            model_backbone=str(row["model_backbone"]) if row["model_backbone"] is not None else None,
            candidate_limit=int(row["candidate_limit"]) if row["candidate_limit"] is not None else None,
            confidence_threshold=(
                float(row["confidence_threshold"]) if row["confidence_threshold"] is not None else None
            ),
            input_fingerprint=(
                str(row["input_fingerprint"]) if row["input_fingerprint"] is not None else None
            ),
            status=str(row["status"]),
            box_count=int(row["box_count"] or 0),
            worker_id=str(row["worker_id"]) if row["worker_id"] is not None else None,
            accepted_revision_id=(
                str(row["accepted_revision_id"]) if row["accepted_revision_id"] is not None else None
            ),
            metadata=cls._load_json_dict(row["metadata_json"]),
            created_at=str(row["created_at"]) if row["created_at"] is not None else None,
            updated_at=str(row["updated_at"]) if row["updated_at"] is not None else None,
            superseded_at=str(row["superseded_at"]) if row["superseded_at"] is not None else None,
        )

    @classmethod
    def _row_to_worker_node(cls, row: sqlite3.Row) -> WorkerNodeRecord:
        return WorkerNodeRecord(
            worker_id=str(row["worker_id"]),
            display_name=str(row["display_name"]) if row["display_name"] is not None else None,
            host_name=str(row["host_name"]) if row["host_name"] is not None else None,
            device_label=str(row["device_label"]) if row["device_label"] is not None else None,
            status=str(row["status"]),
            capabilities=cls._load_json_dict(row["capabilities_json"]),
            last_seen_at=str(row["last_seen_at"]) if row["last_seen_at"] is not None else None,
            last_claimed_at=str(row["last_claimed_at"]) if row["last_claimed_at"] is not None else None,
            created_at=str(row["created_at"]) if row["created_at"] is not None else None,
            updated_at=str(row["updated_at"]) if row["updated_at"] is not None else None,
        )

    @classmethod
    def _row_to_dataset_snapshot(cls, row: sqlite3.Row) -> DatasetSnapshotRecord:
        return DatasetSnapshotRecord(
            snapshot_id=str(row["snapshot_id"]),
            snapshot_name=str(row["snapshot_name"]),
            document_relpath=str(row["document_relpath"]),
            task_count=int(row["task_count"] or 0),
            annotation_count=int(row["annotation_count"] or 0),
            created_by=str(row["created_by"]) if row["created_by"] is not None else None,
            metadata=cls._load_json_dict(row["metadata_json"]),
            created_at=str(row["created_at"]) if row["created_at"] is not None else None,
            updated_at=str(row["updated_at"]) if row["updated_at"] is not None else None,
        )

    @classmethod
    def _row_to_training_job(cls, row: sqlite3.Row) -> TrainingJobRecord:
        return TrainingJobRecord(
            job_id=str(row["job_id"]),
            snapshot_id=str(row["snapshot_id"]),
            requested_by=str(row["requested_by"]),
            task_type=str(row["task_type"]),
            model_version=str(row["model_version"]),
            model_id=str(row["model_id"]),
            model_backbone=str(row["model_backbone"]),
            status=str(row["status"]),
            train_config=cls._load_json_dict(row["train_config_json"]),
            priority=int(row["priority"] or 100),
            promote_on_success=bool(int(row["promote_on_success"] or 0)),
            enqueue_prelabels_on_success=bool(int(row["enqueue_prelabels_on_success"] or 0)),
            prelabel_task_ids=[str(item) for item in cls._load_json_list(row["prelabel_task_ids_json"]) if str(item)],
            force_prelabel=bool(int(row["force_prelabel"] or 0)),
            claim_worker_id=str(row["claim_worker_id"]) if row["claim_worker_id"] is not None else None,
            claimed_at=str(row["claimed_at"]) if row["claimed_at"] is not None else None,
            claim_expires_at=str(row["claim_expires_at"]) if row["claim_expires_at"] is not None else None,
            last_heartbeat_at=str(row["last_heartbeat_at"]) if row["last_heartbeat_at"] is not None else None,
            attempt_count=int(row["attempt_count"] or 0),
            error_message=str(row["error_message"]) if row["error_message"] is not None else None,
            run_id=str(row["run_id"]) if row["run_id"] is not None else None,
            created_at=str(row["created_at"]) if row["created_at"] is not None else None,
            updated_at=str(row["updated_at"]) if row["updated_at"] is not None else None,
        )

    @classmethod
    def _row_to_training_run(cls, row: sqlite3.Row) -> TrainingRunRecord:
        return TrainingRunRecord(
            run_id=str(row["run_id"]),
            job_id=str(row["job_id"]),
            snapshot_id=str(row["snapshot_id"]),
            task_type=str(row["task_type"]),
            status=str(row["status"]),
            worker_id=str(row["worker_id"]) if row["worker_id"] is not None else None,
            model_id=str(row["model_id"]) if row["model_id"] is not None else None,
            model_version=str(row["model_version"]) if row["model_version"] is not None else None,
            model_backbone=str(row["model_backbone"]) if row["model_backbone"] is not None else None,
            artifact_path=str(row["artifact_path"]) if row["artifact_path"] is not None else None,
            metrics=cls._load_json_dict(row["metrics_json"]),
            metadata=cls._load_json_dict(row["metadata_json"]),
            started_at=str(row["started_at"]) if row["started_at"] is not None else None,
            finished_at=str(row["finished_at"]) if row["finished_at"] is not None else None,
            created_at=str(row["created_at"]) if row["created_at"] is not None else None,
            updated_at=str(row["updated_at"]) if row["updated_at"] is not None else None,
        )

    @classmethod
    def _row_to_registered_model(cls, row: sqlite3.Row) -> RegisteredModelRecord:
        return RegisteredModelRecord(
            model_id=str(row["model_id"]),
            model_version=str(row["model_version"]),
            model_backbone=str(row["model_backbone"]),
            task_type=str(row["task_type"]),
            training_run_id=str(row["training_run_id"]) if row["training_run_id"] is not None else None,
            snapshot_id=str(row["snapshot_id"]) if row["snapshot_id"] is not None else None,
            artifact_path=str(row["artifact_path"]) if row["artifact_path"] is not None else None,
            metrics=cls._load_json_dict(row["metrics_json"]),
            metadata=cls._load_json_dict(row["metadata_json"]),
            created_by=str(row["created_by"]) if row["created_by"] is not None else None,
            is_promoted=bool(int(row["is_promoted"] or 0)),
            promoted_at=str(row["promoted_at"]) if row["promoted_at"] is not None else None,
            promoted_by=str(row["promoted_by"]) if row["promoted_by"] is not None else None,
            created_at=str(row["created_at"]) if row["created_at"] is not None else None,
            updated_at=str(row["updated_at"]) if row["updated_at"] is not None else None,
        )

    @staticmethod
    def _job_matches_worker_capabilities(
        job: PrelabelJobRecord,
        *,
        model_versions: list[str] | None = None,
        model_ids: list[str] | None = None,
        model_backbones: list[str] | None = None,
    ) -> bool:
        normalized_versions = {
            str(item).strip()
            for item in (model_versions or [])
            if str(item).strip() and str(item).strip().lower() not in _CAPABILITY_WILDCARDS
        }
        normalized_ids = {
            str(item).strip()
            for item in (model_ids or [])
            if str(item).strip() and str(item).strip().lower() not in _CAPABILITY_WILDCARDS
        }
        normalized_backbones = {
            str(item).strip()
            for item in (model_backbones or [])
            if str(item).strip() and str(item).strip().lower() not in _CAPABILITY_WILDCARDS
        }

        if normalized_ids and (job.model_id or "").strip():
            if str(job.model_id).strip() not in normalized_ids:
                return False
        elif normalized_versions and job.model_version not in normalized_versions:
            return False

        if normalized_backbones and (job.model_backbone or "").strip():
            if str(job.model_backbone).strip() not in normalized_backbones:
                return False
        return True

    def upsert_worker_node(
        self,
        *,
        worker_id: str,
        display_name: str | None = None,
        host_name: str | None = None,
        device_label: str | None = None,
        status: str = "online",
        capabilities: dict[str, Any] | None = None,
        last_claimed_at: str | None = None,
    ) -> WorkerNodeRecord:
        now = _utc_now_iso()
        with self._connect() as connection:
            self._ensure_schema(connection)
            connection.execute(
                """
                INSERT INTO worker_nodes (
                    worker_id,
                    display_name,
                    host_name,
                    device_label,
                    status,
                    capabilities_json,
                    last_seen_at,
                    last_claimed_at,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(worker_id) DO UPDATE SET
                    display_name = COALESCE(excluded.display_name, worker_nodes.display_name),
                    host_name = COALESCE(excluded.host_name, worker_nodes.host_name),
                    device_label = COALESCE(excluded.device_label, worker_nodes.device_label),
                    status = excluded.status,
                    capabilities_json = CASE
                        WHEN excluded.capabilities_json = '{}' THEN worker_nodes.capabilities_json
                        ELSE excluded.capabilities_json
                    END,
                    last_seen_at = excluded.last_seen_at,
                    last_claimed_at = COALESCE(excluded.last_claimed_at, worker_nodes.last_claimed_at),
                    updated_at = excluded.updated_at
                """,
                (
                    worker_id,
                    display_name,
                    host_name,
                    device_label,
                    status,
                    json.dumps(capabilities or {}, ensure_ascii=False),
                    now,
                    last_claimed_at,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM worker_nodes WHERE worker_id = ?",
                (worker_id,),
            ).fetchone()
            connection.commit()
        return self._row_to_worker_node(row) if row is not None else WorkerNodeRecord(worker_id=worker_id)

    def list_worker_nodes(self, *, limit: int = 100) -> list[WorkerNodeRecord]:
        normalized_limit = max(1, min(int(limit), 500))
        with self._connect() as connection:
            self._ensure_schema(connection)
            rows = connection.execute(
                """
                SELECT *
                FROM worker_nodes
                ORDER BY last_seen_at DESC, updated_at DESC, rowid DESC
                LIMIT ?
                """,
                (normalized_limit,),
            ).fetchall()
        return [self._row_to_worker_node(row) for row in rows]

    def enqueue_prelabel_job(
        self,
        *,
        task_id: str,
        requested_by: str,
        model_version: str,
        model_id: str | None = None,
        model_backbone: str | None = None,
        candidate_limit: int | None = None,
        confidence_threshold: float | None = None,
        input_fingerprint: str,
        priority: int = 100,
        cancel_existing: bool = False,
    ) -> tuple[PrelabelJobRecord, bool]:
        now = _utc_now_iso()
        with self._connect() as connection:
            self._ensure_schema(connection)
            if cancel_existing:
                connection.execute(
                    """
                    UPDATE prelabel_jobs
                    SET status = 'cancelled',
                        error_message = 'superseded by newer enqueue request',
                        claim_expires_at = NULL,
                        updated_at = ?
                    WHERE task_id = ?
                      AND status IN ('queued', 'claimed')
                    """,
                    (now, task_id),
                )

            existing = connection.execute(
                """
                SELECT * FROM prelabel_jobs
                WHERE task_id = ?
                  AND status IN ('queued', 'claimed')
                ORDER BY created_at DESC, rowid DESC
                LIMIT 1
                """,
                (task_id,),
            ).fetchone()
            if existing is not None:
                connection.commit()
                return self._row_to_prelabel_job(existing), False

            job_id = uuid.uuid4().hex
            connection.execute(
                """
                INSERT INTO prelabel_jobs (
                    job_id,
                    task_id,
                    requested_by,
                    model_version,
                    model_id,
                    model_backbone,
                    candidate_limit,
                    confidence_threshold,
                    input_fingerprint,
                    status,
                    priority,
                    attempt_count,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, 0, ?, ?)
                """,
                (
                    job_id,
                    task_id,
                    requested_by,
                    model_version,
                    model_id,
                    model_backbone,
                    int(candidate_limit) if candidate_limit is not None else None,
                    float(confidence_threshold) if confidence_threshold is not None else None,
                    input_fingerprint,
                    int(priority),
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM prelabel_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            connection.commit()
        if row is None:
            raise RuntimeError("failed to create prelabel job")
        return self._row_to_prelabel_job(row), True

    def _requeue_stale_prelabel_jobs(self, connection: sqlite3.Connection, *, now_iso: str) -> None:
        connection.execute(
            """
            UPDATE prelabel_jobs
            SET status = 'queued',
                claim_worker_id = NULL,
                claimed_at = NULL,
                claim_expires_at = NULL,
                updated_at = ?,
                error_message = CASE
                    WHEN error_message IS NULL OR error_message = '' THEN 'job claim timed out and was requeued'
                    ELSE error_message
                END
            WHERE status = 'claimed'
              AND claim_expires_at IS NOT NULL
              AND claim_expires_at <= ?
            """,
            (now_iso, now_iso),
        )

    def get_prelabel_job(self, job_id: str) -> PrelabelJobRecord | None:
        with self._connect() as connection:
            self._ensure_schema(connection)
            row = connection.execute(
                "SELECT * FROM prelabel_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return self._row_to_prelabel_job(row) if row is not None else None

    def list_prelabel_jobs(
        self,
        *,
        limit: int = 100,
        statuses: Iterable[str] | None = None,
        task_ids: Iterable[str] | None = None,
    ) -> list[PrelabelJobRecord]:
        normalized_limit = max(1, min(int(limit), 500))
        normalized_statuses = [str(item).strip() for item in (statuses or []) if str(item).strip()]
        normalized_task_ids = [str(item).strip() for item in dict.fromkeys(task_ids or []) if str(item).strip()]

        filters: list[str] = []
        params: list[Any] = []
        if normalized_statuses:
            placeholders = ",".join("?" for _ in normalized_statuses)
            filters.append(f"status IN ({placeholders})")
            params.extend(normalized_statuses)
        if normalized_task_ids:
            placeholders = ",".join("?" for _ in normalized_task_ids)
            filters.append(f"task_id IN ({placeholders})")
            params.extend(normalized_task_ids)

        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        with self._connect() as connection:
            self._ensure_schema(connection)
            rows = connection.execute(
                f"""
                SELECT *
                FROM prelabel_jobs
                {where_clause}
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?
                """,
                (*params, normalized_limit),
            ).fetchall()
        return [self._row_to_prelabel_job(row) for row in rows]

    def cancel_prelabel_jobs(
        self,
        *,
        job_ids: Iterable[str] | None = None,
        task_ids: Iterable[str] | None = None,
        statuses: Iterable[str] | None = None,
        cancelled_by: str | None = None,
        reason: str | None = None,
    ) -> list[PrelabelJobRecord]:
        normalized_job_ids = [str(item).strip() for item in dict.fromkeys(job_ids or []) if str(item).strip()]
        normalized_task_ids = [str(item).strip() for item in dict.fromkeys(task_ids or []) if str(item).strip()]
        normalized_statuses = [str(item).strip() for item in (statuses or []) if str(item).strip()]
        if not normalized_statuses:
            normalized_statuses = ["queued", "claimed"]
        if not normalized_job_ids and not normalized_task_ids:
            return []

        target_filters: list[str] = []
        target_params: list[Any] = []
        if normalized_job_ids:
            placeholders = ",".join("?" for _ in normalized_job_ids)
            target_filters.append(f"job_id IN ({placeholders})")
            target_params.extend(normalized_job_ids)
        if normalized_task_ids:
            placeholders = ",".join("?" for _ in normalized_task_ids)
            target_filters.append(f"task_id IN ({placeholders})")
            target_params.extend(normalized_task_ids)

        status_placeholders = ",".join("?" for _ in normalized_statuses)
        cancel_reason = str(reason or "").strip() or "cancelled by admin"
        if cancelled_by and str(cancelled_by).strip():
            cancel_reason = f"{cancel_reason} ({str(cancelled_by).strip()})"
        now = _utc_now_iso()

        with self._connect() as connection:
            self._ensure_schema(connection)
            rows = connection.execute(
                f"""
                SELECT rowid
                FROM prelabel_jobs
                WHERE ({' OR '.join(target_filters)})
                  AND status IN ({status_placeholders})
                ORDER BY created_at DESC, rowid DESC
                """,
                (*target_params, *normalized_statuses),
            ).fetchall()
            rowids = [int(row["rowid"]) for row in rows]
            if not rowids:
                connection.commit()
                return []

            rowid_placeholders = ",".join("?" for _ in rowids)
            connection.execute(
                f"""
                UPDATE prelabel_jobs
                SET status = 'cancelled',
                    claim_expires_at = NULL,
                    error_message = ?,
                    updated_at = ?
                WHERE rowid IN ({rowid_placeholders})
                """,
                (cancel_reason, now, *rowids),
            )
            updated_rows = connection.execute(
                f"""
                SELECT *
                FROM prelabel_jobs
                WHERE rowid IN ({rowid_placeholders})
                ORDER BY updated_at DESC, rowid DESC
                """,
                tuple(rowids),
            ).fetchall()
            connection.commit()
        return [self._row_to_prelabel_job(row) for row in updated_rows]

    def claim_next_prelabel_job(
        self,
        *,
        worker_id: str,
        timeout_seconds: int,
        model_versions: list[str] | None = None,
        model_ids: list[str] | None = None,
        model_backbones: list[str] | None = None,
    ) -> PrelabelJobRecord | None:
        now_dt = datetime.now(timezone.utc)
        now_iso = now_dt.isoformat(timespec="seconds")
        expires_at = self._expires_after(now_dt, timeout_seconds)
        with self._connect() as connection:
            self._ensure_schema(connection)
            self._requeue_stale_prelabel_jobs(connection, now_iso=now_iso)
            for _ in range(8):
                candidate_rows = connection.execute(
                    """
                    SELECT *
                    FROM prelabel_jobs
                    WHERE status = 'queued'
                    ORDER BY priority DESC, created_at ASC, rowid ASC
                    LIMIT 32
                    """
                ).fetchall()
                row = None
                for candidate_row in candidate_rows:
                    candidate = self._row_to_prelabel_job(candidate_row)
                    if self._job_matches_worker_capabilities(
                        candidate,
                        model_versions=model_versions,
                        model_ids=model_ids,
                        model_backbones=model_backbones,
                    ):
                        row = candidate_row
                        break
                if row is None:
                    connection.commit()
                    return None
                job_id = str(row["job_id"])
                updated = connection.execute(
                    """
                    UPDATE prelabel_jobs
                    SET status = 'claimed',
                        claim_worker_id = ?,
                        claimed_at = ?,
                        claim_expires_at = ?,
                        last_heartbeat_at = ?,
                        attempt_count = attempt_count + 1,
                        error_message = NULL,
                        updated_at = ?
                    WHERE job_id = ?
                      AND status = 'queued'
                    """,
                    (worker_id, now_iso, expires_at, now_iso, now_iso, job_id),
                )
                if int(updated.rowcount or 0) <= 0:
                    continue
                connection.execute(
                    """
                    UPDATE worker_nodes
                    SET status = 'online',
                        last_seen_at = ?,
                        last_claimed_at = ?,
                        updated_at = ?
                    WHERE worker_id = ?
                    """,
                    (now_iso, now_iso, now_iso, worker_id),
                )
                claimed = connection.execute(
                    "SELECT * FROM prelabel_jobs WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
                connection.commit()
                return self._row_to_prelabel_job(claimed) if claimed is not None else None
            connection.commit()
        return None

    def heartbeat_prelabel_job(self, *, job_id: str, worker_id: str, timeout_seconds: int) -> bool:
        now_dt = datetime.now(timezone.utc)
        now_iso = now_dt.isoformat(timespec="seconds")
        expires_at = self._expires_after(now_dt, timeout_seconds)
        with self._connect() as connection:
            self._ensure_schema(connection)
            updated = connection.execute(
                """
                UPDATE prelabel_jobs
                SET last_heartbeat_at = ?,
                    claim_expires_at = ?,
                    updated_at = ?
                WHERE job_id = ?
                  AND status = 'claimed'
                  AND claim_worker_id = ?
                """,
                (now_iso, expires_at, now_iso, job_id, worker_id),
            )
            connection.execute(
                """
                UPDATE worker_nodes
                SET status = 'online',
                    last_seen_at = ?,
                    updated_at = ?
                WHERE worker_id = ?
                """,
                (now_iso, now_iso, worker_id),
            )
            connection.commit()
        return int(updated.rowcount or 0) > 0

    def fail_prelabel_job(
        self,
        *,
        job_id: str,
        worker_id: str,
        error_message: str,
        retryable: bool = False,
    ) -> bool:
        now = _utc_now_iso()
        status = "queued" if retryable else "failed"
        claim_worker_id = None if retryable else worker_id
        claimed_at = None if retryable else now
        with self._connect() as connection:
            self._ensure_schema(connection)
            updated = connection.execute(
                """
                UPDATE prelabel_jobs
                SET status = ?,
                    claim_worker_id = ?,
                    claimed_at = ?,
                    claim_expires_at = NULL,
                    last_heartbeat_at = ?,
                    error_message = ?,
                    updated_at = ?
                WHERE job_id = ?
                  AND status = 'claimed'
                  AND claim_worker_id = ?
                """,
                (status, claim_worker_id, claimed_at, now, error_message, now, job_id, worker_id),
            )
            connection.execute(
                """
                UPDATE worker_nodes
                SET status = 'online',
                    last_seen_at = ?,
                    updated_at = ?
                WHERE worker_id = ?
                """,
                (now, now, worker_id),
            )
            connection.commit()
        return int(updated.rowcount or 0) > 0

    def complete_prelabel_job(
        self,
        *,
        job_id: str,
        worker_id: str,
        source_view: str | None,
        ai_suggestion: str | None,
        ai_confidence: float | None,
        annotations: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> TaskAIPrelabelRecord | None:
        now = _utc_now_iso()
        with self._connect() as connection:
            self._ensure_schema(connection)
            job_row = connection.execute(
                "SELECT * FROM prelabel_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if job_row is None:
                return None
            job = self._row_to_prelabel_job(job_row)
            if job.status != "claimed" or job.claim_worker_id != worker_id:
                return None

            connection.execute(
                """
                UPDATE task_ai_prelabels
                SET status = 'superseded',
                    superseded_at = ?,
                    updated_at = ?
                WHERE task_id = ?
                  AND status = 'available'
                """,
                (now, now, job.task_id),
            )

            prelabel_id = uuid.uuid4().hex
            connection.execute(
                """
                INSERT INTO task_ai_prelabels (
                    prelabel_id,
                    task_id,
                    job_id,
                    source_view,
                    ai_suggestion,
                    ai_confidence,
                    model_version,
                    model_id,
                    model_backbone,
                    candidate_limit,
                    confidence_threshold,
                    input_fingerprint,
                    status,
                    box_count,
                    worker_id,
                    accepted_revision_id,
                    metadata_json,
                    created_at,
                    updated_at,
                    superseded_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'available', ?, ?, NULL, ?, ?, ?, NULL)
                """,
                (
                    prelabel_id,
                    job.task_id,
                    job.job_id,
                    source_view,
                    ai_suggestion,
                    ai_confidence,
                    job.model_version,
                    job.model_id,
                    job.model_backbone,
                    job.candidate_limit,
                    job.confidence_threshold,
                    job.input_fingerprint,
                    len(annotations),
                    worker_id,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            for index, annotation in enumerate(annotations):
                connection.execute(
                    """
                    INSERT INTO task_ai_prelabel_boxes (
                        prelabel_id,
                        box_index,
                        x,
                        y,
                        width,
                        height,
                        label,
                        detail_type,
                        confidence
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        prelabel_id,
                        index,
                        float(annotation.get("x", 0.0)),
                        float(annotation.get("y", 0.0)),
                        float(annotation.get("width", 0.0)),
                        float(annotation.get("height", 0.0)),
                        annotation.get("label"),
                        annotation.get("detail_type"),
                        float(annotation.get("confidence", 1.0)),
                    ),
                )

            connection.execute(
                """
                UPDATE prelabel_jobs
                SET status = 'completed',
                    last_heartbeat_at = ?,
                    claim_expires_at = NULL,
                    result_prelabel_id = ?,
                    error_message = NULL,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (now, prelabel_id, now, job_id),
            )
            prelabel_row = connection.execute(
                "SELECT * FROM task_ai_prelabels WHERE prelabel_id = ?",
                (prelabel_id,),
            ).fetchone()
            connection.execute(
                """
                UPDATE worker_nodes
                SET status = 'online',
                    last_seen_at = ?,
                    updated_at = ?
                WHERE worker_id = ?
                """,
                (now, now, worker_id),
            )
            connection.commit()
        return self._row_to_ai_prelabel(prelabel_row) if prelabel_row is not None else None

    def _fetch_task_prelabel_row(
        self,
        connection: sqlite3.Connection,
        *,
        task_id: str,
        statuses: Iterable[str] | None = None,
    ) -> sqlite3.Row | None:
        filters = ["task_id = ?"]
        params: list[Any] = [task_id]
        normalized_statuses = [str(status).strip() for status in (statuses or []) if str(status).strip()]
        if normalized_statuses:
            placeholders = ",".join("?" for _ in normalized_statuses)
            filters.append(f"status IN ({placeholders})")
            params.extend(normalized_statuses)
        where_clause = " AND ".join(filters)
        return connection.execute(
            f"""
            SELECT *
            FROM task_ai_prelabels
            WHERE {where_clause}
            ORDER BY updated_at DESC, created_at DESC, rowid DESC
            LIMIT 1
            """,
            tuple(params),
        ).fetchone()

    def get_latest_task_prelabel_record(
        self,
        task_id: str,
        *,
        statuses: Iterable[str] | None = None,
    ) -> TaskAIPrelabelRecord | None:
        with self._connect() as connection:
            self._ensure_schema(connection)
            row = self._fetch_task_prelabel_row(
                connection,
                task_id=task_id,
                statuses=statuses,
            )
        return self._row_to_ai_prelabel(row) if row is not None else None

    def mark_prelabel_accepted(
        self,
        *,
        task_id: str,
        prelabel_id: str,
        accepted_revision_id: str,
        accepted_by: str,
        acceptance_metadata: dict[str, Any] | None = None,
    ) -> TaskAIPrelabelRecord | None:
        now = _utc_now_iso()
        with self._connect() as connection:
            self._ensure_schema(connection)
            row = connection.execute(
                """
                SELECT *
                FROM task_ai_prelabels
                WHERE task_id = ?
                  AND prelabel_id = ?
                LIMIT 1
                """,
                (task_id, prelabel_id),
            ).fetchone()
            if row is None:
                return None
            prelabel = self._row_to_ai_prelabel(row)
            if prelabel.status not in {"available", "accepted"}:
                return None

            metadata = dict(prelabel.metadata or {})
            metadata["acceptance"] = {
                "accepted_at": now,
                "accepted_by": accepted_by,
                "accepted_revision_id": accepted_revision_id,
                **(acceptance_metadata or {}),
            }
            connection.execute(
                """
                UPDATE task_ai_prelabels
                SET status = 'accepted',
                    accepted_revision_id = ?,
                    metadata_json = ?,
                    updated_at = ?
                WHERE task_id = ?
                  AND prelabel_id = ?
                """,
                (
                    accepted_revision_id,
                    json.dumps(metadata, ensure_ascii=False),
                    now,
                    task_id,
                    prelabel_id,
                ),
            )
            updated_row = connection.execute(
                """
                SELECT *
                FROM task_ai_prelabels
                WHERE task_id = ?
                  AND prelabel_id = ?
                LIMIT 1
                """,
                (task_id, prelabel_id),
            ).fetchone()
            connection.commit()
        return self._row_to_ai_prelabel(updated_row) if updated_row is not None else None

    def get_task_prelabel(self, task_id: str) -> tuple[TaskAIPrelabelRecord, list[dict[str, Any]]] | None:
        with self._connect() as connection:
            self._ensure_schema(connection)
            row = self._fetch_task_prelabel_row(
                connection,
                task_id=task_id,
                statuses=("available",),
            )
            if row is None:
                return None
            boxes = connection.execute(
                """
                SELECT box_index, x, y, width, height, label, detail_type, confidence
                FROM task_ai_prelabel_boxes
                WHERE prelabel_id = ?
                ORDER BY box_index ASC
                """,
                (str(row["prelabel_id"]),),
            ).fetchall()
        annotations = [
            {
                "x": float(box["x"]),
                "y": float(box["y"]),
                "width": float(box["width"]),
                "height": float(box["height"]),
                "label": box["label"],
                "detail_type": box["detail_type"],
                "confidence": float(box["confidence"]) if box["confidence"] is not None else 1.0,
            }
            for box in boxes
        ]
        return self._row_to_ai_prelabel(row), annotations

    def list_task_prelabel_summaries(self, task_ids: Iterable[str]) -> dict[str, TaskPrelabelSummaryRecord]:
        normalized_ids = [task_id for task_id in dict.fromkeys(task_ids) if task_id]
        if not normalized_ids:
            return {}
        placeholders = ",".join("?" for _ in normalized_ids)
        with self._connect() as connection:
            self._ensure_schema(connection)
            prelabel_rows = connection.execute(
                f"""
                SELECT p.*
                FROM task_ai_prelabels p
                INNER JOIN (
                    SELECT task_id, MAX(rowid) AS latest_rowid
                    FROM task_ai_prelabels
                    WHERE status IN ('available', 'accepted', 'hidden')
                      AND task_id IN ({placeholders})
                    GROUP BY task_id
                ) latest
                    ON latest.latest_rowid = p.rowid
                """,
                tuple(normalized_ids),
            ).fetchall()
            job_rows = connection.execute(
                f"""
                SELECT j.*
                FROM prelabel_jobs j
                INNER JOIN (
                    SELECT task_id, MAX(rowid) AS latest_rowid
                    FROM prelabel_jobs
                    WHERE task_id IN ({placeholders})
                    GROUP BY task_id
                ) latest
                    ON latest.latest_rowid = j.rowid
                """,
                tuple(normalized_ids),
            ).fetchall()

        summaries: dict[str, TaskPrelabelSummaryRecord] = {}
        for row in job_rows:
            job = self._row_to_prelabel_job(row)
            mapped_status = {
                "claimed": "processing",
                "queued": "queued",
                "failed": "failed",
                "cancelled": "cancelled",
                "completed": "completed",
            }.get(job.status, job.status)
            summaries[job.task_id] = TaskPrelabelSummaryRecord(
                task_id=job.task_id,
                prelabel_status=mapped_status,
                prelabel_model_version=job.model_version,
                prelabel_model_id=job.model_id,
                prelabel_model_backbone=job.model_backbone,
                prelabel_candidate_limit=job.candidate_limit,
                prelabel_confidence_threshold=job.confidence_threshold,
                prelabel_updated_at=job.updated_at,
                prelabel_box_count=0,
                prelabel_job_id=job.job_id,
            )

        for row in prelabel_rows:
            prelabel = self._row_to_ai_prelabel(row)
            summaries[prelabel.task_id] = TaskPrelabelSummaryRecord(
                task_id=prelabel.task_id,
                prelabel_status=prelabel.status,
                prelabel_model_version=prelabel.model_version,
                prelabel_model_id=prelabel.model_id,
                prelabel_model_backbone=prelabel.model_backbone,
                prelabel_candidate_limit=prelabel.candidate_limit,
                prelabel_confidence_threshold=prelabel.confidence_threshold,
                prelabel_updated_at=prelabel.updated_at,
                prelabel_box_count=prelabel.box_count,
                prelabel_id=prelabel.prelabel_id,
                prelabel_job_id=prelabel.job_id,
            )
        return summaries

    def create_dataset_snapshot(
        self,
        *,
        snapshot_id: str,
        snapshot_name: str,
        document_relpath: str,
        task_count: int,
        annotation_count: int,
        created_by: str,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetSnapshotRecord:
        now = _utc_now_iso()
        with self._connect() as connection:
            self._ensure_schema(connection)
            connection.execute(
                """
                INSERT INTO dataset_snapshots (
                    snapshot_id,
                    snapshot_name,
                    document_relpath,
                    task_count,
                    annotation_count,
                    created_by,
                    metadata_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    snapshot_name,
                    document_relpath,
                    int(task_count),
                    int(annotation_count),
                    created_by,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM dataset_snapshots WHERE snapshot_id = ?",
                (snapshot_id,),
            ).fetchone()
            connection.commit()
        if row is None:
            raise RuntimeError("failed to create dataset snapshot")
        return self._row_to_dataset_snapshot(row)

    def get_dataset_snapshot(self, snapshot_id: str) -> DatasetSnapshotRecord | None:
        with self._connect() as connection:
            self._ensure_schema(connection)
            row = connection.execute(
                "SELECT * FROM dataset_snapshots WHERE snapshot_id = ?",
                (snapshot_id,),
            ).fetchone()
        return self._row_to_dataset_snapshot(row) if row is not None else None

    def list_dataset_snapshots(self, *, limit: int = 100) -> list[DatasetSnapshotRecord]:
        normalized_limit = max(1, min(int(limit), 500))
        with self._connect() as connection:
            self._ensure_schema(connection)
            rows = connection.execute(
                """
                SELECT *
                FROM dataset_snapshots
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?
                """,
                (normalized_limit,),
            ).fetchall()
        return [self._row_to_dataset_snapshot(row) for row in rows]

    def enqueue_training_job(
        self,
        *,
        snapshot_id: str,
        requested_by: str,
        task_type: str,
        model_version: str,
        model_id: str,
        model_backbone: str,
        train_config: dict[str, Any] | None = None,
        priority: int = 100,
        promote_on_success: bool = False,
        enqueue_prelabels_on_success: bool = False,
        prelabel_task_ids: list[str] | None = None,
        force_prelabel: bool = False,
    ) -> TrainingJobRecord:
        now = _utc_now_iso()
        with self._connect() as connection:
            self._ensure_schema(connection)
            job_id = uuid.uuid4().hex
            connection.execute(
                """
                INSERT INTO training_jobs (
                    job_id,
                    snapshot_id,
                    requested_by,
                    task_type,
                    model_version,
                    model_id,
                    model_backbone,
                    status,
                    train_config_json,
                    priority,
                    promote_on_success,
                    enqueue_prelabels_on_success,
                    prelabel_task_ids_json,
                    force_prelabel,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    snapshot_id,
                    requested_by,
                    task_type,
                    model_version,
                    model_id,
                    model_backbone,
                    json.dumps(train_config or {}, ensure_ascii=False),
                    int(priority),
                    1 if promote_on_success else 0,
                    1 if enqueue_prelabels_on_success else 0,
                    json.dumps(prelabel_task_ids or [], ensure_ascii=False),
                    1 if force_prelabel else 0,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM training_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            connection.commit()
        if row is None:
            raise RuntimeError("failed to create training job")
        return self._row_to_training_job(row)

    def get_training_job(self, job_id: str) -> TrainingJobRecord | None:
        with self._connect() as connection:
            self._ensure_schema(connection)
            row = connection.execute(
                "SELECT * FROM training_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return self._row_to_training_job(row) if row is not None else None

    def list_training_jobs(self, *, limit: int = 100) -> list[TrainingJobRecord]:
        normalized_limit = max(1, min(int(limit), 500))
        with self._connect() as connection:
            self._ensure_schema(connection)
            rows = connection.execute(
                """
                SELECT *
                FROM training_jobs
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?
                """,
                (normalized_limit,),
            ).fetchall()
        return [self._row_to_training_job(row) for row in rows]

    def _requeue_stale_training_jobs(self, connection: sqlite3.Connection, *, now_iso: str) -> None:
        connection.execute(
            """
            UPDATE training_jobs
            SET status = 'queued',
                claim_worker_id = NULL,
                claimed_at = NULL,
                claim_expires_at = NULL,
                updated_at = ?,
                error_message = CASE
                    WHEN error_message IS NULL OR error_message = '' THEN 'training job claim timed out and was requeued'
                    ELSE error_message
                END
            WHERE status = 'claimed'
              AND claim_expires_at IS NOT NULL
              AND claim_expires_at <= ?
            """,
            (now_iso, now_iso),
        )

    def claim_next_training_job(
        self,
        *,
        worker_id: str,
        timeout_seconds: int,
        task_types: list[str] | None = None,
        model_backbones: list[str] | None = None,
    ) -> TrainingJobRecord | None:
        now_dt = datetime.now(timezone.utc)
        now_iso = now_dt.isoformat(timespec="seconds")
        expires_at = self._expires_after(now_dt, timeout_seconds)
        normalized_task_types = {str(item).strip() for item in (task_types or []) if str(item).strip()}
        normalized_backbones = {str(item).strip() for item in (model_backbones or []) if str(item).strip()}
        with self._connect() as connection:
            self._ensure_schema(connection)
            self._requeue_stale_training_jobs(connection, now_iso=now_iso)
            for _ in range(8):
                rows = connection.execute(
                    """
                    SELECT *
                    FROM training_jobs
                    WHERE status = 'queued'
                    ORDER BY priority DESC, created_at ASC, rowid ASC
                    LIMIT 32
                    """
                ).fetchall()
                row = None
                for candidate_row in rows:
                    candidate = self._row_to_training_job(candidate_row)
                    if normalized_task_types and candidate.task_type not in normalized_task_types:
                        continue
                    if normalized_backbones and candidate.model_backbone not in normalized_backbones:
                        continue
                    row = candidate_row
                    break
                if row is None:
                    connection.commit()
                    return None
                job_id = str(row["job_id"])
                updated = connection.execute(
                    """
                    UPDATE training_jobs
                    SET status = 'claimed',
                        claim_worker_id = ?,
                        claimed_at = ?,
                        claim_expires_at = ?,
                        last_heartbeat_at = ?,
                        attempt_count = attempt_count + 1,
                        error_message = NULL,
                        updated_at = ?
                    WHERE job_id = ?
                      AND status = 'queued'
                    """,
                    (worker_id, now_iso, expires_at, now_iso, now_iso, job_id),
                )
                if int(updated.rowcount or 0) <= 0:
                    continue
                connection.execute(
                    """
                    UPDATE worker_nodes
                    SET status = 'online',
                        last_seen_at = ?,
                        updated_at = ?
                    WHERE worker_id = ?
                    """,
                    (now_iso, now_iso, worker_id),
                )
                claimed = connection.execute(
                    "SELECT * FROM training_jobs WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
                connection.commit()
                return self._row_to_training_job(claimed) if claimed is not None else None
            connection.commit()
        return None

    def heartbeat_training_job(self, *, job_id: str, worker_id: str, timeout_seconds: int) -> bool:
        now_dt = datetime.now(timezone.utc)
        now_iso = now_dt.isoformat(timespec="seconds")
        expires_at = self._expires_after(now_dt, timeout_seconds)
        with self._connect() as connection:
            self._ensure_schema(connection)
            updated = connection.execute(
                """
                UPDATE training_jobs
                SET last_heartbeat_at = ?,
                    claim_expires_at = ?,
                    updated_at = ?
                WHERE job_id = ?
                  AND status = 'claimed'
                  AND claim_worker_id = ?
                """,
                (now_iso, expires_at, now_iso, job_id, worker_id),
            )
            connection.commit()
        return int(updated.rowcount or 0) > 0

    def fail_training_job(self, *, job_id: str, worker_id: str, error_message: str, retryable: bool = False) -> bool:
        now = _utc_now_iso()
        status = "queued" if retryable else "failed"
        claim_worker_id = None if retryable else worker_id
        claimed_at = None if retryable else now
        with self._connect() as connection:
            self._ensure_schema(connection)
            updated = connection.execute(
                """
                UPDATE training_jobs
                SET status = ?,
                    claim_worker_id = ?,
                    claimed_at = ?,
                    claim_expires_at = NULL,
                    last_heartbeat_at = ?,
                    error_message = ?,
                    updated_at = ?
                WHERE job_id = ?
                  AND status = 'claimed'
                  AND claim_worker_id = ?
                """,
                (status, claim_worker_id, claimed_at, now, error_message, now, job_id, worker_id),
            )
            connection.commit()
        return int(updated.rowcount or 0) > 0

    def complete_training_job(
        self,
        *,
        job_id: str,
        worker_id: str,
        artifact_path: str,
        metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[TrainingJobRecord, TrainingRunRecord, RegisteredModelRecord] | None:
        now = _utc_now_iso()
        with self._connect() as connection:
            self._ensure_schema(connection)
            job_row = connection.execute(
                "SELECT * FROM training_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if job_row is None:
                return None
            job = self._row_to_training_job(job_row)
            if job.status != "claimed" or job.claim_worker_id != worker_id:
                return None

            run_id = uuid.uuid4().hex
            connection.execute(
                """
                INSERT INTO training_runs (
                    run_id,
                    job_id,
                    snapshot_id,
                    task_type,
                    status,
                    worker_id,
                    model_id,
                    model_version,
                    model_backbone,
                    artifact_path,
                    metrics_json,
                    metadata_json,
                    started_at,
                    finished_at,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, 'completed', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    job.job_id,
                    job.snapshot_id,
                    job.task_type,
                    worker_id,
                    job.model_id,
                    job.model_version,
                    job.model_backbone,
                    artifact_path,
                    json.dumps(metrics or {}, ensure_ascii=False),
                    json.dumps(metadata or {}, ensure_ascii=False),
                    job.claimed_at or now,
                    now,
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO model_registry (
                    model_id,
                    model_version,
                    model_backbone,
                    task_type,
                    training_run_id,
                    snapshot_id,
                    artifact_path,
                    metrics_json,
                    metadata_json,
                    created_by,
                    is_promoted,
                    promoted_at,
                    promoted_by,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, NULL, ?, ?)
                ON CONFLICT(model_id) DO UPDATE SET
                    model_version = excluded.model_version,
                    model_backbone = excluded.model_backbone,
                    task_type = excluded.task_type,
                    training_run_id = excluded.training_run_id,
                    snapshot_id = excluded.snapshot_id,
                    artifact_path = excluded.artifact_path,
                    metrics_json = excluded.metrics_json,
                    metadata_json = excluded.metadata_json,
                    created_by = excluded.created_by,
                    updated_at = excluded.updated_at
                """,
                (
                    job.model_id,
                    job.model_version,
                    job.model_backbone,
                    job.task_type,
                    run_id,
                    job.snapshot_id,
                    artifact_path,
                    json.dumps(metrics or {}, ensure_ascii=False),
                    json.dumps(metadata or {}, ensure_ascii=False),
                    job.requested_by,
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                UPDATE training_jobs
                SET status = 'completed',
                    claim_expires_at = NULL,
                    last_heartbeat_at = ?,
                    error_message = NULL,
                    run_id = ?,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (now, run_id, now, job_id),
            )
            job_row = connection.execute(
                "SELECT * FROM training_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            run_row = connection.execute(
                "SELECT * FROM training_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            model_row = connection.execute(
                "SELECT * FROM model_registry WHERE model_id = ?",
                (job.model_id,),
            ).fetchone()
            connection.commit()
        if job_row is None or run_row is None or model_row is None:
            return None
        return (
            self._row_to_training_job(job_row),
            self._row_to_training_run(run_row),
            self._row_to_registered_model(model_row),
        )

    def list_training_runs(self, *, limit: int = 100) -> list[TrainingRunRecord]:
        normalized_limit = max(1, min(int(limit), 500))
        with self._connect() as connection:
            self._ensure_schema(connection)
            rows = connection.execute(
                """
                SELECT *
                FROM training_runs
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?
                """,
                (normalized_limit,),
            ).fetchall()
        return [self._row_to_training_run(row) for row in rows]

    def list_registered_models(self, *, task_type: str | None = None, limit: int = 100) -> list[RegisteredModelRecord]:
        normalized_limit = max(1, min(int(limit), 500))
        with self._connect() as connection:
            self._ensure_schema(connection)
            if task_type:
                rows = connection.execute(
                    """
                    SELECT *
                    FROM model_registry
                    WHERE task_type = ?
                    ORDER BY is_promoted DESC, created_at DESC, rowid DESC
                    LIMIT ?
                    """,
                    (task_type, normalized_limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT *
                    FROM model_registry
                    ORDER BY is_promoted DESC, created_at DESC, rowid DESC
                    LIMIT ?
                    """,
                    (normalized_limit,),
                ).fetchall()
        return [self._row_to_registered_model(row) for row in rows]

    def get_registered_model(self, model_id: str) -> RegisteredModelRecord | None:
        with self._connect() as connection:
            self._ensure_schema(connection)
            row = connection.execute(
                "SELECT * FROM model_registry WHERE model_id = ?",
                (model_id,),
            ).fetchone()
        return self._row_to_registered_model(row) if row is not None else None

    def get_promoted_model(self, *, task_type: str) -> RegisteredModelRecord | None:
        with self._connect() as connection:
            self._ensure_schema(connection)
            row = connection.execute(
                """
                SELECT *
                FROM model_registry
                WHERE task_type = ?
                  AND is_promoted = 1
                ORDER BY promoted_at DESC, updated_at DESC, rowid DESC
                LIMIT 1
                """,
                (task_type,),
            ).fetchone()
        return self._row_to_registered_model(row) if row is not None else None

    def promote_registered_model(self, *, model_id: str, promoted_by: str) -> RegisteredModelRecord | None:
        now = _utc_now_iso()
        with self._connect() as connection:
            self._ensure_schema(connection)
            row = connection.execute(
                "SELECT * FROM model_registry WHERE model_id = ?",
                (model_id,),
            ).fetchone()
            if row is None:
                return None
            model = self._row_to_registered_model(row)
            connection.execute(
                """
                UPDATE model_registry
                SET is_promoted = 0,
                    updated_at = ?
                WHERE task_type = ?
                """,
                (now, model.task_type),
            )
            connection.execute(
                """
                UPDATE model_registry
                SET is_promoted = 1,
                    promoted_at = ?,
                    promoted_by = ?,
                    updated_at = ?
                WHERE model_id = ?
                """,
                (now, promoted_by, now, model_id),
            )
            updated_row = connection.execute(
                "SELECT * FROM model_registry WHERE model_id = ?",
                (model_id,),
            ).fetchone()
            connection.commit()
        return self._row_to_registered_model(updated_row) if updated_row is not None else None

    def try_claim_task(self, task_id: str, client_id: str, expires_at: str, now_iso: str) -> bool:
        with self._connect() as connection:
            self._ensure_schema(connection)
            result = connection.execute(
                """
                UPDATE tasks
                SET claim_client_id = ?,
                    claim_locked_at = ?,
                    claim_expires_at = ?,
                    preprocess_status = CASE
                        WHEN preprocess_status IN ('ready', 'viewed', 'annotated', 'claimed', 'align_failed') THEN 'claimed'
                        ELSE preprocess_status
                    END,
                    updated_at = ?
                WHERE task_id = ?
                  AND preprocess_status IN ('ready', 'viewed', 'annotated', 'claimed', 'align_failed')
                  AND (
                      claim_client_id IS NULL
                      OR claim_client_id = ?
                      OR claim_expires_at IS NULL
                      OR claim_expires_at <= ?
                  )
                """,
                (client_id, now_iso, expires_at, now_iso, task_id, client_id, now_iso),
            )
            if int(result.rowcount or 0) > 0:
                self._clear_other_claims_for_client(
                    connection,
                    client_id=client_id,
                    keep_task_id=task_id,
                    now_iso=now_iso,
                )
            connection.commit()
        return int(result.rowcount or 0) > 0

    def claim_task(self, task_id: str, client_id: str, expires_at: str) -> bool:
        now = _utc_now_iso()
        with self._connect() as connection:
            self._ensure_schema(connection)
            connection.execute(
                """
                UPDATE tasks
                SET claim_client_id = ?,
                    claim_locked_at = ?,
                    claim_expires_at = ?,
                    preprocess_status = CASE
                        WHEN preprocess_status IN ('ready', 'viewed', 'annotated') THEN 'claimed'
                        ELSE preprocess_status
                    END,
                    updated_at = ?
                WHERE task_id = ?
                """,
                (client_id, now, expires_at, now, task_id),
            )
            self._clear_other_claims_for_client(
                connection,
                client_id=client_id,
                keep_task_id=task_id,
                now_iso=now,
            )
            connection.commit()
        return True

    def refresh_claim(self, task_id: str, client_id: str, expires_at: str) -> bool:
        now = _utc_now_iso()
        with self._connect() as connection:
            self._ensure_schema(connection)
            row = connection.execute(
                "SELECT claim_client_id FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if row is None or str(row["claim_client_id"] or "") != client_id:
                return False
            connection.execute(
                """
                UPDATE tasks
                SET claim_expires_at = ?,
                    updated_at = ?
                WHERE task_id = ?
                """,
                (expires_at, now, task_id),
            )
            self._clear_other_claims_for_client(
                connection,
                client_id=client_id,
                keep_task_id=task_id,
                now_iso=now,
            )
            connection.commit()
        return True

    def release_claim(self, task_id: str, client_id: str | None = None) -> bool:
        now = _utc_now_iso()
        with self._connect() as connection:
            self._ensure_schema(connection)
            row = connection.execute(
                """
                SELECT claim_client_id, local_annotation_status, online_annotation_status
                FROM tasks
                WHERE task_id = ?
                """,
                (task_id,),
            ).fetchone()
            if row is None:
                return False
            stored_client = str(row["claim_client_id"] or "")
            if client_id is not None and stored_client and stored_client != client_id:
                return False
            local_status = str(row["local_annotation_status"] or "").strip().lower()
            online_status = str(row["online_annotation_status"] or "").strip().lower()
            next_status = "annotated" if "annotated" in {local_status, online_status} else "ready"
            connection.execute(
                """
                UPDATE tasks
                SET claim_client_id = NULL,
                    claim_locked_at = NULL,
                    claim_expires_at = NULL,
                    preprocess_status = CASE
                        WHEN preprocess_status = 'claimed' THEN ?
                        ELSE preprocess_status
                    END,
                    updated_at = ?
                WHERE task_id = ?
                """,
                (next_status, now, task_id),
            )
            connection.commit()
        return True

    @staticmethod
    def _row_to_raw_asset(row: sqlite3.Row) -> RawAssetRecord:
        metadata_raw = row["metadata_json"]
        metadata: dict[str, Any] | None = None
        if metadata_raw:
            try:
                parsed = json.loads(str(metadata_raw))
                if isinstance(parsed, dict):
                    metadata = parsed
            except Exception:
                metadata = None
        return RawAssetRecord(
            asset_id=str(row["asset_id"]),
            asset_role=str(row["asset_role"]),
            field_key=str(row["field_key"]),
            field_name=str(row["field_name"]),
            capture_key=str(row["capture_key"]),
            relpath=str(row["relpath"]),
            file_name=str(row["file_name"]),
            file_stem=str(row["file_stem"]),
            suffix=str(row["suffix"]),
            date_obs=str(row["date_obs"]) if row["date_obs"] is not None else None,
            size_bytes=int(row["size_bytes"] or 0),
            modified_time=float(row["modified_time"] or 0.0),
            metadata=metadata,
        )

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> TaskRecord:
        return TaskRecord(
            task_id=str(row["task_id"]),
            field_key=str(row["field_key"]),
            field_name=str(row["field_name"]),
            capture_key=str(row["capture_key"]),
            new_asset_id=str(row["new_asset_id"]),
            old_asset_id=str(row["old_asset_id"]) if row["old_asset_id"] is not None else None,
            new_marked_asset_id=(
                str(row["new_marked_asset_id"]) if row["new_marked_asset_id"] is not None else None
            ),
            preprocess_status=str(row["preprocess_status"]),
            current_source_view=(
                str(row["current_source_view"]) if row["current_source_view"] is not None else None
            ),
            current_label=str(row["current_label"]) if row["current_label"] is not None else None,
            current_detail_type=(
                str(row["current_detail_type"]) if row["current_detail_type"] is not None else None
            ),
            current_annotation_count=int(row["current_annotation_count"] or 0),
            current_ai_suggestion=(
                str(row["current_ai_suggestion"]) if row["current_ai_suggestion"] is not None else None
            ),
            current_ai_confidence=(
                float(row["current_ai_confidence"]) if row["current_ai_confidence"] is not None else None
            ),
            annotation_updated_at=(
                str(row["annotation_updated_at"]) if row["annotation_updated_at"] is not None else None
            ),
            local_viewed_at=str(row["local_viewed_at"]) if row["local_viewed_at"] is not None else None,
            local_annotation_status=(
                str(row["local_annotation_status"]) if row["local_annotation_status"] is not None else None
            ),
            online_annotation_status=(
                str(row["online_annotation_status"]) if row["online_annotation_status"] is not None else None
            ),
            claim_client_id=str(row["claim_client_id"]) if row["claim_client_id"] is not None else None,
            claim_locked_at=str(row["claim_locked_at"]) if row["claim_locked_at"] is not None else None,
            claim_expires_at=str(row["claim_expires_at"]) if row["claim_expires_at"] is not None else None,
            crop_x0=int(row["crop_x0"]) if row["crop_x0"] is not None else None,
            crop_x1=int(row["crop_x1"]) if row["crop_x1"] is not None else None,
            crop_y0=int(row["crop_y0"]) if row["crop_y0"] is not None else None,
            crop_y1=int(row["crop_y1"]) if row["crop_y1"] is not None else None,
            align_dx=float(row["align_dx"]) if row["align_dx"] is not None else None,
            align_dy=float(row["align_dy"]) if row["align_dy"] is not None else None,
        )
