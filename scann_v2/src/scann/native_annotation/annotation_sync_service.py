from __future__ import annotations

import json
import os
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel

from scann.core.dataset_storage import DatasetStorage


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _validate_identifier(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not _IDENTIFIER_RE.fullmatch(normalized):
        raise ValueError(f"invalid PostgreSQL {field_name}: {value!r}")
    return normalized


def _default_dataset_id(dataset_root: Path) -> str:
    name = dataset_root.name.strip() or "dataset"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_.-") or "dataset"


@dataclass(frozen=True)
class AnnotationSyncConfig:
    enabled: bool
    database_url: str
    dataset_id: str
    schema_name: str = "public"
    interval_seconds: int = 0
    connect_timeout_seconds: int = 10

    @property
    def configured(self) -> bool:
        return bool(self.database_url.strip())


class AnnotationSyncResult(BaseModel):
    success: bool
    dataset_id: str
    started_at: str
    finished_at: str
    sync_mode: str = "incremental"
    previous_revision_rowid: int = 0
    last_revision_rowid: int = 0
    tasks_synced: int = 0
    revisions_synced: int = 0
    current_boxes_synced: int = 0
    revision_boxes_synced: int = 0
    error_message: str = ""


class AnnotationSyncStatus(BaseModel):
    enabled: bool
    configured: bool
    scheduled: bool
    running: bool
    interval_seconds: int
    dataset_id: str
    schema_name: str
    scope: str = "annotations_only"
    last_result: Optional[AnnotationSyncResult] = None


ConnectFactory = Callable[[str], Any]


class AnnotationSyncService:
    def __init__(
        self,
        storage: DatasetStorage,
        config: AnnotationSyncConfig,
        *,
        connect_factory: ConnectFactory | None = None,
    ) -> None:
        self._storage = storage
        self.config = config
        self._connect_factory = connect_factory
        self._schema_name = _validate_identifier(config.schema_name, field_name="schema name")

    @property
    def dataset_id(self) -> str:
        return self.config.dataset_id

    def status(self, *, scheduler: "AnnotationSyncScheduler | None" = None) -> AnnotationSyncStatus:
        return AnnotationSyncStatus(
            enabled=self.config.enabled,
            configured=self.config.configured,
            scheduled=bool(scheduler and scheduler.is_started),
            running=bool(scheduler and scheduler.is_running),
            interval_seconds=max(0, self.config.interval_seconds),
            dataset_id=self.config.dataset_id,
            schema_name=self._schema_name,
            last_result=scheduler.last_result if scheduler else None,
        )

    def _table(self, name: str) -> str:
        table_name = _validate_identifier(name, field_name="table name")
        return f"{self._schema_name}.{table_name}"

    def _connect(self) -> Any:
        if not self.config.configured:
            raise ValueError("PostgreSQL sync is not configured")
        if self._connect_factory is not None:
            return self._connect_factory(self.config.database_url)

        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError(
                "PostgreSQL sync requires the optional dependency 'psycopg[binary]'."
            ) from exc
        return psycopg.connect(
            self.config.database_url,
            connect_timeout=max(1, self.config.connect_timeout_seconds),
        )

    def _ensure_remote_schema(self, cursor: Any) -> None:
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {self._schema_name}")
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table("annotation_datasets")} (
                dataset_id TEXT PRIMARY KEY,
                updated_at TIMESTAMPTZ NOT NULL
            )
            """
        )
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table("annotation_tasks")} (
                dataset_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                source_view TEXT,
                label TEXT,
                detail_type TEXT,
                annotation_count INTEGER NOT NULL DEFAULT 0,
                ai_suggestion TEXT,
                ai_confidence DOUBLE PRECISION,
                metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                synced_at TIMESTAMPTZ NOT NULL,
                PRIMARY KEY(dataset_id, task_id)
            )
            """
        )
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table("annotation_current_boxes")} (
                dataset_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                box_index INTEGER NOT NULL,
                x DOUBLE PRECISION NOT NULL,
                y DOUBLE PRECISION NOT NULL,
                width DOUBLE PRECISION NOT NULL,
                height DOUBLE PRECISION NOT NULL,
                label TEXT,
                detail_type TEXT,
                confidence DOUBLE PRECISION,
                synced_at TIMESTAMPTZ NOT NULL,
                PRIMARY KEY(dataset_id, task_id, box_index)
            )
            """
        )
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table("annotation_revisions")} (
                dataset_id TEXT NOT NULL,
                revision_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                source_view TEXT,
                parent_revision_id TEXT,
                rollback_of_revision_id TEXT,
                submitted_by TEXT NOT NULL,
                origin TEXT NOT NULL,
                saved_at TEXT NOT NULL,
                metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                annotation_count INTEGER NOT NULL DEFAULT 0,
                synced_at TIMESTAMPTZ NOT NULL,
                PRIMARY KEY(dataset_id, revision_id)
            )
            """
        )
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table("annotation_revision_boxes")} (
                dataset_id TEXT NOT NULL,
                revision_id TEXT NOT NULL,
                box_index INTEGER NOT NULL,
                x DOUBLE PRECISION NOT NULL,
                y DOUBLE PRECISION NOT NULL,
                width DOUBLE PRECISION NOT NULL,
                height DOUBLE PRECISION NOT NULL,
                label TEXT,
                detail_type TEXT,
                confidence DOUBLE PRECISION,
                synced_at TIMESTAMPTZ NOT NULL,
                PRIMARY KEY(dataset_id, revision_id, box_index)
            )
            """
        )
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table("annotation_sync_state")} (
                dataset_id TEXT PRIMARY KEY,
                last_revision_rowid INTEGER NOT NULL DEFAULT 0,
                updated_at TIMESTAMPTZ NOT NULL
            )
            """
        )
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table("annotation_sync_runs")} (
                run_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                started_at TIMESTAMPTZ NOT NULL,
                finished_at TIMESTAMPTZ NOT NULL,
                success BOOLEAN NOT NULL,
                tasks_synced INTEGER NOT NULL DEFAULT 0,
                revisions_synced INTEGER NOT NULL DEFAULT 0,
                current_boxes_synced INTEGER NOT NULL DEFAULT 0,
                revision_boxes_synced INTEGER NOT NULL DEFAULT 0,
                error_message TEXT NOT NULL DEFAULT ''
            )
            """
        )

    def _fetch_last_synced_revision_rowid(self, cursor: Any) -> int:
        cursor.execute(
            f"""
            SELECT last_revision_rowid
            FROM {self._table("annotation_sync_state")}
            WHERE dataset_id = %s
            """,
            (self.config.dataset_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return 0
        try:
            return max(0, int(row[0]))
        except (TypeError, ValueError, IndexError):
            return 0

    def _upsert_sync_state(self, cursor: Any, *, last_revision_rowid: int, synced_at: str) -> None:
        cursor.execute(
            f"""
            INSERT INTO {self._table("annotation_sync_state")} (
                dataset_id, last_revision_rowid, updated_at
            )
            VALUES (%s, %s, %s)
            ON CONFLICT(dataset_id) DO UPDATE SET
                last_revision_rowid = EXCLUDED.last_revision_rowid,
                updated_at = EXCLUDED.updated_at
            """,
            (self.config.dataset_id, int(last_revision_rowid), synced_at),
        )

    def _upsert_dataset(self, cursor: Any, *, synced_at: str) -> None:
        cursor.execute(
            f"""
            INSERT INTO {self._table("annotation_datasets")} (
                dataset_id, updated_at
            )
            VALUES (%s, %s)
            ON CONFLICT(dataset_id) DO UPDATE SET
                updated_at = EXCLUDED.updated_at
            """,
            (
                self.config.dataset_id,
                synced_at,
            ),
        )

    def _sync_current_annotations(
        self,
        cursor: Any,
        *,
        current_annotations: dict[str, dict[str, Any]],
        synced_at: str,
    ) -> tuple[int, int]:
        box_count = 0
        for task_id, record in sorted(current_annotations.items()):
            annotations = record.get("annotations")
            if not isinstance(annotations, list):
                annotations = []
            metadata = record.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            cursor.execute(
                f"""
                INSERT INTO {self._table("annotation_tasks")} (
                    dataset_id,
                    task_id,
                    source_view,
                    label,
                    detail_type,
                    annotation_count,
                    ai_suggestion,
                    ai_confidence,
                    metadata_json,
                    synced_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                ON CONFLICT(dataset_id, task_id) DO UPDATE SET
                    source_view = EXCLUDED.source_view,
                    label = EXCLUDED.label,
                    detail_type = EXCLUDED.detail_type,
                    annotation_count = EXCLUDED.annotation_count,
                    ai_suggestion = EXCLUDED.ai_suggestion,
                    ai_confidence = EXCLUDED.ai_confidence,
                    metadata_json = EXCLUDED.metadata_json,
                    synced_at = EXCLUDED.synced_at
                """,
                (
                    self.config.dataset_id,
                    task_id,
                    record.get("source_view"),
                    record.get("label"),
                    record.get("detail_type"),
                    len(annotations),
                    record.get("ai_suggestion"),
                    record.get("ai_confidence"),
                    json.dumps(metadata, ensure_ascii=False),
                    synced_at,
                ),
            )
            cursor.execute(
                f"""
                DELETE FROM {self._table("annotation_current_boxes")}
                WHERE dataset_id = %s AND task_id = %s
                """,
                (self.config.dataset_id, task_id),
            )
            for box_index, annotation in enumerate(annotations):
                if not isinstance(annotation, dict):
                    continue
                cursor.execute(
                    f"""
                    INSERT INTO {self._table("annotation_current_boxes")} (
                        dataset_id,
                        task_id,
                        box_index,
                        x,
                        y,
                        width,
                        height,
                        label,
                        detail_type,
                        confidence,
                        synced_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        self.config.dataset_id,
                        task_id,
                        box_index,
                        float(annotation.get("x", 0.0)),
                        float(annotation.get("y", 0.0)),
                        float(annotation.get("width", 0.0)),
                        float(annotation.get("height", 0.0)),
                        annotation.get("label"),
                        annotation.get("detail_type"),
                        float(annotation.get("confidence", 1.0)),
                        synced_at,
                    ),
                )
                box_count += 1
        return len(current_annotations), box_count

    def _sync_revisions(
        self,
        cursor: Any,
        *,
        revisions: list[dict[str, Any]],
        synced_at: str,
    ) -> tuple[int, int]:
        box_count = 0
        for revision in revisions:
            annotations = revision.get("annotations")
            if not isinstance(annotations, list):
                annotations = []
            metadata = revision.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            revision_id = str(revision["revision_id"])
            cursor.execute(
                f"""
                INSERT INTO {self._table("annotation_revisions")} (
                    dataset_id,
                    revision_id,
                    task_id,
                    source_view,
                    parent_revision_id,
                    rollback_of_revision_id,
                    submitted_by,
                    origin,
                    saved_at,
                    metadata_json,
                    annotation_count,
                    synced_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                ON CONFLICT(dataset_id, revision_id) DO UPDATE SET
                    task_id = EXCLUDED.task_id,
                    source_view = EXCLUDED.source_view,
                    parent_revision_id = EXCLUDED.parent_revision_id,
                    rollback_of_revision_id = EXCLUDED.rollback_of_revision_id,
                    submitted_by = EXCLUDED.submitted_by,
                    origin = EXCLUDED.origin,
                    saved_at = EXCLUDED.saved_at,
                    metadata_json = EXCLUDED.metadata_json,
                    annotation_count = EXCLUDED.annotation_count,
                    synced_at = EXCLUDED.synced_at
                """,
                (
                    self.config.dataset_id,
                    revision_id,
                    str(revision["task_id"]),
                    revision.get("source_view"),
                    revision.get("parent_revision_id"),
                    revision.get("rollback_of_revision_id"),
                    str(revision["submitted_by"]),
                    str(revision.get("origin") or "unknown"),
                    str(revision["saved_at"]),
                    json.dumps(metadata, ensure_ascii=False),
                    len(annotations),
                    synced_at,
                ),
            )
            cursor.execute(
                f"""
                DELETE FROM {self._table("annotation_revision_boxes")}
                WHERE dataset_id = %s AND revision_id = %s
                """,
                (self.config.dataset_id, revision_id),
            )
            for box_index, annotation in enumerate(annotations):
                if not isinstance(annotation, dict):
                    continue
                cursor.execute(
                    f"""
                    INSERT INTO {self._table("annotation_revision_boxes")} (
                        dataset_id,
                        revision_id,
                        box_index,
                        x,
                        y,
                        width,
                        height,
                        label,
                        detail_type,
                        confidence,
                        synced_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        self.config.dataset_id,
                        revision_id,
                        box_index,
                        float(annotation.get("x", 0.0)),
                        float(annotation.get("y", 0.0)),
                        float(annotation.get("width", 0.0)),
                        float(annotation.get("height", 0.0)),
                        annotation.get("label"),
                        annotation.get("detail_type"),
                        float(annotation.get("confidence", 1.0)),
                        synced_at,
                    ),
                )
                box_count += 1
        return len(revisions), box_count

    def _insert_run_record(self, cursor: Any, result: AnnotationSyncResult) -> None:
        cursor.execute(
            f"""
            INSERT INTO {self._table("annotation_sync_runs")} (
                run_id,
                dataset_id,
                started_at,
                finished_at,
                success,
                tasks_synced,
                revisions_synced,
                current_boxes_synced,
                revision_boxes_synced,
                error_message
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                uuid.uuid4().hex,
                result.dataset_id,
                result.started_at,
                result.finished_at,
                result.success,
                result.tasks_synced,
                result.revisions_synced,
                result.current_boxes_synced,
                result.revision_boxes_synced,
                result.error_message,
            ),
        )

    def sync_now(self, *, full: bool = False) -> AnnotationSyncResult:
        started_at = _utc_now_iso()
        sync_mode = "full" if full else "incremental"
        connection = None
        revisions: list[dict[str, Any]] = []
        current_annotations: dict[str, dict[str, Any]] = {}
        previous_revision_rowid = 0
        last_revision_rowid = 0
        try:
            connection = self._connect()
            cursor = connection.cursor()
            try:
                self._ensure_remote_schema(cursor)
                previous_revision_rowid = 0 if full else self._fetch_last_synced_revision_rowid(cursor)
                revisions = self._storage.list_annotation_revisions_after_rowid(previous_revision_rowid)
                last_revision_rowid = max(
                    [previous_revision_rowid]
                    + [
                        int(item["storage_rowid"])
                        for item in revisions
                        if item.get("storage_rowid") is not None
                    ]
                )
                revision_task_ids = {
                    str(item["task_id"])
                    for item in revisions
                    if item.get("task_id") is not None
                }
                if revision_task_ids or full:
                    current_annotations = {
                        task_id: record
                        for task_id, record in self._storage.list_current_annotations().items()
                        if (
                            task_id in revision_task_ids
                            or (
                                full
                                and (
                                    bool(record.get("annotations"))
                                    or bool(record.get("label"))
                                    or bool(record.get("detail_type"))
                                )
                            )
                        )
                    }
                should_write_backup = full or last_revision_rowid > previous_revision_rowid
                tasks_synced = 0
                current_boxes_synced = 0
                revisions_synced = 0
                revision_boxes_synced = 0
                if should_write_backup:
                    self._upsert_dataset(cursor, synced_at=started_at)
                    tasks_synced, current_boxes_synced = self._sync_current_annotations(
                        cursor,
                        current_annotations=current_annotations,
                        synced_at=started_at,
                    )
                    revisions_synced, revision_boxes_synced = self._sync_revisions(
                        cursor,
                        revisions=revisions,
                        synced_at=started_at,
                    )
                result = AnnotationSyncResult(
                    success=True,
                    dataset_id=self.config.dataset_id,
                    started_at=started_at,
                    finished_at=_utc_now_iso(),
                    sync_mode=sync_mode,
                    previous_revision_rowid=previous_revision_rowid,
                    last_revision_rowid=last_revision_rowid,
                    tasks_synced=tasks_synced,
                    revisions_synced=revisions_synced,
                    current_boxes_synced=current_boxes_synced,
                    revision_boxes_synced=revision_boxes_synced,
                )
                if should_write_backup:
                    self._upsert_sync_state(
                        cursor,
                        last_revision_rowid=last_revision_rowid,
                        synced_at=result.finished_at,
                    )
                    self._insert_run_record(cursor, result)
                connection.commit()
                return result
            finally:
                close = getattr(cursor, "close", None)
                if close is not None:
                    close()
        except Exception as exc:
            if connection is not None:
                rollback = getattr(connection, "rollback", None)
                if rollback is not None:
                    rollback()
            return AnnotationSyncResult(
                success=False,
                dataset_id=self.config.dataset_id,
                started_at=started_at,
                finished_at=_utc_now_iso(),
                sync_mode=sync_mode,
                previous_revision_rowid=previous_revision_rowid,
                last_revision_rowid=last_revision_rowid,
                tasks_synced=len(current_annotations),
                revisions_synced=len(revisions),
                current_boxes_synced=sum(
                    len(item.get("annotations", []))
                    for item in current_annotations.values()
                    if isinstance(item.get("annotations", []), list)
                ),
                revision_boxes_synced=sum(
                    len(item.get("annotations", []))
                    for item in revisions
                    if isinstance(item.get("annotations", []), list)
                ),
                error_message=str(exc),
            )
        finally:
            if connection is not None:
                close = getattr(connection, "close", None)
                if close is not None:
                    close()


class AnnotationSyncScheduler:
    def __init__(self, service: AnnotationSyncService) -> None:
        self._service = service
        self._stop_event = threading.Event()
        self._run_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._last_result: AnnotationSyncResult | None = None

    @property
    def is_started(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_running(self) -> bool:
        return self._run_lock.locked()

    @property
    def last_result(self) -> AnnotationSyncResult | None:
        return self._last_result

    @property
    def dataset_id(self) -> str:
        return self._service.config.dataset_id

    def start(self) -> None:
        if self.is_started:
            return
        if not self._service.config.enabled or not self._service.config.configured:
            return
        if self._service.config.interval_seconds <= 0:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="scann-annotation-sync",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def run_once(self, *, full: bool = False) -> AnnotationSyncResult:
        if not self._run_lock.acquire(blocking=False):
            now = _utc_now_iso()
            return AnnotationSyncResult(
                success=False,
                dataset_id=self._service.config.dataset_id,
                started_at=now,
                finished_at=now,
                sync_mode="full" if full else "incremental",
                error_message="annotation sync is already running",
            )
        try:
            self._last_result = self._service.sync_now(full=full)
            return self._last_result
        finally:
            self._run_lock.release()

    def _run_loop(self) -> None:
        interval = max(1, self._service.config.interval_seconds)
        self.run_once()
        while not self._stop_event.wait(interval):
            self.run_once()


def build_annotation_sync_config_from_env(dataset_root: Path) -> AnnotationSyncConfig:
    database_url = (
        os.getenv("SCANN_ANNOTATION_SYNC_DATABASE_URL", "").strip()
        or os.getenv("SCANN_ANNOTATION_SYNC_PG_DSN", "").strip()
    )
    dataset_id = os.getenv("SCANN_ANNOTATION_SYNC_DATASET_ID", "").strip() or _default_dataset_id(dataset_root)
    schema_name = os.getenv("SCANN_ANNOTATION_SYNC_SCHEMA", "public").strip() or "public"
    return AnnotationSyncConfig(
        enabled=_env_bool("SCANN_ANNOTATION_SYNC_ENABLED", default=False),
        database_url=database_url,
        dataset_id=dataset_id,
        schema_name=schema_name,
        interval_seconds=max(0, _env_int("SCANN_ANNOTATION_SYNC_INTERVAL_SECONDS", 0)),
        connect_timeout_seconds=max(1, _env_int("SCANN_ANNOTATION_SYNC_CONNECT_TIMEOUT_SECONDS", 10)),
    )


def build_annotation_sync_service_from_env(dataset_root: Path) -> AnnotationSyncService:
    storage = DatasetStorage(dataset_root)
    storage.ensure_schema()
    config = build_annotation_sync_config_from_env(dataset_root)
    return AnnotationSyncService(storage=storage, config=config)


def build_annotation_sync_scheduler_from_env(dataset_root: Path) -> AnnotationSyncScheduler | None:
    config = build_annotation_sync_config_from_env(dataset_root)
    if not config.enabled or not config.configured or config.interval_seconds <= 0:
        return None
    storage = DatasetStorage(dataset_root)
    storage.ensure_schema()
    service = AnnotationSyncService(storage=storage, config=config)
    return AnnotationSyncScheduler(service)
