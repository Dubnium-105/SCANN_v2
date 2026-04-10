from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET_DB_FILE = "scann_dataset.db"

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


class DatasetStorage:
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
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA synchronous=NORMAL;")
        connection.execute("PRAGMA foreign_keys=ON;")
        return connection

    def ensure_schema(self) -> None:
        with self._connect() as connection:
            self._ensure_schema(connection)

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
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
        connection.commit()

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
                        WHEN preprocess_status = 'claimed' AND lower(coalesce(local_annotation_status, '')) = 'annotated'
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
            connection.commit()

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
            connection.commit()
        return True

    def release_claim(self, task_id: str, client_id: str | None = None) -> bool:
        now = _utc_now_iso()
        with self._connect() as connection:
            self._ensure_schema(connection)
            row = connection.execute(
                "SELECT claim_client_id, local_annotation_status FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if row is None:
                return False
            stored_client = str(row["claim_client_id"] or "")
            if client_id is not None and stored_client and stored_client != client_id:
                return False
            annotation_status = str(row["local_annotation_status"] or "").strip().lower()
            next_status = "annotated" if annotation_status == "annotated" else "ready"
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
