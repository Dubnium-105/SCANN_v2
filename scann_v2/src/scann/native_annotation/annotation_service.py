from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, cast

from pydantic import BaseModel, Field

from scann.services.dataset_preprocess_service import DatasetPreprocessService


class AnnotationBox(BaseModel):
    x: float = Field(..., ge=0)
    y: float = Field(..., ge=0)
    width: float = Field(..., ge=0)
    height: float = Field(..., ge=0)
    label: Optional[str] = None
    detail_type: Optional[str] = None


class AnnotationSaveRequest(BaseModel):
    annotations: List[AnnotationBox]
    source_view: Optional[Literal["old", "new", "new_marked"]] = "new"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnnotationSaveResponse(BaseModel):
    task_id: str
    format_version: str
    saved_path: str
    saved_count: int


class AnnotationRevision(BaseModel):
    revision_id: str
    task_id: str
    source_view: Optional[Literal["old", "new", "new_marked"]] = "new"
    parent_revision_id: Optional[str] = None
    rollback_of_revision_id: Optional[str] = None
    submitted_by: str
    saved_at: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    annotations: List[AnnotationBox]


class AnnotationHistoryItem(BaseModel):
    revision_id: str
    task_id: str
    parent_revision_id: Optional[str] = None
    rollback_of_revision_id: Optional[str] = None
    submitted_by: str
    saved_at: str
    format_version: str = "v2"
    annotation_count: int
    change_summary: "AnnotationChangeSummary" = Field(default_factory=lambda: AnnotationChangeSummary())


class AnnotationHistoryResponse(BaseModel):
    task_id: str
    revisions: List[AnnotationHistoryItem]


class AnnotationDiffItem(BaseModel):
    change_type: Literal["added", "removed", "modified"]
    before: Optional[AnnotationBox] = None
    after: Optional[AnnotationBox] = None
    changed_fields: List[str] = Field(default_factory=list)
    match_score: Optional[float] = None


class AnnotationChangeSummary(BaseModel):
    added: int = 0
    removed: int = 0
    modified: int = 0


class AnnotationRevisionDetail(AnnotationRevision):
    change_summary: AnnotationChangeSummary = Field(default_factory=AnnotationChangeSummary)
    changed_items: List[AnnotationDiffItem] = Field(default_factory=list)


class AnnotationRollbackResponse(BaseModel):
    task_id: str
    rolled_back_to_revision_id: str
    new_revision_id: str
    saved_count: int


class AnnotationService:
    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = dataset_root.resolve()
        self.revisions_dir = self.dataset_root / "annotation_revisions"
        configured_db_path = os.getenv("SCANN_NATIVE_DB_PATH", "").strip()
        self.db_path = Path(configured_db_path).resolve() if configured_db_path else self.dataset_root / "scann_native.db"
        self._ensure_db_schema()

    def _connect_db(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.db_path))
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_db_schema(self) -> None:
        with self._connect_db() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS annotation_revisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    revision_id TEXT UNIQUE NOT NULL,
                    task_id TEXT NOT NULL,
                    source_view TEXT,
                    parent_revision_id TEXT,
                    rollback_of_revision_id TEXT,
                    submitted_by TEXT NOT NULL,
                    saved_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    annotations_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_annotation_revisions_task_id_id
                ON annotation_revisions(task_id, id)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS annotation_task_state (
                    task_id TEXT PRIMARY KEY,
                    source_view TEXT,
                    updated_at TEXT NOT NULL,
                    image_entry_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS annotation_dataset_snapshot (
                    id INTEGER PRIMARY KEY CHECK(id = 1),
                    updated_at TEXT NOT NULL,
                    doc_json TEXT NOT NULL
                )
                """
            )
            connection.commit()

    @staticmethod
    def _validate_task_id(task_id: str) -> str:
        normalized = task_id.strip()
        if not normalized:
            raise ValueError("task_id cannot be empty")
        if Path(normalized).name != normalized:
            raise ValueError("invalid task_id")
        if ".." in normalized:
            raise ValueError("invalid task_id")
        return normalized

    @staticmethod
    def _normalize_source_view(source_view: Optional[str]) -> Optional[Literal["old", "new", "new_marked"]]:
        if source_view in {"old", "new", "new_marked"}:
            return cast(Literal["old", "new", "new_marked"], source_view)
        return None

    def _history_file_path(self, task_id: str) -> Path:
        return self.revisions_dir / f"{task_id}.jsonl"

    def _append_revision(self, revision: AnnotationRevision) -> None:
        with self._connect_db() as connection:
            connection.execute(
                """
                INSERT INTO annotation_revisions (
                    revision_id,
                    task_id,
                    source_view,
                    parent_revision_id,
                    rollback_of_revision_id,
                    submitted_by,
                    saved_at,
                    metadata_json,
                    annotations_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    revision.revision_id,
                    revision.task_id,
                    revision.source_view,
                    revision.parent_revision_id,
                    revision.rollback_of_revision_id,
                    revision.submitted_by,
                    revision.saved_at,
                    json.dumps(revision.metadata, ensure_ascii=False),
                    json.dumps([ann.model_dump(exclude_none=True) for ann in revision.annotations], ensure_ascii=False),
                ),
            )
            connection.commit()

        self.revisions_dir.mkdir(parents=True, exist_ok=True)
        history_file = self._history_file_path(revision.task_id)
        with history_file.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(revision.model_dump(), ensure_ascii=False))
            fp.write("\n")

    def _read_revisions(self, task_id: str) -> list[AnnotationRevision]:
        with self._connect_db() as connection:
            rows = connection.execute(
                """
                SELECT
                    revision_id,
                    task_id,
                    source_view,
                    parent_revision_id,
                    rollback_of_revision_id,
                    submitted_by,
                    saved_at,
                    metadata_json,
                    annotations_json
                FROM annotation_revisions
                WHERE task_id = ?
                ORDER BY id ASC
                """,
                (task_id,),
            ).fetchall()

        if rows:
            revisions: list[AnnotationRevision] = []
            for row in rows:
                try:
                    metadata = json.loads(str(row["metadata_json"]))
                except Exception:
                    metadata = {}
                try:
                    annotation_dicts = json.loads(str(row["annotations_json"]))
                except Exception:
                    annotation_dicts = []

                annotations = [AnnotationBox.model_validate(item) for item in annotation_dicts if isinstance(item, dict)]
                revisions.append(
                    AnnotationRevision(
                        revision_id=str(row["revision_id"]),
                        task_id=str(row["task_id"]),
                        source_view=self._normalize_source_view(
                            str(row["source_view"]) if row["source_view"] is not None else None
                        ),
                        parent_revision_id=str(row["parent_revision_id"]) if row["parent_revision_id"] is not None else None,
                        rollback_of_revision_id=str(row["rollback_of_revision_id"]) if row["rollback_of_revision_id"] is not None else None,
                        submitted_by=str(row["submitted_by"]),
                        saved_at=str(row["saved_at"]),
                        metadata=metadata if isinstance(metadata, dict) else {},
                        annotations=annotations,
                    )
                )
            return revisions

        history_file = self._history_file_path(task_id)
        if not history_file.exists():
            return []

        revisions: list[AnnotationRevision] = []
        for line in history_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            revisions.append(AnnotationRevision.model_validate_json(stripped))
        return revisions

    @staticmethod
    def _bbox_iou(left: AnnotationBox, right: AnnotationBox) -> float:
        left_x2 = left.x + left.width
        left_y2 = left.y + left.height
        right_x2 = right.x + right.width
        right_y2 = right.y + right.height

        inter_w = max(0.0, min(left_x2, right_x2) - max(left.x, right.x))
        inter_h = max(0.0, min(left_y2, right_y2) - max(left.y, right.y))
        inter = inter_w * inter_h
        if inter <= 0:
            return 0.0

        left_area = left.width * left.height
        right_area = right.width * right.height
        union = left_area + right_area - inter
        if union <= 0:
            return 0.0
        return inter / union

    @staticmethod
    def _center_distance(left: AnnotationBox, right: AnnotationBox) -> float:
        left_cx = left.x + (left.width / 2.0)
        left_cy = left.y + (left.height / 2.0)
        right_cx = right.x + (right.width / 2.0)
        right_cy = right.y + (right.height / 2.0)
        return sqrt((left_cx - right_cx) ** 2 + (left_cy - right_cy) ** 2)

    @staticmethod
    def _changed_fields(before: AnnotationBox, after: AnnotationBox) -> list[str]:
        changed: list[str] = []
        if abs(before.x - after.x) > 1e-6:
            changed.append("x")
        if abs(before.y - after.y) > 1e-6:
            changed.append("y")
        if abs(before.width - after.width) > 1e-6:
            changed.append("width")
        if abs(before.height - after.height) > 1e-6:
            changed.append("height")
        if before.label != after.label:
            changed.append("label")
        if before.detail_type != after.detail_type:
            changed.append("detail_type")
        return changed

    def _build_diff(
        self,
        before: list[AnnotationBox],
        after: list[AnnotationBox],
    ) -> tuple[AnnotationChangeSummary, list[AnnotationDiffItem]]:
        unmatched_before: set[int] = set(range(len(before)))
        unmatched_after: set[int] = set(range(len(after)))

        candidate_pairs: list[tuple[float, int, int]] = []
        for before_index, before_ann in enumerate(before):
            for after_index, after_ann in enumerate(after):
                iou = self._bbox_iou(before_ann, after_ann)
                distance = self._center_distance(before_ann, after_ann)
                similar_enough = iou > 0.12 or distance <= 12.0
                if not similar_enough:
                    continue
                score = iou + (0.08 if before_ann.label == after_ann.label else 0.0)
                score += 0.04 if before_ann.detail_type == after_ann.detail_type else 0.0
                score += max(0.0, 0.08 - min(distance, 80.0) / 1000.0)
                candidate_pairs.append((score, before_index, after_index))

        candidate_pairs.sort(key=lambda item: item[0], reverse=True)

        modified_items: list[AnnotationDiffItem] = []
        for score, before_index, after_index in candidate_pairs:
            if before_index not in unmatched_before or after_index not in unmatched_after:
                continue
            changed_fields = self._changed_fields(before[before_index], after[after_index])
            if not changed_fields:
                unmatched_before.remove(before_index)
                unmatched_after.remove(after_index)
                continue
            unmatched_before.remove(before_index)
            unmatched_after.remove(after_index)
            modified_items.append(
                AnnotationDiffItem(
                    change_type="modified",
                    before=before[before_index],
                    after=after[after_index],
                    changed_fields=changed_fields,
                    match_score=round(score, 5),
                )
            )

        added_items = [
            AnnotationDiffItem(change_type="added", after=after[index])
            for index in sorted(unmatched_after)
        ]
        removed_items = [
            AnnotationDiffItem(change_type="removed", before=before[index])
            for index in sorted(unmatched_before)
        ]

        changed_items = [*modified_items, *added_items, *removed_items]
        summary = AnnotationChangeSummary(
            added=len(added_items),
            removed=len(removed_items),
            modified=len(modified_items),
        )
        return summary, changed_items

    def _find_revision(self, revisions: list[AnnotationRevision], revision_id: str) -> AnnotationRevision:
        for revision in revisions:
            if revision.revision_id == revision_id:
                return revision
        raise ValueError("revision not found")

    def _parent_revision(self, revisions: list[AnnotationRevision], revision: AnnotationRevision) -> Optional[AnnotationRevision]:
        if revision.parent_revision_id:
            for candidate in revisions:
                if candidate.revision_id == revision.parent_revision_id:
                    return candidate
        return None

    def list_history(self, task_id: str) -> AnnotationHistoryResponse:
        task_id = self._validate_task_id(task_id)
        revisions = self._read_revisions(task_id)
        items: list[AnnotationHistoryItem] = []
        for revision in reversed(revisions):
            parent = self._parent_revision(revisions, revision)
            summary, _ = self._build_diff(
                before=parent.annotations if parent else [],
                after=revision.annotations,
            )
            items.append(
                AnnotationHistoryItem(
                    revision_id=revision.revision_id,
                    task_id=revision.task_id,
                    parent_revision_id=revision.parent_revision_id,
                    rollback_of_revision_id=revision.rollback_of_revision_id,
                    submitted_by=revision.submitted_by,
                    saved_at=revision.saved_at,
                    annotation_count=len(revision.annotations),
                    change_summary=summary,
                )
            )
        return AnnotationHistoryResponse(task_id=task_id, revisions=items)

    def _dataset_v2_path(self) -> Path:
        return self.dataset_root / "annotations.json"

    def _resolve_task_paths(self, task_id: str) -> dict[str, str]:
        tasks = DatasetPreprocessService.load_task_manifest(self.dataset_root)
        for task in tasks:
            if task.task_id != task_id:
                continue
            return {
                "new": task.new_path.relative_to(self.dataset_root).as_posix() if task.new_path else "",
                "old": task.old_path.relative_to(self.dataset_root).as_posix() if task.old_path else "",
                "new_marked": (
                    task.new_marked_path.relative_to(self.dataset_root).as_posix()
                    if task.new_marked_path
                    else ""
                ),
            }
        return {}

    def _upsert_dataset_v2(
        self,
        task_id: str,
        source_view: str,
        annotations: list[AnnotationBox],
        saved_at: str,
    ) -> str:
        dataset_path = self._dataset_v2_path()
        default_doc: dict[str, Any] = {
            "version": "2.2",
            "images": [],
            "updated_at": saved_at,
        }

        if dataset_path.exists():
            try:
                doc = json.loads(dataset_path.read_text(encoding="utf-8"))
                if not isinstance(doc, dict):
                    doc = default_doc
            except Exception:
                doc = default_doc
        else:
            doc = default_doc

        images = doc.get("images")
        if not isinstance(images, list):
            images = []

        task_paths = self._resolve_task_paths(task_id)
        resolved_file = task_paths.get(source_view) or f"{source_view}/{task_id}.fts"
        new_image_entry = {
            "id": task_id,
            "file": resolved_file,
            "source_view": source_view,
            "updated_at": saved_at,
            "annotations": [ann.model_dump(exclude_none=True) for ann in annotations],
        }
        if task_paths:
            new_image_entry["paths"] = task_paths
            file_name = Path(task_paths.get("new") or task_paths.get("old") or resolved_file).name
            new_image_entry["file_name"] = file_name

        updated = False
        for index, image_item in enumerate(images):
            if isinstance(image_item, dict) and str(image_item.get("id") or "") == task_id:
                images[index] = new_image_entry
                updated = True
                break

        if not updated:
            images.append(new_image_entry)

        doc["images"] = images
        doc["version"] = "2.2"
        doc["updated_at"] = saved_at

        with self._connect_db() as connection:
            connection.execute(
                """
                INSERT INTO annotation_task_state (task_id, source_view, updated_at, image_entry_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    source_view=excluded.source_view,
                    updated_at=excluded.updated_at,
                    image_entry_json=excluded.image_entry_json
                """,
                (
                    task_id,
                    source_view,
                    saved_at,
                    json.dumps(new_image_entry, ensure_ascii=False),
                ),
            )
            connection.execute(
                """
                INSERT INTO annotation_dataset_snapshot (id, updated_at, doc_json)
                VALUES (1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    updated_at=excluded.updated_at,
                    doc_json=excluded.doc_json
                """,
                (
                    saved_at,
                    json.dumps(doc, ensure_ascii=False),
                ),
            )
            connection.commit()

        dataset_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        return dataset_path.relative_to(self.dataset_root).as_posix()

    def get_revision(self, task_id: str, revision_id: str) -> AnnotationRevisionDetail:
        task_id = self._validate_task_id(task_id)
        revisions = self._read_revisions(task_id)
        revision = self._find_revision(revisions, revision_id)
        parent = self._parent_revision(revisions, revision)
        summary, changed_items = self._build_diff(
            before=parent.annotations if parent else [],
            after=revision.annotations,
        )
        return AnnotationRevisionDetail(
            **revision.model_dump(),
            change_summary=summary,
            changed_items=changed_items,
        )

    def rollback_to_revision(
        self,
        task_id: str,
        revision_id: str,
        submitted_by: str,
    ) -> AnnotationRollbackResponse:
        task_id = self._validate_task_id(task_id)
        revisions = self._read_revisions(task_id)
        target_revision = self._find_revision(revisions, revision_id)
        latest_revision = revisions[-1] if revisions else None

        saved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        new_revision = AnnotationRevision(
            revision_id=uuid.uuid4().hex,
            task_id=task_id,
            source_view=target_revision.source_view,
            parent_revision_id=latest_revision.revision_id if latest_revision else None,
            rollback_of_revision_id=target_revision.revision_id,
            submitted_by=submitted_by,
            saved_at=saved_at,
            metadata={
                **target_revision.metadata,
                "rollback": True,
                "rollback_source_revision_id": target_revision.revision_id,
            },
            annotations=target_revision.annotations,
        )
        self._append_revision(new_revision)

        self._upsert_dataset_v2(
            task_id=task_id,
            source_view=target_revision.source_view or "new",
            annotations=target_revision.annotations,
            saved_at=saved_at,
        )

        return AnnotationRollbackResponse(
            task_id=task_id,
            rolled_back_to_revision_id=target_revision.revision_id,
            new_revision_id=new_revision.revision_id,
            saved_count=len(target_revision.annotations),
        )

    def save(
        self,
        task_id: str,
        payload: AnnotationSaveRequest,
        submitted_by: str = "system",
    ) -> AnnotationSaveResponse:
        task_id = self._validate_task_id(task_id)
        existing_revisions = self._read_revisions(task_id)
        parent_revision = existing_revisions[-1] if existing_revisions else None
        saved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        revision = AnnotationRevision(
            revision_id=uuid.uuid4().hex,
            task_id=task_id,
            source_view=payload.source_view,
            parent_revision_id=parent_revision.revision_id if parent_revision else None,
            submitted_by=submitted_by,
            saved_at=saved_at,
            metadata=payload.metadata,
            annotations=payload.annotations,
        )
        self._append_revision(revision)

        saved_path = self._upsert_dataset_v2(
            task_id=task_id,
            source_view=payload.source_view or "new",
            annotations=payload.annotations,
            saved_at=saved_at,
        )

        return AnnotationSaveResponse(
            task_id=task_id,
            format_version="v2",
            saved_path=saved_path,
            saved_count=len(payload.annotations),
        )
