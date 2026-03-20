from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


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
    submitted_by: str
    saved_at: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    annotations: List[AnnotationBox]


class AnnotationHistoryItem(BaseModel):
    revision_id: str
    task_id: str
    submitted_by: str
    saved_at: str
    format_version: str = "v2"
    annotation_count: int


class AnnotationHistoryResponse(BaseModel):
    task_id: str
    revisions: List[AnnotationHistoryItem]


class AnnotationService:
    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = dataset_root.resolve()
        self.revisions_dir = self.dataset_root / "annotation_revisions"

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

    def _history_file_path(self, task_id: str) -> Path:
        return self.revisions_dir / f"{task_id}.jsonl"

    def _append_revision(self, revision: AnnotationRevision) -> None:
        self.revisions_dir.mkdir(parents=True, exist_ok=True)
        history_file = self._history_file_path(revision.task_id)
        with history_file.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(revision.model_dump(), ensure_ascii=False))
            fp.write("\n")

    def _read_revisions(self, task_id: str) -> list[AnnotationRevision]:
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

    def list_history(self, task_id: str) -> AnnotationHistoryResponse:
        task_id = self._validate_task_id(task_id)
        revisions = self._read_revisions(task_id)
        items = [
            AnnotationHistoryItem(
                revision_id=rev.revision_id,
                task_id=rev.task_id,
                submitted_by=rev.submitted_by,
                saved_at=rev.saved_at,
                annotation_count=len(rev.annotations),
            )
            for rev in reversed(revisions)
        ]
        return AnnotationHistoryResponse(task_id=task_id, revisions=items)

    def _dataset_v2_path(self) -> Path:
        return self.dataset_root / "annotations.json"

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

        new_image_entry = {
            "id": task_id,
            "file": f"{source_view}/{task_id}.fts",
            "source_view": source_view,
            "updated_at": saved_at,
            "annotations": [ann.model_dump(exclude_none=True) for ann in annotations],
        }

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
        dataset_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        return dataset_path.relative_to(self.dataset_root).as_posix()

    def get_revision(self, task_id: str, revision_id: str) -> AnnotationRevision:
        task_id = self._validate_task_id(task_id)
        for revision in self._read_revisions(task_id):
            if revision.revision_id == revision_id:
                return revision
        raise ValueError("revision not found")

    def save(
        self,
        task_id: str,
        payload: AnnotationSaveRequest,
        submitted_by: str = "system",
    ) -> AnnotationSaveResponse:
        task_id = self._validate_task_id(task_id)
        saved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        revision = AnnotationRevision(
            revision_id=uuid.uuid4().hex,
            task_id=task_id,
            source_view=payload.source_view,
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
