from __future__ import annotations

import uuid
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, cast

from pydantic import BaseModel, Field

from scann.core.dataset_storage import DatasetStorage
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
        self._storage = DatasetStorage(self.dataset_root)
        self._storage.ensure_schema()

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

    def _manifest_path_for_response(self) -> str:
        db_path = self._storage.db_path
        try:
            return db_path.relative_to(self.dataset_root).as_posix()
        except ValueError:
            return str(db_path)

    def _append_revision(self, revision: AnnotationRevision, *, origin: str) -> None:
        self._storage.append_annotation_revision(
            task_id=revision.task_id,
            source_view=revision.source_view,
            parent_revision_id=revision.parent_revision_id,
            rollback_of_revision_id=revision.rollback_of_revision_id,
            submitted_by=revision.submitted_by,
            origin=origin,
            saved_at=revision.saved_at,
            metadata=revision.metadata,
            annotations=[ann.model_dump(exclude_none=True) for ann in revision.annotations],
            revision_id=revision.revision_id,
        )

    @staticmethod
    def _candidate_task_ids(task_id: str) -> list[str]:
        candidates: list[str] = []

        def _append(value: str) -> None:
            normalized = value.strip()
            if not normalized or normalized in candidates:
                return
            candidates.append(normalized)

        _append(task_id)
        stripped = DatasetPreprocessService.strip_aligned_crop_suffix(task_id)
        _append(stripped)

        date_token = DatasetPreprocessService.extract_datetime_prefix(stripped)
        field_name = DatasetStorage.normalize_field_name(stripped)
        if date_token and field_name:
            _append(f"{date_token}__{field_name}")
        if field_name:
            _append(field_name)
        return candidates

    def _ensure_task_exists(self, task_id: str) -> str:
        for candidate in self._candidate_task_ids(task_id):
            if self._storage.get_task_by_id(candidate) is not None:
                return candidate
        DatasetPreprocessService().prepare_annotation_dataset(self.dataset_root)
        for candidate in self._candidate_task_ids(task_id):
            if self._storage.get_task_by_id(candidate) is not None:
                return candidate
        raise ValueError("task not found")

    def _read_revisions(self, task_id: str) -> list[AnnotationRevision]:
        revisions = self._storage.list_annotation_revisions(task_id)
        result: list[AnnotationRevision] = []
        for item in revisions:
            annotations = [
                AnnotationBox.model_validate(ann)
                for ann in item.get("annotations", [])
                if isinstance(ann, dict)
            ]
            result.append(
                AnnotationRevision(
                    revision_id=str(item["revision_id"]),
                    task_id=str(item["task_id"]),
                    source_view=self._normalize_source_view(
                        str(item["source_view"]) if item.get("source_view") is not None else None
                    ),
                    parent_revision_id=(
                        str(item["parent_revision_id"]) if item.get("parent_revision_id") is not None else None
                    ),
                    rollback_of_revision_id=(
                        str(item["rollback_of_revision_id"])
                        if item.get("rollback_of_revision_id") is not None
                        else None
                    ),
                    submitted_by=str(item["submitted_by"]),
                    saved_at=str(item["saved_at"]),
                    metadata=item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
                    annotations=annotations,
                )
            )
        return result

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

    def _upsert_current_annotation(
        self,
        task_id: str,
        source_view: str,
        annotations: list[AnnotationBox],
    ) -> None:
        label = None
        detail_type = None
        if annotations:
            label = annotations[0].label
            detail_type = annotations[0].detail_type
        self._storage.upsert_current_annotation(
            task_id=task_id,
            source_view=source_view,
            label=label,
            detail_type=detail_type,
            ai_suggestion=None,
            ai_confidence=None,
            annotations=[ann.model_dump(exclude_none=True) for ann in annotations],
            annotation_origin="online",
        )

    def list_history(self, task_id: str) -> AnnotationHistoryResponse:
        task_id = self._validate_task_id(task_id)
        task_id = self._ensure_task_exists(task_id)
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

    def get_revision(self, task_id: str, revision_id: str) -> AnnotationRevisionDetail:
        task_id = self._validate_task_id(task_id)
        task_id = self._ensure_task_exists(task_id)
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
        task_id = self._ensure_task_exists(task_id)
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
        self._append_revision(new_revision, origin="online")
        self._upsert_current_annotation(
            task_id=task_id,
            source_view=target_revision.source_view or "new",
            annotations=target_revision.annotations,
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
        task_id = self._ensure_task_exists(task_id)
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
        self._append_revision(revision, origin="online")
        self._upsert_current_annotation(
            task_id=task_id,
            source_view=payload.source_view or "new",
            annotations=payload.annotations,
        )

        return AnnotationSaveResponse(
            task_id=task_id,
            format_version="v2",
            saved_path=self._manifest_path_for_response(),
            saved_count=len(payload.annotations),
        )
