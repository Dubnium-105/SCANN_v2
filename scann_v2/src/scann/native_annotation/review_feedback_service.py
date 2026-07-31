"""Versioned comparison of AI prelabels with final human revisions."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Optional

from pydantic import BaseModel, Field

from scann.ai.candidate_evaluation import EvaluationBox, match_boxes
from scann.core.dataset_storage import DatasetStorage
from scann.core.discovery_storage import DiscoveryStorage


REVIEW_MATCH_ALGORITHM_VERSION = "review-match-v1"


class ReviewFeedbackCreateRequest(BaseModel):
    task_id: str = Field(..., min_length=1)
    prelabel_id: str = Field(..., min_length=1)
    revision_id: str = Field(..., min_length=1)
    review_duration_seconds: Optional[float] = Field(default=None, ge=0.0)


class ReviewFeedbackResponse(BaseModel):
    event_id: str
    task_id: str
    prelabel_id: Optional[str] = None
    revision_id: Optional[str] = None
    model_id: Optional[str] = None
    outcome: str
    match_algorithm_version: str
    result: dict[str, Any] = Field(default_factory=dict)
    created_by: Optional[str] = None
    created_at: Optional[str] = None


def _same_text(left: Any, right: Any) -> bool:
    return str(left or "").strip().lower() == str(right or "").strip().lower()


def _finite_float(value: Any) -> float | None:
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None
    return normalized if math.isfinite(normalized) else None


def compare_review_boxes(
    prelabel_boxes: list[Mapping[str, Any]],
    human_boxes: list[Mapping[str, Any]],
    *,
    iou_threshold: float = 0.1,
    center_distance_threshold: float = 8.0,
) -> dict[str, Any]:
    ai = [EvaluationBox.from_mapping(item) for item in prelabel_boxes]
    human = [EvaluationBox.from_mapping(item) for item in human_boxes]
    matches, removed_indices, added_indices = match_boxes(
        ai,
        human,
        iou_threshold=iou_threshold,
        center_distance_threshold=center_distance_threshold,
    )
    geometry_corrections = 0
    reclassifications = 0
    normalized_matches: list[dict[str, Any]] = []
    for match in matches:
        ai_index = int(match["truth_index"])
        human_index = int(match["candidate_index"])
        ai_box = prelabel_boxes[ai_index]
        human_box = human_boxes[human_index]
        geometry_changed = (
            float(match["iou"]) < 0.95
            or float(match["center_distance"]) > 1.0
        )
        classification_changed = (
            not _same_text(ai_box.get("label"), human_box.get("label"))
            or not _same_text(
                ai_box.get("detail_type"),
                human_box.get("detail_type"),
            )
        )
        geometry_corrections += int(geometry_changed)
        reclassifications += int(classification_changed)
        normalized_matches.append(
            {
                **match,
                "prelabel_index": ai_index,
                "human_index": human_index,
                "geometry_changed": geometry_changed,
                "classification_changed": classification_changed,
                "ai_confidence": _finite_float(
                    ai_box.get("confidence")
                ),
            }
        )

    unchanged = (
        len(matches) == len(ai) == len(human)
        and geometry_corrections == 0
        and reclassifications == 0
    )
    if unchanged:
        outcome = "full_accept"
    elif ai and not matches:
        outcome = "full_reject"
    else:
        outcome = "partial_accept"
    return {
        "outcome": outcome,
        "prelabel_box_count": len(ai),
        "human_box_count": len(human),
        "matched_box_count": len(matches),
        "human_added_count": len(added_indices),
        "human_removed_count": len(removed_indices),
        "geometry_correction_count": geometry_corrections,
        "reclassification_count": reclassifications,
        "prelabel_acceptance_rate": (
            len(matches) / len(ai)
            if ai
            else None
        ),
        "matches": normalized_matches,
        "removed_prelabel_indices": removed_indices,
        "added_human_indices": added_indices,
        "iou_threshold": float(iou_threshold),
        "center_distance_threshold": float(center_distance_threshold),
    }


class ReviewFeedbackService:
    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self._dataset_storage = DatasetStorage(self.dataset_root)
        self._dataset_storage.ensure_schema()
        self._discovery_storage = DiscoveryStorage(self.dataset_root)

    @staticmethod
    def _event_id(prelabel_id: str, revision_id: str) -> str:
        payload = (
            f"{REVIEW_MATCH_ALGORITHM_VERSION}:"
            f"{prelabel_id}:{revision_id}"
        ).encode("utf-8")
        return f"review-{hashlib.sha256(payload).hexdigest()[:24]}"

    def create(
        self,
        payload: ReviewFeedbackCreateRequest,
        *,
        created_by: str,
    ) -> ReviewFeedbackResponse:
        prelabel = self._dataset_storage.get_task_prelabel_by_id(
            task_id=payload.task_id,
            prelabel_id=payload.prelabel_id,
        )
        if prelabel is None:
            raise ValueError("prelabel not found")
        prelabel_record, prelabel_boxes = prelabel

        revisions = self._dataset_storage.list_annotation_revisions(
            payload.task_id
        )
        revision = next(
            (
                item
                for item in revisions
                if str(item.get("revision_id")) == payload.revision_id
            ),
            None,
        )
        if revision is None:
            raise ValueError("annotation revision not found")
        result = compare_review_boxes(
            prelabel_boxes,
            [
                item
                for item in revision.get("annotations") or []
                if isinstance(item, Mapping)
            ],
        )
        result.update(
            {
                "review_duration_seconds": (
                    float(payload.review_duration_seconds)
                    if payload.review_duration_seconds is not None
                    else None
                ),
                "model_version": prelabel_record.model_version,
                "model_backbone": prelabel_record.model_backbone,
                "prelabel_confidence": prelabel_record.ai_confidence,
            }
        )
        event = self._discovery_storage.create_review_event(
            event_id=self._event_id(
                payload.prelabel_id,
                payload.revision_id,
            ),
            task_id=payload.task_id,
            prelabel_id=payload.prelabel_id,
            revision_id=payload.revision_id,
            model_id=prelabel_record.model_id,
            outcome=str(result["outcome"]),
            match_algorithm_version=REVIEW_MATCH_ALGORITHM_VERSION,
            result=result,
            created_by=created_by,
        )
        return ReviewFeedbackResponse.model_validate(event)

    def list(
        self,
        *,
        model_id: str | None = None,
        task_id: str | None = None,
        limit: int = 100,
    ) -> list[ReviewFeedbackResponse]:
        return [
            ReviewFeedbackResponse.model_validate(item)
            for item in self._discovery_storage.list_review_events(
                model_id=model_id,
                task_id=task_id,
                limit=limit,
            )
        ]
