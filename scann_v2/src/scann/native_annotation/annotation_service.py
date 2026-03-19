from __future__ import annotations

import json
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


class AnnotationSaveRequest(BaseModel):
    bucket: Literal["positive", "negative"]
    annotations: List[AnnotationBox]
    source_view: Optional[Literal["old", "new", "new_marked"]] = "new"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnnotationSaveResponse(BaseModel):
    task_id: str
    bucket: str
    saved_path: str
    saved_count: int


class AnnotationService:
    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = dataset_root.resolve()

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

    def save(self, task_id: str, payload: AnnotationSaveRequest) -> AnnotationSaveResponse:
        task_id = self._validate_task_id(task_id)
        output_dir = self.dataset_root / payload.bucket
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{task_id}.json"
        document = {
            "task_id": task_id,
            "bucket": payload.bucket,
            "source_view": payload.source_view,
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "metadata": payload.metadata,
            "annotations": [ann.model_dump() for ann in payload.annotations],
        }
        output_file.write_text(
            json.dumps(document, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return AnnotationSaveResponse(
            task_id=task_id,
            bucket=payload.bucket,
            saved_path=output_file.relative_to(self.dataset_root).as_posix(),
            saved_count=len(payload.annotations),
        )
