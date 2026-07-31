"""Offline evaluation registration and immutable artifact orchestration."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from scann.ai.candidate_evaluation import (
    evaluate_candidate_records,
    write_evaluation_artifact,
)
from scann.core.dataset_storage import DatasetStorage
from scann.core.discovery_storage import DiscoveryStorage


EvaluationRunType = Literal["candidate", "injection", "model", "gold"]


class EvaluationCreateRequest(BaseModel):
    run_type: EvaluationRunType
    partition_id: Optional[str] = None
    model_id: Optional[str] = None
    config: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    records: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    per_task: list[dict[str, Any]] = Field(default_factory=list)


class EvaluationResponse(BaseModel):
    run_id: str
    run_type: str
    status: str
    partition_id: Optional[str] = None
    model_id: Optional[str] = None
    artifact_relpath: Optional[str] = None
    artifact_sha256: Optional[str] = None
    config: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    created_by: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class EvaluationService:
    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self._dataset_storage = DatasetStorage(self.dataset_root)
        self._dataset_storage.ensure_schema()
        self._storage = DiscoveryStorage(self.dataset_root)
        configured_root = os.getenv("SCANN_EVALUATION_ROOT", "").strip()
        self.evaluation_root = (
            Path(configured_root).resolve()
            if configured_root
            else (
                self.dataset_root
                / ".scann_control"
                / "evaluations"
            ).resolve()
        )

    def _artifact_location(self, run_path: str) -> str:
        path = Path(run_path).resolve()
        try:
            return path.relative_to(self.dataset_root).as_posix()
        except ValueError:
            return str(path)

    def _validate_references(
        self,
        *,
        partition_id: str | None,
        model_id: str | None,
    ) -> None:
        if (
            partition_id
            and self._dataset_storage.get_dataset_partition(partition_id)
            is None
        ):
            raise ValueError("dataset partition not found")
        if (
            model_id
            and self._dataset_storage.get_registered_model(model_id) is None
        ):
            raise ValueError("model not found")

    def create(
        self,
        payload: EvaluationCreateRequest,
        *,
        created_by: str,
    ) -> EvaluationResponse:
        self._validate_references(
            partition_id=payload.partition_id,
            model_id=payload.model_id,
        )
        run_id = f"evaluation-{uuid.uuid4().hex[:24]}"
        config = dict(payload.config)
        record = self._storage.create_evaluation(
            run_id=run_id,
            run_type=payload.run_type,
            status="running" if (
                payload.records or payload.metrics
            ) else "registered",
            partition_id=payload.partition_id,
            model_id=payload.model_id,
            config=config,
            created_by=created_by,
        )
        if not payload.records and not payload.metrics:
            return EvaluationResponse.model_validate(record)

        try:
            if payload.run_type == "candidate":
                if not payload.records:
                    raise ValueError(
                        "candidate evaluation requires records"
                    )
                metrics, per_task = evaluate_candidate_records(
                    payload.records,
                    iou_threshold=float(
                        config.get("iou_threshold", 0.1)
                    ),
                    center_distance_threshold=float(
                        config.get(
                            "center_distance_threshold",
                            8.0,
                        )
                    ),
                )
            else:
                if not payload.metrics:
                    raise ValueError(
                        f"{payload.run_type} evaluation requires metrics"
                    )
                metrics = dict(payload.metrics)
                per_task = list(payload.per_task)
            artifact = write_evaluation_artifact(
                self.evaluation_root,
                run_id=run_id,
                run_type=payload.run_type,
                config=config,
                metrics=metrics,
                per_task=per_task,
                provenance=payload.provenance,
            )
            record = self._storage.update_evaluation(
                run_id,
                status="completed",
                artifact_relpath=self._artifact_location(
                    str(artifact["run_path"])
                ),
                artifact_sha256=str(artifact["manifest_sha256"]),
                metrics=metrics,
            )
        except Exception as exc:
            self._storage.update_evaluation(
                run_id,
                status="failed",
                error_message=str(exc),
            )
            raise
        return EvaluationResponse.model_validate(record)

    def get(self, run_id: str) -> EvaluationResponse | None:
        record = self._storage.get_evaluation(run_id)
        return (
            EvaluationResponse.model_validate(record)
            if record is not None
            else None
        )

    def list(self, *, limit: int = 100) -> list[EvaluationResponse]:
        return [
            EvaluationResponse.model_validate(item)
            for item in self._storage.list_evaluations(limit=limit)
        ]
