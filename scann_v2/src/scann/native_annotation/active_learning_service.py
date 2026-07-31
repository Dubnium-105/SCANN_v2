"""Active-learning batch orchestration and persistence."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from scann.ai.active_learning import (
    ACTIVE_LEARNING_STRATEGY_VERSION,
    select_active_learning_batch,
)
from scann.core.dataset_storage import DatasetStorage
from scann.core.discovery_storage import DiscoveryStorage


class ActiveLearningItemRequest(BaseModel):
    task_id: str = Field(..., min_length=1)
    uncertainty: float = Field(0.0, ge=0.0, le=1.0)
    model_disagreement: float = Field(0.0, ge=0.0, le=1.0)
    embedding_diversity: float = Field(0.0, ge=0.0, le=1.0)
    rare_class_value: float = Field(0.0, ge=0.0, le=1.0)
    recency_or_domain_shift: float = Field(0.0, ge=0.0, le=1.0)
    group_key: Optional[str] = None
    embedding: list[float] = Field(default_factory=list)
    ood: bool = False
    high_business_value: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActiveLearningBatchCreateRequest(BaseModel):
    batch_name: str = Field(..., min_length=1)
    budget: int = Field(..., ge=1, le=10000)
    items: list[ActiveLearningItemRequest] = Field(default_factory=list)
    model_id: Optional[str] = None
    partition_id: Optional[str] = None
    seed: int = 42
    max_per_group: int = Field(3, ge=1)
    duplicate_similarity: float = Field(0.98, ge=-1.0, le=1.0)
    dual_review_fraction: float = Field(0.10, ge=0.0, le=1.0)
    weights: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActiveLearningBatchResponse(BaseModel):
    batch_id: str
    batch_name: str
    status: str
    strategy_version: str
    model_id: Optional[str] = None
    partition_id: Optional[str] = None
    budget: int
    config: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    created_by: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    items: list[dict[str, Any]] = Field(default_factory=list)


class ActiveLearningService:
    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self._dataset_storage = DatasetStorage(self.dataset_root)
        self._dataset_storage.ensure_schema()
        self._discovery_storage = DiscoveryStorage(self.dataset_root)

    def create_batch(
        self,
        payload: ActiveLearningBatchCreateRequest,
        *,
        created_by: str,
    ) -> ActiveLearningBatchResponse:
        if not payload.items:
            raise ValueError("active-learning items cannot be empty")
        task_ids = [item.task_id.strip() for item in payload.items]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("active-learning items contain duplicate task_id")
        missing_task_ids = [
            task_id
            for task_id in task_ids
            if self._dataset_storage.get_task_by_id(task_id) is None
        ]
        if missing_task_ids:
            raise ValueError(
                "active-learning tasks not found: "
                + ", ".join(missing_task_ids[:10])
            )
        if (
            payload.model_id
            and self._dataset_storage.get_registered_model(payload.model_id)
            is None
        ):
            raise ValueError("model not found")
        if (
            payload.partition_id
            and self._dataset_storage.get_dataset_partition(
                payload.partition_id
            )
            is None
        ):
            raise ValueError("dataset partition not found")

        selection = select_active_learning_batch(
            [item.model_dump() for item in payload.items],
            budget=payload.budget,
            seed=payload.seed,
            max_per_group=payload.max_per_group,
            duplicate_similarity=payload.duplicate_similarity,
            dual_review_fraction=payload.dual_review_fraction,
            weights=payload.weights,
        )
        config = {
            "seed": payload.seed,
            "max_per_group": payload.max_per_group,
            "duplicate_similarity": payload.duplicate_similarity,
            "dual_review_fraction": payload.dual_review_fraction,
            "weights": selection["weights"],
            "metadata": payload.metadata,
        }
        summary = {
            key: value
            for key, value in selection.items()
            if key != "items"
        }
        record = self._discovery_storage.create_active_learning_batch(
            batch_id=f"al-{uuid.uuid4().hex[:24]}",
            batch_name=payload.batch_name.strip(),
            status="ready",
            strategy_version=ACTIVE_LEARNING_STRATEGY_VERSION,
            model_id=payload.model_id,
            partition_id=payload.partition_id,
            budget=payload.budget,
            config=config,
            summary=summary,
            items=selection["items"],
            created_by=created_by,
        )
        return ActiveLearningBatchResponse.model_validate(record)

    def get_batch(
        self,
        batch_id: str,
    ) -> ActiveLearningBatchResponse | None:
        record = self._discovery_storage.get_active_learning_batch(batch_id)
        return (
            ActiveLearningBatchResponse.model_validate(record)
            if record is not None
            else None
        )

    def list_batches(
        self,
        *,
        limit: int = 100,
    ) -> list[ActiveLearningBatchResponse]:
        return [
            ActiveLearningBatchResponse.model_validate(item)
            for item in self._discovery_storage.list_active_learning_batches(
                limit=limit
            )
        ]
