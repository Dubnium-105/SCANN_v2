"""Explicit, auditable shadow/canary/promotion/rollback orchestration."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from scann.ai.model_governance import (
    DeploymentStage,
    evaluate_promotion_gate,
    validate_stage_transition,
)
from scann.core.dataset_storage import DatasetStorage
from scann.core.discovery_storage import DiscoveryStorage

from .training_lifecycle_service import TrainingLifecycleService


class ShadowDeploymentRequest(BaseModel):
    evaluation_run_id: str = Field(..., min_length=1)
    config: dict[str, Any] = Field(default_factory=dict)


class CanaryDeploymentRequest(BaseModel):
    traffic_fraction: float = Field(0.10, gt=0.0, le=0.50)
    human_approved: bool = False
    config: dict[str, Any] = Field(default_factory=dict)


class PromotionDeploymentRequest(BaseModel):
    evaluation_run_id: str = Field(..., min_length=1)
    taxonomy_version: str = Field(..., min_length=1)
    partition_id: str = Field(..., min_length=1)
    required_metrics: dict[str, float] = Field(default_factory=dict)
    unsupported_required_classes: list[str] = Field(default_factory=list)
    shadow_drift_ok: bool = False
    canary_review_ok: bool = False
    human_approved: bool = False
    config: dict[str, Any] = Field(default_factory=dict)


class RollbackDeploymentRequest(BaseModel):
    target_model_id: str = Field(..., min_length=1)
    human_approved: bool = False
    reason: str = Field(..., min_length=1)


class ModelDeploymentResponse(BaseModel):
    deployment_id: str
    model_id: str
    stage: str
    status: str
    traffic_fraction: float
    previous_deployment_id: Optional[str] = None
    config: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    created_by: Optional[str] = None
    created_at: Optional[str] = None
    ended_at: Optional[str] = None


class DeploymentService:
    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self._dataset_storage = DatasetStorage(self.dataset_root)
        self._dataset_storage.ensure_schema()
        self._storage = DiscoveryStorage(self.dataset_root)
        self._training = TrainingLifecycleService(self.dataset_root)

    def _require_model(self, model_id: str):
        model = self._dataset_storage.get_registered_model(model_id)
        if model is None:
            raise ValueError("model not found")
        artifact_path = self._training.get_model_artifact_path(model_id)
        artifact_metadata = (
            model.metadata.get("artifact")
            if isinstance(model.metadata, dict)
            and isinstance(model.metadata.get("artifact"), dict)
            else {}
        )
        expected_sha256 = str(
            artifact_metadata.get("sha256") or ""
        ).strip().lower()
        if expected_sha256:
            digest = hashlib.sha256()
            with artifact_path.open("rb") as handle:
                for chunk in iter(
                    lambda: handle.read(1024 * 1024),
                    b"",
                ):
                    digest.update(chunk)
            if digest.hexdigest() != expected_sha256:
                raise ValueError("model artifact hash mismatch")
        return model

    def _require_evaluation(
        self,
        run_id: str,
        *,
        model_id: str,
    ) -> dict[str, Any]:
        evaluation = self._storage.get_evaluation(run_id)
        if evaluation is None:
            raise ValueError("evaluation run not found")
        if evaluation["status"] != "completed":
            raise ValueError("evaluation run is not completed")
        evaluated_model = str(evaluation.get("model_id") or "").strip()
        if evaluated_model and evaluated_model != model_id:
            raise ValueError("evaluation run belongs to another model")
        return evaluation

    def _create_stage(
        self,
        *,
        model_id: str,
        stage: DeploymentStage,
        created_by: str,
        traffic_fraction: float = 0.0,
        config: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        validate_transition: bool = True,
    ) -> dict[str, Any]:
        previous = self._storage.latest_deployment_for_model(model_id)
        if previous is not None and validate_transition:
            validate_stage_transition(previous["stage"], stage)
        return self._storage.create_deployment(
            deployment_id=f"deployment-{uuid.uuid4().hex[:24]}",
            model_id=model_id,
            stage=stage.value,
            status="active",
            traffic_fraction=traffic_fraction,
            previous_deployment_id=(
                str(previous["deployment_id"])
                if previous is not None
                else None
            ),
            config=config or {},
            metrics=metrics or {},
            created_by=created_by,
            end_previous=previous is not None,
        )

    def start_shadow(
        self,
        model_id: str,
        payload: ShadowDeploymentRequest,
        *,
        created_by: str,
    ) -> ModelDeploymentResponse:
        self._require_model(model_id)
        evaluation = self._require_evaluation(
            payload.evaluation_run_id,
            model_id=model_id,
        )
        latest = self._storage.latest_deployment_for_model(model_id)
        if latest is None:
            self._create_stage(
                model_id=model_id,
                stage=DeploymentStage.REGISTERED,
                created_by=created_by,
                validate_transition=False,
                config={"source": "model_registry"},
            )
            latest = self._storage.latest_deployment_for_model(model_id)
        if latest is not None and latest["stage"] == DeploymentStage.SHADOW:
            return ModelDeploymentResponse.model_validate(latest)
        if latest is not None and latest["stage"] == DeploymentStage.REGISTERED:
            self._create_stage(
                model_id=model_id,
                stage=DeploymentStage.OFFLINE_PASSED,
                created_by=created_by,
                config={
                    "evaluation_run_id": payload.evaluation_run_id,
                },
                metrics=evaluation["metrics"],
            )
        shadow = self._create_stage(
            model_id=model_id,
            stage=DeploymentStage.SHADOW,
            created_by=created_by,
            config={
                **payload.config,
                "evaluation_run_id": payload.evaluation_run_id,
                "affects_visible_prelabels": False,
            },
            metrics=evaluation["metrics"],
        )
        return ModelDeploymentResponse.model_validate(shadow)

    def start_canary(
        self,
        model_id: str,
        payload: CanaryDeploymentRequest,
        *,
        created_by: str,
    ) -> ModelDeploymentResponse:
        self._require_model(model_id)
        if not payload.human_approved:
            raise ValueError("human approval is required for canary")
        canary = self._create_stage(
            model_id=model_id,
            stage=DeploymentStage.CANARY,
            created_by=created_by,
            traffic_fraction=payload.traffic_fraction,
            config={
                **payload.config,
                "selection": "stable_task_hash",
                "human_approved": True,
            },
        )
        return ModelDeploymentResponse.model_validate(canary)

    def promote(
        self,
        model_id: str,
        payload: PromotionDeploymentRequest,
        *,
        created_by: str,
    ) -> ModelDeploymentResponse:
        self._require_model(model_id)
        if (
            self._dataset_storage.get_dataset_partition(
                payload.partition_id
            )
            is None
        ):
            raise ValueError("dataset partition not found")
        evaluation = self._require_evaluation(
            payload.evaluation_run_id,
            model_id=model_id,
        )
        if (
            evaluation.get("partition_id")
            and evaluation["partition_id"] != payload.partition_id
        ):
            raise ValueError("evaluation partition does not match promotion")
        gate = evaluate_promotion_gate(
            artifact_valid=True,
            taxonomy_version=payload.taxonomy_version,
            partition_id=payload.partition_id,
            gold_metrics=evaluation["metrics"],
            required_metrics=payload.required_metrics,
            unsupported_required_classes=(
                payload.unsupported_required_classes
            ),
            shadow_drift_ok=payload.shadow_drift_ok,
            canary_review_ok=payload.canary_review_ok,
            human_approved=payload.human_approved,
        )
        if not gate["passed"]:
            raise ValueError(
                "promotion gate failed: "
                + ", ".join(gate["failures"])
            )
        previous_promoted = [
            item
            for item in self._storage.list_deployments(
                stage=DeploymentStage.PROMOTED.value,
                status="active",
                limit=500,
            )
            if item["model_id"] != model_id
        ]
        promoted_model = self._training.promote_model(
            model_id=model_id,
            promoted_by=created_by,
            enqueue_prelabels=False,
        )
        if promoted_model is None:
            raise ValueError("model not found")
        deployment = self._create_stage(
            model_id=model_id,
            stage=DeploymentStage.PROMOTED,
            created_by=created_by,
            traffic_fraction=1.0,
            config={
                **payload.config,
                "evaluation_run_id": payload.evaluation_run_id,
                "taxonomy_version": payload.taxonomy_version,
                "partition_id": payload.partition_id,
                "promotion_gate": gate,
                "auto_promotion": False,
                "replaces_deployment_ids": [
                    item["deployment_id"]
                    for item in previous_promoted
                ],
            },
            metrics=evaluation["metrics"],
        )
        for previous in previous_promoted:
            self._storage.end_deployment(
                str(previous["deployment_id"])
            )
        return ModelDeploymentResponse.model_validate(deployment)

    def rollback(
        self,
        model_id: str,
        payload: RollbackDeploymentRequest,
        *,
        created_by: str,
    ) -> ModelDeploymentResponse:
        self._require_model(model_id)
        self._require_model(payload.target_model_id)
        if payload.target_model_id == model_id:
            raise ValueError("rollback target must be a different model")
        if not payload.human_approved:
            raise ValueError("human approval is required for rollback")
        current = self._storage.latest_deployment_for_model(model_id)
        if current is None or current["stage"] != DeploymentStage.PROMOTED:
            raise ValueError("current model is not promoted")
        target_history = self._storage.latest_deployment_for_model(
            payload.target_model_id
        )
        if (
            target_history is None
            or target_history["stage"]
            not in {
                DeploymentStage.PROMOTED,
                DeploymentStage.RETIRED,
            }
        ):
            raise ValueError(
                "rollback target has no previously valid deployment"
            )
        self._create_stage(
            model_id=model_id,
            stage=DeploymentStage.RETIRED,
            created_by=created_by,
            config={
                "rollback_reason": payload.reason,
                "rollback_target_model_id": payload.target_model_id,
            },
        )
        promoted_model = self._training.promote_model(
            model_id=payload.target_model_id,
            promoted_by=created_by,
            enqueue_prelabels=False,
        )
        if promoted_model is None:
            raise ValueError("rollback target model not found")
        target = self._create_stage(
            model_id=payload.target_model_id,
            stage=DeploymentStage.PROMOTED,
            created_by=created_by,
            traffic_fraction=1.0,
            config={
                "rollback": True,
                "rollback_from_model_id": model_id,
                "rollback_reason": payload.reason,
                "human_approved": True,
            },
            validate_transition=False,
        )
        return ModelDeploymentResponse.model_validate(target)

    def list(
        self,
        *,
        model_id: str | None = None,
        limit: int = 100,
    ) -> list[ModelDeploymentResponse]:
        return [
            ModelDeploymentResponse.model_validate(item)
            for item in self._storage.list_deployments(
                model_id=model_id,
                limit=limit,
            )
        ]
