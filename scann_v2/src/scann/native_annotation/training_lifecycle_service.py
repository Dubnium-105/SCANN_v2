from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from scann.ai.class_balance import (
    build_class_audit,
    merge_imbalance_config,
    sample_records_from_snapshot_document,
)
from scann.core.dataset_storage import (
    DatasetSnapshotRecord,
    DatasetStorage,
    RegisteredModelRecord,
    TrainingJobRecord,
    TrainingRunRecord,
)

from .prelabel_service import PrelabelEnqueueRequest, PrelabelService


DEFAULT_TRAINING_JOB_TIMEOUT_SECONDS = 6 * 60 * 60
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


class DatasetSnapshotCreateRequest(BaseModel):
    snapshot_name: Optional[str] = None
    task_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetSnapshotResponse(BaseModel):
    snapshot_id: str
    snapshot_name: str
    document_relpath: str
    task_count: int
    annotation_count: int
    created_by: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class TrainingJobCreateRequest(BaseModel):
    snapshot_id: Optional[str] = None
    snapshot_name: Optional[str] = None
    snapshot_task_ids: list[str] = Field(default_factory=list)
    snapshot_metadata: dict[str, Any] = Field(default_factory=dict)
    task_type: Literal["classification", "detection"] = "classification"
    model_version: str = Field(..., min_length=1)
    model_id: Optional[str] = None
    model_backbone: str = Field(..., min_length=1)
    train_config: dict[str, Any] = Field(default_factory=dict)
    priority: int = 100
    promote_on_success: bool = False
    enqueue_prelabels_on_success: bool = False
    prelabel_task_ids: list[str] = Field(default_factory=list)
    force_prelabel: bool = False


class TrainingJobResponse(BaseModel):
    job_id: str
    snapshot_id: str
    requested_by: str
    task_type: str
    model_version: str
    model_id: str
    model_backbone: str
    status: str
    train_config: dict[str, Any] = Field(default_factory=dict)
    priority: int = 100
    promote_on_success: bool = False
    enqueue_prelabels_on_success: bool = False
    prelabel_task_ids: list[str] = Field(default_factory=list)
    force_prelabel: bool = False
    claim_worker_id: Optional[str] = None
    claimed_at: Optional[str] = None
    claim_expires_at: Optional[str] = None
    last_heartbeat_at: Optional[str] = None
    attempt_count: int = 0
    error_message: Optional[str] = None
    run_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class TrainingRunResponse(BaseModel):
    run_id: str
    job_id: str
    snapshot_id: str
    task_type: str
    status: str
    worker_id: Optional[str] = None
    model_id: Optional[str] = None
    model_version: Optional[str] = None
    model_backbone: Optional[str] = None
    artifact_path: Optional[str] = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class RegisteredModelResponse(BaseModel):
    model_id: str
    model_version: str
    model_backbone: str
    task_type: str
    training_run_id: Optional[str] = None
    snapshot_id: Optional[str] = None
    artifact_path: Optional[str] = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_by: Optional[str] = None
    is_promoted: bool = False
    promoted_at: Optional[str] = None
    promoted_by: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class TrainingWorkerClaimRequest(BaseModel):
    worker_id: str = Field(..., min_length=1)
    display_name: Optional[str] = None
    host_name: Optional[str] = None
    device_label: Optional[str] = None
    capabilities: dict[str, Any] = Field(default_factory=dict)


class TrainingWorkerClaimResponse(BaseModel):
    job_id: str
    snapshot_id: str
    task_type: str
    model_version: str
    model_id: str
    model_backbone: str
    train_config: dict[str, Any] = Field(default_factory=dict)
    promote_on_success: bool = False
    enqueue_prelabels_on_success: bool = False
    prelabel_task_ids: list[str] = Field(default_factory=list)
    force_prelabel: bool = False
    claimed_at: Optional[str] = None
    claim_expires_at: Optional[str] = None


class TrainingWorkerHeartbeatRequest(BaseModel):
    worker_id: str = Field(..., min_length=1)


class TrainingWorkerJobAckResponse(BaseModel):
    job_id: str
    accepted: bool


class TrainingArtifactUploadResponse(BaseModel):
    job_id: str
    artifact_path: str


class TrainingWorkerCompleteRequest(BaseModel):
    worker_id: str = Field(..., min_length=1)
    artifact_path: str = Field(..., min_length=1)
    metrics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrainingWorkerFailRequest(BaseModel):
    worker_id: str = Field(..., min_length=1)
    error_message: str = Field(..., min_length=1)
    retryable: bool = False


class TrainingJobLifecycleResponse(BaseModel):
    job: TrainingJobResponse
    run: TrainingRunResponse
    model: RegisteredModelResponse
    prelabel_enqueue: Optional[dict[str, Any]] = None
    promotion_warnings: list[str] = Field(default_factory=list)
    auto_promoted: bool = False


class PromoteModelResponse(BaseModel):
    model: RegisteredModelResponse
    prelabel_enqueue: Optional[dict[str, Any]] = None
    promotion_warnings: list[str] = Field(default_factory=list)


class TrainingLifecycleService:
    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self._storage = DatasetStorage(self.dataset_root)
        self._storage.ensure_schema()
        self._prelabel_service = PrelabelService(self.dataset_root)
        self.job_timeout_seconds = max(
            60,
            int(os.getenv("SCANN_TRAINING_JOB_TIMEOUT_SECONDS", str(DEFAULT_TRAINING_JOB_TIMEOUT_SECONDS))),
        )

    @staticmethod
    def _snapshot_to_response(record: DatasetSnapshotRecord) -> DatasetSnapshotResponse:
        return DatasetSnapshotResponse(
            snapshot_id=record.snapshot_id,
            snapshot_name=record.snapshot_name,
            document_relpath=record.document_relpath,
            task_count=record.task_count,
            annotation_count=record.annotation_count,
            created_by=record.created_by,
            metadata=record.metadata or {},
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    @staticmethod
    def _job_to_response(record: TrainingJobRecord) -> TrainingJobResponse:
        return TrainingJobResponse(
            job_id=record.job_id,
            snapshot_id=record.snapshot_id,
            requested_by=record.requested_by,
            task_type=record.task_type,
            model_version=record.model_version,
            model_id=record.model_id,
            model_backbone=record.model_backbone,
            status=record.status,
            train_config=record.train_config or {},
            priority=record.priority,
            promote_on_success=record.promote_on_success,
            enqueue_prelabels_on_success=record.enqueue_prelabels_on_success,
            prelabel_task_ids=list(record.prelabel_task_ids or []),
            force_prelabel=record.force_prelabel,
            claim_worker_id=record.claim_worker_id,
            claimed_at=record.claimed_at,
            claim_expires_at=record.claim_expires_at,
            last_heartbeat_at=record.last_heartbeat_at,
            attempt_count=record.attempt_count,
            error_message=record.error_message,
            run_id=record.run_id,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    @staticmethod
    def _run_to_response(record: TrainingRunRecord) -> TrainingRunResponse:
        return TrainingRunResponse(
            run_id=record.run_id,
            job_id=record.job_id,
            snapshot_id=record.snapshot_id,
            task_type=record.task_type,
            status=record.status,
            worker_id=record.worker_id,
            model_id=record.model_id,
            model_version=record.model_version,
            model_backbone=record.model_backbone,
            artifact_path=record.artifact_path,
            metrics=record.metrics or {},
            metadata=record.metadata or {},
            started_at=record.started_at,
            finished_at=record.finished_at,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    @staticmethod
    def _model_to_response(record: RegisteredModelRecord) -> RegisteredModelResponse:
        return RegisteredModelResponse(
            model_id=record.model_id,
            model_version=record.model_version,
            model_backbone=record.model_backbone,
            task_type=record.task_type,
            training_run_id=record.training_run_id,
            snapshot_id=record.snapshot_id,
            artifact_path=record.artifact_path,
            metrics=record.metrics or {},
            metadata=record.metadata or {},
            created_by=record.created_by,
            is_promoted=record.is_promoted,
            promoted_at=record.promoted_at,
            promoted_by=record.promoted_by,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    @staticmethod
    def _normalize_model_id(value: str | None, *, model_version: str) -> str:
        normalized = str(value or "").strip()
        if normalized:
            return normalized
        return f"{model_version}-{uuid.uuid4().hex[:8]}"

    @property
    def _control_root(self) -> Path:
        path = self.dataset_root / ".scann_control"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def _snapshot_root(self) -> Path:
        path = self._control_root / "training_snapshots"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def _model_root(self) -> Path:
        path = self._control_root / "models"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _relative_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.dataset_root).as_posix()
        except ValueError:
            return str(path.resolve())

    @staticmethod
    def _class_audit_for_document(
        document: dict[str, Any],
        *,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        imbalance_config = merge_imbalance_config(config)
        return build_class_audit(
            sample_records_from_snapshot_document(document),
            min_train_support=int(imbalance_config["min_train_support_warning"]),
            min_val_support=int(imbalance_config["min_val_support_warning"]),
        )

    def _load_snapshot_document(self, snapshot: DatasetSnapshotRecord) -> dict[str, Any]:
        file_path = (self.dataset_root / snapshot.document_relpath).resolve()
        try:
            file_path.relative_to(self.dataset_root)
        except ValueError as exc:
            raise ValueError("snapshot path is invalid") from exc
        if not file_path.is_file():
            raise ValueError("snapshot file does not exist")
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("snapshot document is invalid")
        return payload

    @staticmethod
    def _promotion_warnings_from_model(model: RegisteredModelRecord) -> list[str]:
        warnings: list[str] = []
        for source in (model.metrics or {}, model.metadata or {}):
            raw = source.get("promotion_warnings") if isinstance(source, dict) else None
            if isinstance(raw, list):
                warnings.extend(str(item) for item in raw if str(item).strip())
        if warnings:
            return list(dict.fromkeys(warnings))

        metrics = model.metrics or {}
        class_support = metrics.get("class_support") if isinstance(metrics, dict) else None
        if isinstance(class_support, dict):
            raw_warnings = class_support.get("promotion_warnings")
            if isinstance(raw_warnings, list):
                return [str(item) for item in raw_warnings if str(item).strip()]
        return []

    @staticmethod
    def _promotion_warnings_for_completed_job(
        *,
        job: TrainingJobRecord,
        snapshot: DatasetSnapshotRecord | None,
        metrics: dict[str, Any],
        metadata: dict[str, Any],
    ) -> list[str]:
        warnings: list[str] = []
        for source in (metrics, metadata):
            raw = source.get("promotion_warnings") if isinstance(source, dict) else None
            if isinstance(raw, list):
                warnings.extend(str(item) for item in raw if str(item).strip())
        class_support = metrics.get("class_support") if isinstance(metrics, dict) else None
        if isinstance(class_support, dict):
            raw = class_support.get("promotion_warnings")
            if isinstance(raw, list):
                warnings.extend(str(item) for item in raw if str(item).strip())
        if snapshot is not None and isinstance(snapshot.metadata, dict):
            audit = snapshot.metadata.get("class_audit")
            raw = audit.get("promotion_warnings") if isinstance(audit, dict) else None
            if isinstance(raw, list):
                warnings.extend(str(item) for item in raw if str(item).strip())
        if warnings:
            warnings.append("auto_promotion_suppressed_due_to_class_coverage")
        return list(dict.fromkeys(warnings))

    def _build_snapshot_document(self, *, task_ids: list[str] | None = None) -> tuple[dict[str, Any], int, int]:
        annotations_by_id = self._storage.list_current_annotations()
        selected_ids = [task_id for task_id in dict.fromkeys(task_ids or []) if task_id]
        if selected_ids:
            images = [annotations_by_id[task_id] for task_id in selected_ids if task_id in annotations_by_id]
        else:
            images = list(annotations_by_id.values())
        images = [item for item in images if isinstance(item.get("annotations"), list) and item.get("annotations")]
        annotation_count = sum(len(item.get("annotations") or []) for item in images)
        document = {
            "version": "2.3",
            "storage": "training_snapshot",
            "images": images,
        }
        class_audit = self._class_audit_for_document(document)
        document["metadata"] = {
            "class_audit": class_audit,
        }
        return (
            document,
            len(images),
            annotation_count,
        )

    def create_snapshot(
        self,
        *,
        payload: DatasetSnapshotCreateRequest,
        created_by: str,
    ) -> DatasetSnapshotResponse:
        document, task_count, annotation_count = self._build_snapshot_document(task_ids=payload.task_ids)
        if task_count <= 0 or annotation_count <= 0:
            raise ValueError("no annotated tasks are available for snapshot")

        snapshot_id = uuid.uuid4().hex
        snapshot_name = str(payload.snapshot_name or "").strip() or f"snapshot-{snapshot_id[:8]}"
        snapshot_path = self._snapshot_root / f"{snapshot_id}.json"
        snapshot_path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        record = self._storage.create_dataset_snapshot(
            snapshot_id=snapshot_id,
            snapshot_name=snapshot_name,
            document_relpath=self._relative_path(snapshot_path),
            task_count=task_count,
            annotation_count=annotation_count,
            created_by=created_by,
            metadata={
                **payload.metadata,
                "task_ids": list(payload.task_ids or []),
                "class_audit": document.get("metadata", {}).get("class_audit", {}),
            },
        )
        return self._snapshot_to_response(record)

    def list_snapshots(self, *, limit: int = 100) -> list[DatasetSnapshotResponse]:
        return [self._snapshot_to_response(item) for item in self._storage.list_dataset_snapshots(limit=limit)]

    def _resolve_snapshot_for_job(self, payload: TrainingJobCreateRequest, *, requested_by: str) -> DatasetSnapshotRecord:
        if payload.snapshot_id:
            snapshot = self._storage.get_dataset_snapshot(payload.snapshot_id)
            if snapshot is None:
                raise ValueError("snapshot not found")
            return snapshot
        snapshot_response = self.create_snapshot(
            payload=DatasetSnapshotCreateRequest(
                snapshot_name=payload.snapshot_name,
                task_ids=payload.snapshot_task_ids,
                metadata=payload.snapshot_metadata,
            ),
            created_by=requested_by,
        )
        snapshot = self._storage.get_dataset_snapshot(snapshot_response.snapshot_id)
        if snapshot is None:
            raise RuntimeError("failed to resolve created snapshot")
        return snapshot

    def create_training_job(
        self,
        *,
        payload: TrainingJobCreateRequest,
        requested_by: str,
    ) -> TrainingJobResponse:
        snapshot = self._resolve_snapshot_for_job(payload, requested_by=requested_by)
        normalized_task_type = "classification"
        if payload.task_type != "classification":
            logger.warning(
                "训练链路已统一为11类细分类，忽略 task_type=%s，强制使用 classification",
                payload.task_type,
            )
        train_config = dict(payload.train_config)
        train_config.update(merge_imbalance_config(payload.train_config))
        if isinstance(snapshot.metadata, dict) and isinstance(snapshot.metadata.get("class_audit"), dict):
            class_audit = snapshot.metadata["class_audit"]
            train_config["class_audit"] = class_audit
            train_config["promotion_warnings"] = class_audit.get("promotion_warnings", [])
        train_config["snapshot_document_relpath"] = snapshot.document_relpath
        record = self._storage.enqueue_training_job(
            snapshot_id=snapshot.snapshot_id,
            requested_by=requested_by,
            task_type=normalized_task_type,
            model_version=payload.model_version,
            model_id=self._normalize_model_id(payload.model_id, model_version=payload.model_version),
            model_backbone=payload.model_backbone,
            train_config=train_config,
            priority=payload.priority,
            promote_on_success=payload.promote_on_success,
            enqueue_prelabels_on_success=payload.enqueue_prelabels_on_success,
            prelabel_task_ids=payload.prelabel_task_ids,
            force_prelabel=payload.force_prelabel,
        )
        return self._job_to_response(record)

    def list_training_jobs(self, *, limit: int = 100) -> list[TrainingJobResponse]:
        return [self._job_to_response(item) for item in self._storage.list_training_jobs(limit=limit)]

    def list_runs(self, *, limit: int = 100) -> list[TrainingRunResponse]:
        return [self._run_to_response(item) for item in self._storage.list_training_runs(limit=limit)]

    def list_models(self, *, task_type: str | None = None, limit: int = 100) -> list[RegisteredModelResponse]:
        return [
            self._model_to_response(item)
            for item in self._storage.list_registered_models(task_type=task_type, limit=limit)
        ]

    def get_promoted_model(self, *, task_type: str) -> RegisteredModelResponse | None:
        model = self._storage.get_promoted_model(task_type=task_type)
        return self._model_to_response(model) if model is not None else None

    def promote_model(
        self,
        *,
        model_id: str,
        promoted_by: str,
        enqueue_prelabels: bool = False,
        force_prelabel: bool = False,
        prelabel_task_ids: list[str] | None = None,
    ) -> PromoteModelResponse | None:
        promoted = self._storage.promote_registered_model(model_id=model_id, promoted_by=promoted_by)
        if promoted is None:
            return None
        promotion_warnings = self._promotion_warnings_from_model(promoted)
        prelabel_enqueue = None
        if enqueue_prelabels:
            result = self._prelabel_service.enqueue(
                payload=PrelabelEnqueueRequest(
                    model_version=promoted.model_version,
                    model_id=promoted.model_id,
                    model_backbone=promoted.model_backbone,
                    task_ids=list(prelabel_task_ids or []),
                    force=force_prelabel,
                ),
                requested_by=promoted_by,
            )
            prelabel_enqueue = result.model_dump()
        return PromoteModelResponse(
            model=self._model_to_response(promoted),
            prelabel_enqueue=prelabel_enqueue,
            promotion_warnings=promotion_warnings,
        )

    def claim_next_job(self, payload: TrainingWorkerClaimRequest) -> TrainingWorkerClaimResponse | None:
        task_types_raw = payload.capabilities.get("task_types")
        task_types = [str(item) for item in task_types_raw if str(item).strip()] if isinstance(task_types_raw, list) else None
        model_backbones_raw = payload.capabilities.get("model_backbones")
        model_backbones = (
            [str(item) for item in model_backbones_raw if str(item).strip()]
            if isinstance(model_backbones_raw, list)
            else None
        )
        self._storage.upsert_worker_node(
            worker_id=payload.worker_id,
            display_name=payload.display_name,
            host_name=payload.host_name,
            device_label=payload.device_label,
            capabilities=payload.capabilities,
            status="online",
        )
        job = self._storage.claim_next_training_job(
            worker_id=payload.worker_id,
            timeout_seconds=self.job_timeout_seconds,
            task_types=task_types,
            model_backbones=model_backbones,
        )
        if job is None:
            return None
        return TrainingWorkerClaimResponse(
            job_id=job.job_id,
            snapshot_id=job.snapshot_id,
            task_type=job.task_type,
            model_version=job.model_version,
            model_id=job.model_id,
            model_backbone=job.model_backbone,
            train_config=job.train_config or {},
            promote_on_success=job.promote_on_success,
            enqueue_prelabels_on_success=job.enqueue_prelabels_on_success,
            prelabel_task_ids=list(job.prelabel_task_ids or []),
            force_prelabel=job.force_prelabel,
            claimed_at=job.claimed_at,
            claim_expires_at=job.claim_expires_at,
        )

    def heartbeat_job(self, *, job_id: str, payload: TrainingWorkerHeartbeatRequest) -> TrainingWorkerJobAckResponse:
        accepted = self._storage.heartbeat_training_job(
            job_id=job_id,
            worker_id=payload.worker_id,
            timeout_seconds=self.job_timeout_seconds,
        )
        return TrainingWorkerJobAckResponse(job_id=job_id, accepted=accepted)

    def get_claimed_job_snapshot_path(self, *, job_id: str, worker_id: str) -> Path:
        job = self._storage.get_training_job(job_id)
        if job is None:
            raise ValueError("job not found")
        if job.status != "claimed" or job.claim_worker_id != worker_id:
            raise ValueError("job is not claimed by this worker")
        snapshot = self._storage.get_dataset_snapshot(job.snapshot_id)
        if snapshot is None:
            raise ValueError("snapshot not found")
        file_path = (self.dataset_root / snapshot.document_relpath).resolve()
        try:
            file_path.relative_to(self.dataset_root)
        except ValueError as exc:
            raise ValueError("snapshot path is invalid") from exc
        if not file_path.is_file():
            raise ValueError("snapshot file does not exist")
        return file_path

    def store_uploaded_model_artifact(
        self,
        *,
        job_id: str,
        worker_id: str,
        filename: str,
        content: bytes,
    ) -> TrainingArtifactUploadResponse:
        job = self._storage.get_training_job(job_id)
        if job is None:
            raise ValueError("job not found")
        if job.status != "claimed" or job.claim_worker_id != worker_id:
            raise ValueError("job is not claimed by this worker")
        safe_name = _SAFE_FILENAME_RE.sub("-", Path(filename or f"{job.model_id}.pt").name).strip("-.")
        if not safe_name:
            safe_name = f"{job.model_id}.pt"
        artifact_dir = self._model_root / job.model_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / safe_name
        artifact_path.write_bytes(content)
        return TrainingArtifactUploadResponse(
            job_id=job_id,
            artifact_path=self._relative_path(artifact_path),
        )

    def complete_job(self, *, job_id: str, payload: TrainingWorkerCompleteRequest) -> TrainingJobLifecycleResponse | None:
        pre_job = self._storage.get_training_job(job_id)
        snapshot = self._storage.get_dataset_snapshot(pre_job.snapshot_id) if pre_job is not None else None
        metrics = dict(payload.metrics or {})
        metadata = dict(payload.metadata or {})
        promotion_warnings = self._promotion_warnings_for_completed_job(
            job=pre_job,
            snapshot=snapshot,
            metrics=metrics,
            metadata=metadata,
        ) if pre_job is not None else []
        if promotion_warnings:
            metrics["promotion_warnings"] = promotion_warnings
            metadata["promotion_warnings"] = promotion_warnings
        completed = self._storage.complete_training_job(
            job_id=job_id,
            worker_id=payload.worker_id,
            artifact_path=payload.artifact_path,
            metrics=metrics,
            metadata=metadata,
        )
        if completed is None:
            return None
        job, run, model = completed
        prelabel_enqueue = None
        promoted = model
        auto_promoted = False
        if job.promote_on_success and not promotion_warnings:
            promoted_model = self._storage.promote_registered_model(model_id=model.model_id, promoted_by=job.requested_by)
            if promoted_model is not None:
                promoted = promoted_model
                auto_promoted = True
        if job.enqueue_prelabels_on_success and not promotion_warnings:
            result = self._prelabel_service.enqueue(
                payload=PrelabelEnqueueRequest(
                    model_version=job.model_version,
                    model_id=job.model_id,
                    model_backbone=job.model_backbone,
                    task_ids=list(job.prelabel_task_ids or []),
                    force=job.force_prelabel,
                ),
                requested_by=job.requested_by,
            )
            prelabel_enqueue = result.model_dump()
        return TrainingJobLifecycleResponse(
            job=self._job_to_response(job),
            run=self._run_to_response(run),
            model=self._model_to_response(promoted),
            prelabel_enqueue=prelabel_enqueue,
            promotion_warnings=promotion_warnings,
            auto_promoted=auto_promoted,
        )

    def fail_job(self, *, job_id: str, payload: TrainingWorkerFailRequest) -> TrainingWorkerJobAckResponse:
        accepted = self._storage.fail_training_job(
            job_id=job_id,
            worker_id=payload.worker_id,
            error_message=payload.error_message,
            retryable=payload.retryable,
        )
        return TrainingWorkerJobAckResponse(job_id=job_id, accepted=accepted)

    def get_model_artifact_path(self, model_id: str) -> Path:
        model = self._storage.get_registered_model(model_id)
        if model is None:
            raise ValueError("model not found")
        artifact_path = str(model.artifact_path or "").strip()
        if not artifact_path:
            raise ValueError("model artifact path is missing")
        file_path = (self.dataset_root / artifact_path).resolve()
        try:
            file_path.relative_to(self.dataset_root)
        except ValueError as exc:
            raise ValueError("model artifact path is invalid") from exc
        if not file_path.is_file():
            raise ValueError("model artifact file does not exist")
        return file_path
