from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from scann.core.dataset_storage import (
    DatasetStorage,
    PrelabelJobRecord,
    TaskAIPrelabelRecord,
    TaskPrelabelSummaryRecord,
)

from .dataset_service import DatasetService, TaskSession


DEFAULT_PRELABEL_JOB_TIMEOUT_SECONDS = 15 * 60


class PrelabelBox(BaseModel):
    x: float = Field(..., ge=0)
    y: float = Field(..., ge=0)
    width: float = Field(..., ge=0)
    height: float = Field(..., ge=0)
    label: Optional[str] = None
    detail_type: Optional[str] = None
    confidence: Optional[float] = Field(default=1.0, ge=0)


class PrelabelEnqueueRequest(BaseModel):
    model_version: str = Field(..., min_length=1)
    model_id: Optional[str] = None
    model_backbone: Optional[str] = None
    task_ids: list[str] = Field(default_factory=list)
    priority: int = 100
    force: bool = False


class PrelabelEnqueueResponse(BaseModel):
    requested_count: int
    enqueued_count: int
    skipped_count: int
    job_ids: list[str] = Field(default_factory=list)
    skipped_task_ids: list[str] = Field(default_factory=list)


class TaskPrelabelResponse(BaseModel):
    prelabel_id: str
    task_id: str
    job_id: Optional[str] = None
    status: str
    source_view: Optional[Literal["old", "new", "new_marked"]] = "new"
    ai_suggestion: Optional[str] = None
    ai_confidence: Optional[float] = None
    model_version: Optional[str] = None
    model_id: Optional[str] = None
    model_backbone: Optional[str] = None
    input_fingerprint: Optional[str] = None
    box_count: int
    worker_id: Optional[str] = None
    accepted_revision_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    annotations: list[PrelabelBox] = Field(default_factory=list)


class WorkerClaimRequest(BaseModel):
    worker_id: str = Field(..., min_length=1)
    display_name: Optional[str] = None
    host_name: Optional[str] = None
    device_label: Optional[str] = None
    capabilities: dict[str, Any] = Field(default_factory=dict)


class WorkerClaimResponse(BaseModel):
    job_id: str
    task_id: str
    model_version: str
    model_id: Optional[str] = None
    model_backbone: Optional[str] = None
    input_fingerprint: str
    paths: dict[str, str] = Field(default_factory=dict)
    claimed_at: Optional[str] = None
    claim_expires_at: Optional[str] = None


class WorkerHeartbeatRequest(BaseModel):
    worker_id: str = Field(..., min_length=1)


class WorkerJobAckResponse(BaseModel):
    job_id: str
    accepted: bool


class WorkerCompleteRequest(BaseModel):
    worker_id: str = Field(..., min_length=1)
    source_view: Optional[Literal["old", "new", "new_marked"]] = "new"
    ai_suggestion: Optional[str] = None
    ai_confidence: Optional[float] = None
    annotations: list[PrelabelBox] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkerFailRequest(BaseModel):
    worker_id: str = Field(..., min_length=1)
    error_message: str = Field(..., min_length=1)
    retryable: bool = False


class PrelabelService:
    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self._storage = DatasetStorage(self.dataset_root)
        self._storage.ensure_schema()
        self._dataset_service = DatasetService(dataset_root=self.dataset_root)
        self.job_timeout_seconds = max(
            30,
            int(os.getenv("SCANN_PRELABEL_JOB_TIMEOUT_SECONDS", str(DEFAULT_PRELABEL_JOB_TIMEOUT_SECONDS))),
        )

    def _task_sessions_by_id(self) -> dict[str, TaskSession]:
        return {task.task_id: task for task in self._dataset_service.list_tasks()}

    def _build_input_fingerprint(
        self,
        task: TaskSession,
        *,
        model_version: str,
        model_id: str | None = None,
        model_backbone: str | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "task_id": task.task_id,
            "paths": {
                "new": task.new_path,
                "old": task.old_path,
                "new_marked": task.new_marked_path,
            },
            "files": [],
        }
        for relpath in [task.new_path, task.old_path, task.new_marked_path]:
            if not relpath:
                continue
            file_path = self.dataset_root / relpath
            if not file_path.exists():
                continue
            stat = file_path.stat()
            payload["files"].append(
                {
                    "relpath": relpath,
                    "size_bytes": int(stat.st_size),
                    "modified_ns": int(stat.st_mtime_ns),
                }
            )
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _request_matches_existing_model(
        existing: TaskAIPrelabelRecord,
        *,
        model_version: str,
        model_id: str | None,
        model_backbone: str | None,
    ) -> bool:
        if existing.model_version != model_version:
            return False
        normalized_model_id = str(model_id or "").strip()
        if normalized_model_id and (existing.model_id or "").strip() != normalized_model_id:
            return False
        normalized_model_backbone = str(model_backbone or "").strip()
        if normalized_model_backbone and (existing.model_backbone or "").strip() != normalized_model_backbone:
            return False
        return True

    @staticmethod
    def _record_to_response(
        record: TaskAIPrelabelRecord,
        annotations: list[dict[str, Any]],
    ) -> TaskPrelabelResponse:
        return TaskPrelabelResponse(
            prelabel_id=record.prelabel_id,
            task_id=record.task_id,
            job_id=record.job_id,
            status=record.status,
            source_view=record.source_view or "new",
            ai_suggestion=record.ai_suggestion,
            ai_confidence=record.ai_confidence,
            model_version=record.model_version,
            model_id=record.model_id,
            model_backbone=record.model_backbone,
            input_fingerprint=record.input_fingerprint,
            box_count=record.box_count,
            worker_id=record.worker_id,
            accepted_revision_id=record.accepted_revision_id,
            metadata=record.metadata or {},
            created_at=record.created_at,
            updated_at=record.updated_at,
            annotations=[PrelabelBox.model_validate(item) for item in annotations if isinstance(item, dict)],
        )

    def enqueue(
        self,
        *,
        payload: PrelabelEnqueueRequest,
        requested_by: str,
    ) -> PrelabelEnqueueResponse:
        tasks_by_id = self._task_sessions_by_id()
        requested_ids = payload.task_ids or list(tasks_by_id.keys())

        job_ids: list[str] = []
        skipped_task_ids: list[str] = []
        enqueued_count = 0
        skipped_count = 0

        for task_id in requested_ids:
            task = tasks_by_id.get(task_id)
            if task is None:
                skipped_task_ids.append(task_id)
                skipped_count += 1
                continue

            normalized_model_id = str(payload.model_id or "").strip() or None
            normalized_model_backbone = str(payload.model_backbone or "").strip() or None
            input_fingerprint = self._build_input_fingerprint(
                task,
                model_version=payload.model_version,
                model_id=normalized_model_id,
                model_backbone=normalized_model_backbone,
            )
            existing = self._storage.get_latest_task_prelabel_record(
                task.task_id,
                statuses=("available", "accepted"),
            )
            if (
                existing is not None
                and not payload.force
                and self._request_matches_existing_model(
                    existing,
                    model_version=payload.model_version,
                    model_id=normalized_model_id,
                    model_backbone=normalized_model_backbone,
                )
                and existing.input_fingerprint == input_fingerprint
            ):
                skipped_task_ids.append(task.task_id)
                skipped_count += 1
                continue

            job, created = self._storage.enqueue_prelabel_job(
                task_id=task.task_id,
                requested_by=requested_by,
                model_version=payload.model_version,
                model_id=normalized_model_id,
                model_backbone=normalized_model_backbone,
                input_fingerprint=input_fingerprint,
                priority=payload.priority,
                cancel_existing=payload.force,
            )
            if created:
                enqueued_count += 1
                job_ids.append(job.job_id)
            else:
                skipped_count += 1
                skipped_task_ids.append(task.task_id)

        return PrelabelEnqueueResponse(
            requested_count=len(requested_ids),
            enqueued_count=enqueued_count,
            skipped_count=skipped_count,
            job_ids=job_ids,
            skipped_task_ids=skipped_task_ids,
        )

    def get_task_prelabel(self, task_id: str) -> TaskPrelabelResponse | None:
        result = self._storage.get_task_prelabel(task_id)
        if result is None:
            return None
        record, annotations = result
        return self._record_to_response(record, annotations)

    def list_task_prelabel_summaries(self, task_ids: list[str]) -> dict[str, TaskPrelabelSummaryRecord]:
        return self._storage.list_task_prelabel_summaries(task_ids)

    @staticmethod
    def _paths_for_task(task: TaskSession) -> dict[str, str]:
        paths = {"new": task.new_path}
        if task.old_path:
            paths["old"] = task.old_path
        if task.new_marked_path:
            paths["new_marked"] = task.new_marked_path
        return paths

    def claim_next_job(self, payload: WorkerClaimRequest) -> WorkerClaimResponse | None:
        model_versions_raw = payload.capabilities.get("model_versions")
        model_versions = (
            [str(item) for item in model_versions_raw if str(item).strip()]
            if isinstance(model_versions_raw, list)
            else None
        )
        model_ids_raw = payload.capabilities.get("model_ids")
        model_ids = (
            [str(item) for item in model_ids_raw if str(item).strip()]
            if isinstance(model_ids_raw, list)
            else None
        )
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
        job = self._storage.claim_next_prelabel_job(
            worker_id=payload.worker_id,
            timeout_seconds=self.job_timeout_seconds,
            model_versions=model_versions,
            model_ids=model_ids,
            model_backbones=model_backbones,
        )
        if job is None:
            return None

        task = self._task_sessions_by_id().get(job.task_id)
        if task is None:
            self._storage.fail_prelabel_job(
                job_id=job.job_id,
                worker_id=payload.worker_id,
                error_message="prepared task paths are missing",
                retryable=False,
            )
            return None

        return WorkerClaimResponse(
            job_id=job.job_id,
            task_id=job.task_id,
            model_version=job.model_version,
            model_id=job.model_id,
            model_backbone=job.model_backbone,
            input_fingerprint=job.input_fingerprint,
            paths=self._paths_for_task(task),
            claimed_at=job.claimed_at,
            claim_expires_at=job.claim_expires_at,
        )

    def get_claimed_job_asset_path(
        self,
        *,
        job_id: str,
        worker_id: str,
        view_name: str,
    ) -> Path:
        job = self._storage.get_prelabel_job(job_id)
        if job is None:
            raise ValueError("job not found")
        if job.status != "claimed" or job.claim_worker_id != worker_id:
            raise ValueError("job is not claimed by this worker")

        task = self._task_sessions_by_id().get(job.task_id)
        if task is None:
            raise ValueError("task not found")

        relpath = {
            "new": task.new_path,
            "old": task.old_path,
            "new_marked": task.new_marked_path,
        }.get(view_name)
        if not relpath:
            raise ValueError("requested asset is not available")

        file_path = (self.dataset_root / relpath).resolve()
        file_path.relative_to(self.dataset_root)
        if not file_path.is_file():
            raise ValueError("requested asset file does not exist")
        return file_path

    def heartbeat_job(self, *, job_id: str, payload: WorkerHeartbeatRequest) -> WorkerJobAckResponse:
        accepted = self._storage.heartbeat_prelabel_job(
            job_id=job_id,
            worker_id=payload.worker_id,
            timeout_seconds=self.job_timeout_seconds,
        )
        return WorkerJobAckResponse(job_id=job_id, accepted=accepted)

    def complete_job(self, *, job_id: str, payload: WorkerCompleteRequest) -> TaskPrelabelResponse | None:
        record = self._storage.complete_prelabel_job(
            job_id=job_id,
            worker_id=payload.worker_id,
            source_view=payload.source_view,
            ai_suggestion=payload.ai_suggestion,
            ai_confidence=payload.ai_confidence,
            annotations=[item.model_dump(exclude_none=True) for item in payload.annotations],
            metadata=payload.metadata,
        )
        if record is None:
            return None
        result = self._storage.get_task_prelabel(record.task_id)
        if result is None:
            return None
        latest_record, annotations = result
        return self._record_to_response(latest_record, annotations)

    def fail_job(self, *, job_id: str, payload: WorkerFailRequest) -> WorkerJobAckResponse:
        accepted = self._storage.fail_prelabel_job(
            job_id=job_id,
            worker_id=payload.worker_id,
            error_message=payload.error_message,
            retryable=payload.retryable,
        )
        return WorkerJobAckResponse(job_id=job_id, accepted=accepted)
