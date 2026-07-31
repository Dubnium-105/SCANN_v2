from __future__ import annotations

import hmac
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel

from .annotation_service import (
    AnnotationHistoryResponse,
    AnnotationRevisionDetail,
    AnnotationRollbackResponse,
    AnnotationSaveRequest,
    AnnotationSaveResponse,
    AnnotationService,
)
from .annotation_sync_service import (
    AnnotationSyncResult,
    AnnotationSyncStatus,
    build_annotation_sync_service_from_env,
)
from .auth_service import (
    AuthUser,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    authenticate_user,
    create_access_token,
    get_current_user,
    register_user,
)
from .dataset_service import DatasetService, TaskSession
from .fits_engine import FITSEngine
from .prelabel_service import (
    PrelabelEnqueueRequest,
    PrelabelEnqueueResponse,
    PrelabelJobsCancelRequest,
    PrelabelJobsCancelResponse,
    PrelabelJobResponse,
    PrelabelService,
    PrelabelWorkerResponse,
    TaskPrelabelResponse,
    WorkerClaimRequest,
    WorkerClaimResponse,
    WorkerCompleteRequest,
    WorkerFailRequest,
    WorkerHeartbeatRequest,
    WorkerJobAckResponse,
)
from .task_lock_service import TaskLockService
from .training_lifecycle_service import (
    DatasetPartitionCreateRequest,
    DatasetPartitionResponse,
    DatasetSnapshotCreateRequest,
    DatasetSnapshotResponse,
    PromoteModelResponse,
    RegisteredModelResponse,
    TrainingArtifactUploadResponse,
    TrainingJobCreateRequest,
    TrainingJobLifecycleResponse,
    TrainingJobResponse,
    TrainingLifecycleService,
    TrainingRunResponse,
    TrainingWorkerClaimRequest,
    TrainingWorkerClaimResponse,
    TrainingWorkerCompleteRequest,
    TrainingWorkerFailRequest,
    TrainingWorkerHeartbeatRequest,
    TrainingWorkerJobAckResponse,
)
from scann.core.dataset_storage import DatasetStorage
from scann.services.dataset_preprocess_service import DatasetPreprocessService

api_router = APIRouter(prefix="/api", tags=["api"])
_fits_engines: Dict[str, FITSEngine] = {}
_task_lock_services: Dict[str, TaskLockService] = {}
_DATASET_STATS_CACHE_TTL_SECONDS = 30.0
_dataset_stats_cache: Dict[str, Any] = {
    "dataset_root": "",
    "expires_at": 0.0,
    "payload": None,
}


class TaskClaimResponse(TaskSession):
    client_id: str
    lock_expires_at: str


class TaskListResponse(TaskSession):
    lock_expires_at: Optional[str] = None
    locked_by_current_client: Optional[bool] = None
    prelabel_status: Optional[str] = None
    prelabel_model_version: Optional[str] = None
    prelabel_model_id: Optional[str] = None
    prelabel_model_backbone: Optional[str] = None
    prelabel_updated_at: Optional[str] = None
    prelabel_box_count: Optional[int] = None


class TaskLockHeartbeatResponse(BaseModel):
    task_id: str
    client_id: str
    lock_expires_at: str


class TaskReleaseResponse(BaseModel):
    task_id: str
    client_id: str
    released: bool


class DatasetPreprocessResponse(BaseModel):
    dataset_root: str
    standardized_files: int
    reused_aligned_pairs: int
    generated_aligned_pairs: int
    generated_marked_crops: int
    task_count: int
    total_task_count: int = 0
    align_failed_count: int = 0


def get_dataset_root() -> Path:
    return Path(os.getenv("SCANN_NATIVE_DATASET_ROOT", "dataset")).resolve()


def get_dataset_service() -> DatasetService:
    return DatasetService(dataset_root=get_dataset_root())


def get_dataset_storage() -> DatasetStorage:
    return DatasetStorage(dataset_root=get_dataset_root())


def get_dataset_preprocess_service() -> DatasetPreprocessService:
    return DatasetPreprocessService()


def get_fits_engine() -> FITSEngine:
    dataset_root = get_dataset_root()
    key = str(dataset_root)
    engine = _fits_engines.get(key)
    if engine is None:
        engine = FITSEngine(dataset_root=dataset_root)
        _fits_engines[key] = engine
    return engine


def get_task_lock_service() -> TaskLockService:
    dataset_root = get_dataset_root()
    key = str(dataset_root)
    lock_service = _task_lock_services.get(key)
    if lock_service is None:
        timeout_seconds = int(os.getenv("SCANN_NATIVE_TASK_LOCK_TIMEOUT_SECONDS", "1200"))
        lock_service = TaskLockService(
            lock_timeout_seconds=timeout_seconds,
            dataset_root=dataset_root,
        )
        _task_lock_services[key] = lock_service
    return lock_service


def get_annotation_service() -> AnnotationService:
    return AnnotationService(dataset_root=get_dataset_root())


def get_annotation_sync_service():
    return build_annotation_sync_service_from_env(get_dataset_root())


def invalidate_dataset_stats_cache() -> None:
    _dataset_stats_cache.update({"dataset_root": "", "expires_at": 0.0, "payload": None})


def load_dataset_stats_cached(fresh: bool = False) -> dict:
    dataset_root = str(get_dataset_root())
    now = time.monotonic()
    cached_payload = _dataset_stats_cache.get("payload")
    if (
        not fresh
        and cached_payload is not None
        and _dataset_stats_cache.get("dataset_root") == dataset_root
        and float(_dataset_stats_cache.get("expires_at") or 0.0) > now
    ):
        return cached_payload

    storage = get_dataset_storage()
    payload = storage.get_dataset_statistics()
    _dataset_stats_cache.update(
        {
            "dataset_root": dataset_root,
            "expires_at": now + _DATASET_STATS_CACHE_TTL_SECONDS,
            "payload": payload,
        }
    )
    return payload


def get_prelabel_service() -> PrelabelService:
    return PrelabelService(dataset_root=get_dataset_root())


def get_training_service() -> TrainingLifecycleService:
    return TrainingLifecycleService(dataset_root=get_dataset_root())


def require_prelabel_worker_token(
    x_scann_worker_token: Optional[str] = Header(None),
) -> str:
    expected = os.getenv("SCANN_PRELABEL_WORKER_TOKEN", "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="Prelabel worker token is not configured")
    presented = (x_scann_worker_token or "").strip()
    if not presented or not hmac.compare_digest(presented, expected):
        raise HTTPException(status_code=401, detail="Invalid worker token")
    return presented


def require_training_worker_token(
    x_scann_worker_token: Optional[str] = Header(None),
) -> str:
    expected = os.getenv("SCANN_TRAINING_WORKER_TOKEN", "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="Training worker token is not configured")
    presented = (x_scann_worker_token or "").strip()
    if not presented or not hmac.compare_digest(presented, expected):
        raise HTTPException(status_code=401, detail="Invalid worker token")
    return presented


def _require_task_lock_owner(
    task_id: str,
    client_id: str,
    lock_service: TaskLockService,
) -> str:
    normalized_client_id = client_id.strip()
    if not normalized_client_id:
        raise HTTPException(status_code=400, detail="client_id cannot be empty")

    lock = lock_service.get_task_lock(task_id)
    if lock is None:
        raise HTTPException(status_code=404, detail="Task lock not found")
    if lock.client_id != normalized_client_id:
        raise HTTPException(status_code=409, detail="Task locked by another client")
    return normalized_client_id


def _require_task_lock_for_save(
    task_id: str,
    client_id: Optional[str],
    lock_service: TaskLockService,
) -> Optional[str]:
    lock = lock_service.get_task_lock(task_id)
    if lock is None:
        if client_id is None:
            return None
        normalized_client_id = client_id.strip()
        return normalized_client_id or None

    if client_id is None or not client_id.strip():
        raise HTTPException(status_code=409, detail="Task locked by another client")
    return _require_task_lock_owner(task_id=task_id, client_id=client_id, lock_service=lock_service)


@api_router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@api_router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest) -> TokenResponse:
    user = authenticate_user(payload.username, payload.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_access_token(user)
    return TokenResponse(
        access_token=token,
        username=user.username,
        role=user.role,
    )


@api_router.post("/register", response_model=TokenResponse)
def register(payload: RegisterRequest) -> TokenResponse:
    try:
        user = register_user(payload.username, payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    token = create_access_token(user)
    return TokenResponse(
        access_token=token,
        username=user.username,
        role=user.role,
    )


@api_router.get("/tasks", response_model=list[TaskListResponse], response_model_exclude_none=True)
def list_tasks(
    client_id: Optional[str] = Query(None),
    current_user: AuthUser = Depends(get_current_user),
) -> list[TaskListResponse]:
    _ = current_user
    service = get_dataset_service()
    prelabel_service = get_prelabel_service()
    lock_service = get_task_lock_service()
    normalized_client_id = client_id.strip() if client_id and client_id.strip() else None
    responses: list[TaskListResponse] = []
    tasks = service.list_tasks()
    locks_by_task = lock_service.get_task_locks([task.task_id for task in tasks])
    prelabel_summaries = prelabel_service.list_task_prelabel_summaries([task.task_id for task in tasks])
    for task in tasks:
        task_payload = task.model_dump()
        lock = locks_by_task.get(task.task_id)
        if lock is not None:
            task_payload["lock_expires_at"] = lock.expires_at.isoformat(timespec="seconds")
            if normalized_client_id is not None:
                task_payload["locked_by_current_client"] = lock.client_id == normalized_client_id
        summary = prelabel_summaries.get(task.task_id)
        if summary is not None:
            task_payload["prelabel_status"] = summary.prelabel_status
            task_payload["prelabel_model_version"] = summary.prelabel_model_version
            task_payload["prelabel_model_id"] = summary.prelabel_model_id
            task_payload["prelabel_model_backbone"] = summary.prelabel_model_backbone
            task_payload["prelabel_updated_at"] = summary.prelabel_updated_at
            task_payload["prelabel_box_count"] = summary.prelabel_box_count
        responses.append(TaskListResponse(**task_payload))
    return responses


@api_router.post("/dataset/preprocess", response_model=DatasetPreprocessResponse)
def preprocess_dataset(
    current_user: AuthUser = Depends(get_current_user),
) -> DatasetPreprocessResponse:
    _ = current_user
    dataset_root = get_dataset_root()
    report = get_dataset_preprocess_service().prepare_dataset(dataset_root)
    return DatasetPreprocessResponse(
        dataset_root=str(dataset_root),
        standardized_files=report.standardized_files,
        reused_aligned_pairs=report.reused_aligned_pairs,
        generated_aligned_pairs=report.generated_aligned_pairs,
        generated_marked_crops=report.generated_marked_crops,
        task_count=report.task_count,
        total_task_count=report.total_task_count,
        align_failed_count=report.align_failed_count,
    )


@api_router.post("/prelabels/enqueue", response_model=PrelabelEnqueueResponse)
def enqueue_prelabels(
    payload: PrelabelEnqueueRequest,
    current_user: AuthUser = Depends(get_current_user),
) -> PrelabelEnqueueResponse:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admin can enqueue prelabels")
    service = get_prelabel_service()
    return service.enqueue(payload=payload, requested_by=current_user.username)


@api_router.get("/prelabels/jobs", response_model=list[PrelabelJobResponse])
def list_prelabel_jobs(
    limit: int = Query(100, ge=1, le=500),
    statuses: Optional[str] = Query(None),
    task_ids: Optional[str] = Query(None),
    current_user: AuthUser = Depends(get_current_user),
) -> list[PrelabelJobResponse]:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admin can list prelabel jobs")
    parsed_statuses = [item.strip() for item in str(statuses or "").split(",") if item and item.strip()]
    parsed_task_ids = [item.strip() for item in str(task_ids or "").split(",") if item and item.strip()]
    return get_prelabel_service().list_jobs(
        limit=limit,
        statuses=parsed_statuses or None,
        task_ids=parsed_task_ids or None,
    )


@api_router.get("/prelabels/workers", response_model=list[PrelabelWorkerResponse])
def list_prelabel_workers(
    limit: int = Query(100, ge=1, le=500),
    current_user: AuthUser = Depends(get_current_user),
) -> list[PrelabelWorkerResponse]:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admin can list prelabel workers")
    return get_prelabel_service().list_workers(limit=limit)


@api_router.post("/prelabels/jobs/cancel", response_model=PrelabelJobsCancelResponse)
def cancel_prelabel_jobs(
    payload: PrelabelJobsCancelRequest,
    current_user: AuthUser = Depends(get_current_user),
) -> PrelabelJobsCancelResponse:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admin can cancel prelabel jobs")
    return get_prelabel_service().cancel_jobs(payload=payload, cancelled_by=current_user.username)


@api_router.get("/prelabels/{task_id}", response_model=TaskPrelabelResponse, response_model_exclude_none=True)
def get_task_prelabel(
    task_id: str,
    current_user: AuthUser = Depends(get_current_user),
) -> TaskPrelabelResponse:
    _ = current_user
    service = get_prelabel_service()
    response = service.get_task_prelabel(task_id)
    if response is None:
        raise HTTPException(status_code=404, detail="Prelabel not found")
    return response


@api_router.post("/prelabel-jobs/claim", response_model=WorkerClaimResponse)
def claim_prelabel_job(
    payload: WorkerClaimRequest,
    _worker_token: str = Depends(require_prelabel_worker_token),
) -> WorkerClaimResponse:
    service = get_prelabel_service()
    response = service.claim_next_job(payload)
    if response is None:
        raise HTTPException(status_code=404, detail=service.explain_claim_miss(payload))
    return response


@api_router.post("/prelabel-jobs/{job_id}/heartbeat", response_model=WorkerJobAckResponse)
def heartbeat_prelabel_job(
    job_id: str,
    payload: WorkerHeartbeatRequest,
    _worker_token: str = Depends(require_prelabel_worker_token),
) -> WorkerJobAckResponse:
    service = get_prelabel_service()
    response = service.heartbeat_job(job_id=job_id, payload=payload)
    if not response.accepted:
        raise HTTPException(status_code=409, detail="Prelabel job is not claimed by this worker")
    return response


@api_router.post("/prelabel-jobs/{job_id}/complete", response_model=TaskPrelabelResponse)
def complete_prelabel_job(
    job_id: str,
    payload: WorkerCompleteRequest,
    _worker_token: str = Depends(require_prelabel_worker_token),
) -> TaskPrelabelResponse:
    service = get_prelabel_service()
    response = service.complete_job(job_id=job_id, payload=payload)
    if response is None:
        raise HTTPException(status_code=409, detail="Prelabel job is not claimed by this worker")
    return response


@api_router.post("/prelabel-jobs/{job_id}/fail", response_model=WorkerJobAckResponse)
def fail_prelabel_job(
    job_id: str,
    payload: WorkerFailRequest,
    _worker_token: str = Depends(require_prelabel_worker_token),
) -> WorkerJobAckResponse:
    service = get_prelabel_service()
    response = service.fail_job(job_id=job_id, payload=payload)
    if not response.accepted:
        raise HTTPException(status_code=409, detail="Prelabel job is not claimed by this worker")
    return response


@api_router.get("/prelabel-jobs/{job_id}/fits/{view_name}")
def fetch_prelabel_job_fits(
    job_id: str,
    view_name: str,
    worker_id: str = Query(..., min_length=1),
    _worker_token: str = Depends(require_prelabel_worker_token),
) -> Response:
    service = get_prelabel_service()
    try:
        file_path = service.get_claimed_job_asset_path(
            job_id=job_id,
            worker_id=worker_id,
            view_name=view_name,
        )
    except ValueError as exc:
        message = str(exc)
        if message in {"job not found", "task not found"}:
            raise HTTPException(status_code=404, detail=message) from exc
        raise HTTPException(status_code=409, detail=message) from exc
    return Response(content=file_path.read_bytes(), media_type="application/octet-stream")


@api_router.post("/dataset-partitions", response_model=DatasetPartitionResponse)
def create_dataset_partition(
    payload: DatasetPartitionCreateRequest,
    current_user: AuthUser = Depends(get_current_user),
) -> DatasetPartitionResponse:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admin can create dataset partitions")
    service = get_training_service()
    try:
        return service.create_partition(
            payload=payload,
            created_by=current_user.username,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@api_router.get("/dataset-partitions", response_model=list[DatasetPartitionResponse])
def list_dataset_partitions(
    limit: int = Query(100, ge=1, le=500),
    current_user: AuthUser = Depends(get_current_user),
) -> list[DatasetPartitionResponse]:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admin can list dataset partitions")
    return get_training_service().list_partitions(limit=limit)


@api_router.post("/training/snapshots", response_model=DatasetSnapshotResponse)
def create_training_snapshot(
    payload: DatasetSnapshotCreateRequest,
    current_user: AuthUser = Depends(get_current_user),
) -> DatasetSnapshotResponse:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admin can create training snapshots")
    service = get_training_service()
    try:
        return service.create_snapshot(payload=payload, created_by=current_user.username)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@api_router.get("/training/snapshots", response_model=list[DatasetSnapshotResponse])
def list_training_snapshots(
    limit: int = Query(100, ge=1, le=500),
    current_user: AuthUser = Depends(get_current_user),
) -> list[DatasetSnapshotResponse]:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admin can list training snapshots")
    return get_training_service().list_snapshots(limit=limit)


@api_router.post("/training/jobs", response_model=TrainingJobResponse)
def create_training_job(
    payload: TrainingJobCreateRequest,
    current_user: AuthUser = Depends(get_current_user),
) -> TrainingJobResponse:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admin can create training jobs")
    service = get_training_service()
    try:
        return service.create_training_job(payload=payload, requested_by=current_user.username)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@api_router.get("/training/jobs", response_model=list[TrainingJobResponse])
def list_training_jobs(
    limit: int = Query(100, ge=1, le=500),
    current_user: AuthUser = Depends(get_current_user),
) -> list[TrainingJobResponse]:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admin can list training jobs")
    return get_training_service().list_training_jobs(limit=limit)


@api_router.get("/training/runs", response_model=list[TrainingRunResponse])
def list_training_runs(
    limit: int = Query(100, ge=1, le=500),
    current_user: AuthUser = Depends(get_current_user),
) -> list[TrainingRunResponse]:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admin can list training runs")
    return get_training_service().list_runs(limit=limit)


@api_router.get("/training/models", response_model=list[RegisteredModelResponse])
def list_registered_models(
    task_type: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    current_user: AuthUser = Depends(get_current_user),
) -> list[RegisteredModelResponse]:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admin can list registered models")
    return get_training_service().list_models(task_type=task_type, limit=limit)


@api_router.get("/training/models/promoted", response_model=RegisteredModelResponse, response_model_exclude_none=True)
def get_promoted_training_model(
    task_type: str = Query("classification"),
    current_user: AuthUser = Depends(get_current_user),
) -> RegisteredModelResponse:
    _ = current_user
    response = get_training_service().get_promoted_model(task_type=task_type)
    if response is None:
        raise HTTPException(status_code=404, detail="Promoted model not found")
    return response


@api_router.post("/training/models/{model_id}/promote", response_model=PromoteModelResponse)
def promote_training_model(
    model_id: str,
    enqueue_prelabels: bool = Query(False),
    force_prelabel: bool = Query(False),
    task_ids: Optional[str] = Query(None),
    current_user: AuthUser = Depends(get_current_user),
) -> PromoteModelResponse:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admin can promote models")
    parsed_task_ids = [
        item.strip()
        for item in str(task_ids or "").split(",")
        if item and item.strip()
    ]
    try:
        response = get_training_service().promote_model(
            model_id=model_id,
            promoted_by=current_user.username,
            enqueue_prelabels=enqueue_prelabels,
            force_prelabel=force_prelabel,
            prelabel_task_ids=parsed_task_ids or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if response is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return response


@api_router.get("/training/models/{model_id}/artifact")
def fetch_training_model_artifact(
    model_id: str,
    current_user: AuthUser = Depends(get_current_user),
) -> Response:
    _ = current_user
    service = get_training_service()
    try:
        file_path = service.get_model_artifact_path(model_id)
    except ValueError as exc:
        message = str(exc)
        if message in {"model not found"}:
            raise HTTPException(status_code=404, detail=message) from exc
        raise HTTPException(status_code=409, detail=message) from exc
    return Response(content=file_path.read_bytes(), media_type="application/octet-stream")


@api_router.post("/training-jobs/claim", response_model=TrainingWorkerClaimResponse)
def claim_training_job(
    payload: TrainingWorkerClaimRequest,
    _worker_token: str = Depends(require_training_worker_token),
) -> TrainingWorkerClaimResponse:
    response = get_training_service().claim_next_job(payload)
    if response is None:
        raise HTTPException(status_code=404, detail="No queued training job")
    return response


@api_router.post("/training-jobs/{job_id}/heartbeat", response_model=TrainingWorkerJobAckResponse)
def heartbeat_training_job(
    job_id: str,
    payload: TrainingWorkerHeartbeatRequest,
    _worker_token: str = Depends(require_training_worker_token),
) -> TrainingWorkerJobAckResponse:
    response = get_training_service().heartbeat_job(job_id=job_id, payload=payload)
    if not response.accepted:
        raise HTTPException(status_code=409, detail="Training job is not claimed by this worker")
    return response


@api_router.get("/training-jobs/{job_id}/snapshot")
def fetch_training_job_snapshot(
    job_id: str,
    worker_id: str = Query(..., min_length=1),
    _worker_token: str = Depends(require_training_worker_token),
) -> Response:
    service = get_training_service()
    try:
        file_path = service.get_claimed_job_snapshot_path(job_id=job_id, worker_id=worker_id)
    except ValueError as exc:
        message = str(exc)
        if message in {"job not found", "snapshot not found"}:
            raise HTTPException(status_code=404, detail=message) from exc
        raise HTTPException(status_code=409, detail=message) from exc
    return Response(content=file_path.read_bytes(), media_type="application/json")


@api_router.post("/training-jobs/{job_id}/artifact", response_model=TrainingArtifactUploadResponse)
async def upload_training_job_artifact(
    job_id: str,
    request: Request,
    worker_id: str = Query(..., min_length=1),
    filename: str = Query(..., min_length=1),
    _worker_token: str = Depends(require_training_worker_token),
) -> TrainingArtifactUploadResponse:
    content = await request.body()
    service = get_training_service()
    try:
        return service.store_uploaded_model_artifact(
            job_id=job_id,
            worker_id=worker_id,
            filename=filename,
            content=content,
        )
    except ValueError as exc:
        message = str(exc)
        if message == "job not found":
            raise HTTPException(status_code=404, detail=message) from exc
        raise HTTPException(status_code=409, detail=message) from exc


@api_router.post("/training-jobs/{job_id}/complete", response_model=TrainingJobLifecycleResponse)
def complete_training_job(
    job_id: str,
    payload: TrainingWorkerCompleteRequest,
    _worker_token: str = Depends(require_training_worker_token),
) -> TrainingJobLifecycleResponse:
    try:
        response = get_training_service().complete_job(job_id=job_id, payload=payload)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if response is None:
        raise HTTPException(status_code=409, detail="Training job is not claimed by this worker")
    return response


@api_router.post("/training-jobs/{job_id}/fail", response_model=TrainingWorkerJobAckResponse)
def fail_training_job(
    job_id: str,
    payload: TrainingWorkerFailRequest,
    _worker_token: str = Depends(require_training_worker_token),
) -> TrainingWorkerJobAckResponse:
    response = get_training_service().fail_job(job_id=job_id, payload=payload)
    if not response.accepted:
        raise HTTPException(status_code=409, detail="Training job is not claimed by this worker")
    return response


@api_router.get("/annotation-sync/status", response_model=AnnotationSyncStatus)
def get_annotation_sync_status(
    request: Request,
    current_user: AuthUser = Depends(get_current_user),
) -> AnnotationSyncStatus:
    _ = current_user
    try:
        service = get_annotation_sync_service()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    scheduler = getattr(request.app.state, "annotation_sync_scheduler", None)
    return service.status(scheduler=scheduler)


@api_router.post("/annotation-sync/run", response_model=AnnotationSyncResult)
def run_annotation_sync(
    request: Request,
    full: bool = False,
    current_user: AuthUser = Depends(get_current_user),
) -> AnnotationSyncResult:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admin can sync annotations")

    try:
        service = get_annotation_sync_service()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not service.config.configured:
        raise HTTPException(status_code=400, detail="PostgreSQL sync is not configured")

    scheduler = getattr(request.app.state, "annotation_sync_scheduler", None)
    if scheduler is not None and getattr(scheduler, "dataset_id", None) == service.dataset_id:
        return scheduler.run_once(full=full)
    return service.sync_now(full=full)


@api_router.get("/dataset/stats")
def get_dataset_stats(
    fresh: bool = Query(False),
    current_user: AuthUser = Depends(get_current_user),
):
    """Return aggregate dataset statistics. Admin-only."""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admin can view dataset statistics")
    return load_dataset_stats_cached(fresh=fresh)


@api_router.get("/tasks/next", response_model=TaskClaimResponse)
def claim_next_task(
    client_id: str = Query(..., min_length=1),
    current_user: AuthUser = Depends(get_current_user),
) -> TaskClaimResponse:
    _ = current_user
    dataset_service = get_dataset_service()
    lock_service = get_task_lock_service()
    try:
        task = lock_service.claim_next_task(client_id=client_id, tasks=dataset_service.list_tasks())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if task is None:
        raise HTTPException(status_code=404, detail="No available task")

    lock = lock_service.get_task_lock(task.task_id)
    if lock is None:
        raise HTTPException(status_code=500, detail="Task lock not found")

    return TaskClaimResponse(
        **task.model_dump(),
        client_id=lock.client_id,
        lock_expires_at=lock.expires_at.isoformat(timespec="seconds"),
    )


@api_router.post("/tasks/{task_id}/claim", response_model=TaskClaimResponse)
def claim_task(
    task_id: str,
    client_id: str = Query(..., min_length=1),
    current_user: AuthUser = Depends(get_current_user),
) -> TaskClaimResponse:
    _ = current_user
    dataset_service = get_dataset_service()
    lock_service = get_task_lock_service()
    tasks = dataset_service.list_tasks()
    normalized_client_id = client_id.strip()
    try:
        task = lock_service.claim_task(task_id=task_id, client_id=client_id, tasks=tasks)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if task is None:
        if not any(item.task_id == task_id for item in tasks):
            raise HTTPException(status_code=404, detail="Task not found")
        lock = lock_service.get_task_lock(task_id)
        if lock is not None and lock.client_id != normalized_client_id:
            raise HTTPException(status_code=409, detail="Task locked by another client")
        raise HTTPException(status_code=409, detail="Task is not available")

    lock = lock_service.get_task_lock(task.task_id)
    if lock is None:
        raise HTTPException(status_code=500, detail="Task lock not found")

    return TaskClaimResponse(
        **task.model_dump(),
        client_id=lock.client_id,
        lock_expires_at=lock.expires_at.isoformat(timespec="seconds"),
    )


@api_router.post("/tasks/{task_id}/heartbeat", response_model=TaskLockHeartbeatResponse)
def heartbeat_task_lock(
    task_id: str,
    client_id: str = Query(..., min_length=1),
    current_user: AuthUser = Depends(get_current_user),
) -> TaskLockHeartbeatResponse:
    _ = current_user
    lock_service = get_task_lock_service()
    normalized_client_id = _require_task_lock_owner(
        task_id=task_id,
        client_id=client_id,
        lock_service=lock_service,
    )
    refreshed_lock = lock_service.refresh_task(task_id=task_id, client_id=normalized_client_id)
    if refreshed_lock is None:
        raise HTTPException(status_code=404, detail="Task lock not found")

    return TaskLockHeartbeatResponse(
        task_id=task_id,
        client_id=refreshed_lock.client_id,
        lock_expires_at=refreshed_lock.expires_at.isoformat(timespec="seconds"),
    )


@api_router.post("/tasks/{task_id}/release", response_model=TaskReleaseResponse)
def release_task_lock(
    task_id: str,
    client_id: str = Query(..., min_length=1),
    current_user: AuthUser = Depends(get_current_user),
) -> TaskReleaseResponse:
    _ = current_user
    lock_service = get_task_lock_service()
    normalized_client_id = _require_task_lock_owner(
        task_id=task_id,
        client_id=client_id,
        lock_service=lock_service,
    )
    released = lock_service.release_task(task_id=task_id, client_id=normalized_client_id)
    if not released:
        raise HTTPException(status_code=404, detail="Task lock not found")

    return TaskReleaseResponse(
        task_id=task_id,
        client_id=normalized_client_id,
        released=True,
    )


@api_router.get("/render/{file_path:path}")
def render_fits_png(
    file_path: str,
    current_user: AuthUser = Depends(get_current_user),
) -> Response:
    _ = current_user
    engine = get_fits_engine()
    try:
        png_data = engine.render_png(file_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="FITS file not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to render FITS") from exc

    return Response(content=png_data, media_type="image/png")


@api_router.get("/fits/{file_path:path}")
def fetch_fits_binary(
    file_path: str,
    current_user: AuthUser = Depends(get_current_user),
) -> Response:
    _ = current_user
    engine = get_fits_engine()
    try:
        fits_data = engine.get_fits_binary(file_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="FITS file not found") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to fetch FITS") from exc

    return Response(content=fits_data, media_type="application/octet-stream")


@api_router.post("/annotations/{task_id}", response_model=AnnotationSaveResponse)
def save_annotations(
    task_id: str,
    payload: AnnotationSaveRequest,
    client_id: Optional[str] = Query(None),
    release_after_save: bool = Query(True),
    current_user: AuthUser = Depends(get_current_user),
) -> AnnotationSaveResponse:
    service = get_annotation_service()
    lock_service = get_task_lock_service()
    normalized_client_id = _require_task_lock_for_save(
        task_id=task_id,
        client_id=client_id,
        lock_service=lock_service,
    )
    try:
        result = service.save(task_id=task_id, payload=payload, submitted_by=current_user.username)
        invalidate_dataset_stats_cache()
        if normalized_client_id and release_after_save:
            lock_service.release_task(task_id=task_id, client_id=normalized_client_id)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@api_router.get("/annotations/{task_id}/history", response_model=AnnotationHistoryResponse)
def get_annotation_history(
    task_id: str,
    current_user: AuthUser = Depends(get_current_user),
) -> AnnotationHistoryResponse:
    _ = current_user
    service = get_annotation_service()
    try:
        return service.list_history(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api_router.get("/annotations/{task_id}/history/{revision_id}", response_model=AnnotationRevisionDetail)
def get_annotation_revision(
    task_id: str,
    revision_id: str,
    current_user: AuthUser = Depends(get_current_user),
) -> AnnotationRevisionDetail:
    _ = current_user
    service = get_annotation_service()
    try:
        return service.get_revision(task_id=task_id, revision_id=revision_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api_router.post("/annotations/{task_id}/rollback/{revision_id}", response_model=AnnotationRollbackResponse)
def rollback_annotation_revision(
    task_id: str,
    revision_id: str,
    current_user: AuthUser = Depends(get_current_user),
) -> AnnotationRollbackResponse:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admin can rollback revisions")

    service = get_annotation_service()
    try:
        return service.rollback_to_revision(
            task_id=task_id,
            revision_id=revision_id,
            submitted_by=current_user.username,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
