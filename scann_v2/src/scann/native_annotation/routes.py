from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

from .annotation_service import AnnotationSaveRequest, AnnotationSaveResponse, AnnotationService
from .dataset_service import DatasetService, TaskSession
from .fits_engine import FITSEngine
from .task_lock_service import TaskLockService

api_router = APIRouter(prefix="/api", tags=["api"])
_fits_engines: Dict[str, FITSEngine] = {}
_task_lock_services: Dict[str, TaskLockService] = {}


class TaskClaimResponse(TaskSession):
    client_id: str
    lock_expires_at: str


def get_dataset_root() -> Path:
    return Path(os.getenv("SCANN_NATIVE_DATASET_ROOT", "dataset")).resolve()


def get_dataset_service() -> DatasetService:
    return DatasetService(dataset_root=get_dataset_root())


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
        lock_service = TaskLockService(lock_timeout_seconds=timeout_seconds)
        _task_lock_services[key] = lock_service
    return lock_service


def get_annotation_service() -> AnnotationService:
    return AnnotationService(dataset_root=get_dataset_root())


@api_router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@api_router.get("/tasks", response_model=list[TaskSession])
def list_tasks() -> list[TaskSession]:
    service = get_dataset_service()
    return service.list_tasks()


@api_router.get("/tasks/next", response_model=TaskClaimResponse)
def claim_next_task(client_id: str = Query(..., min_length=1)) -> TaskClaimResponse:
    dataset_service = get_dataset_service()
    lock_service = get_task_lock_service()
    task = lock_service.claim_next_task(client_id=client_id, tasks=dataset_service.list_tasks())
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


@api_router.get("/render/{file_path:path}")
def render_fits_png(file_path: str) -> Response:
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


@api_router.post("/annotations/{task_id}", response_model=AnnotationSaveResponse)
def save_annotations(
    task_id: str,
    payload: AnnotationSaveRequest,
    client_id: Optional[str] = Query(None),
) -> AnnotationSaveResponse:
    service = get_annotation_service()
    lock_service = get_task_lock_service()
    try:
        result = service.save(task_id=task_id, payload=payload)
        if client_id:
            lock_service.release_task(task_id=task_id, client_id=client_id)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
