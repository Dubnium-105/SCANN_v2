from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter

from .dataset_service import DatasetService, TaskSession

api_router = APIRouter(prefix="/api", tags=["api"])


def get_dataset_service() -> DatasetService:
    dataset_root = Path(os.getenv("SCANN_NATIVE_DATASET_ROOT", "dataset")).resolve()
    return DatasetService(dataset_root=dataset_root)


@api_router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@api_router.get("/tasks", response_model=list[TaskSession])
def list_tasks() -> list[TaskSession]:
    service = get_dataset_service()
    return service.list_tasks()
