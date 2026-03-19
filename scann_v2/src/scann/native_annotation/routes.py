from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from .annotation_service import AnnotationSaveRequest, AnnotationSaveResponse, AnnotationService
from .dataset_service import DatasetService, TaskSession
from .fits_engine import FITSEngine

api_router = APIRouter(prefix="/api", tags=["api"])
_fits_engines: Dict[str, FITSEngine] = {}


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


def get_annotation_service() -> AnnotationService:
    return AnnotationService(dataset_root=get_dataset_root())


@api_router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@api_router.get("/tasks", response_model=list[TaskSession])
def list_tasks() -> list[TaskSession]:
    service = get_dataset_service()
    return service.list_tasks()


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
def save_annotations(task_id: str, payload: AnnotationSaveRequest) -> AnnotationSaveResponse:
    service = get_annotation_service()
    try:
        return service.save(task_id=task_id, payload=payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
