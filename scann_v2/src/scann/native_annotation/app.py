from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .annotation_sync_service import build_annotation_sync_scheduler_from_env
from .routes import api_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    dataset_root = Path(os.getenv("SCANN_NATIVE_DATASET_ROOT", "dataset")).resolve()
    scheduler = build_annotation_sync_scheduler_from_env(dataset_root)
    app.state.annotation_sync_scheduler = scheduler
    if scheduler is not None:
        scheduler.start()
    try:
        yield
    finally:
        if scheduler is not None:
            scheduler.stop()


def create_app() -> FastAPI:
    app = FastAPI(
        title="SCANN Native FITS Annotation API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)
    return app


app = create_app()
