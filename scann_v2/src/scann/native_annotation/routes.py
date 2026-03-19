from __future__ import annotations

from fastapi import APIRouter

api_router = APIRouter(prefix="/api", tags=["api"])


@api_router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
