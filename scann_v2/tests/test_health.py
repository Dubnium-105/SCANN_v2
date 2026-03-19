from __future__ import annotations

from fastapi.testclient import TestClient

from scann.native_annotation.app import app


def test_api_health_returns_ok() -> None:
    client = TestClient(app)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
