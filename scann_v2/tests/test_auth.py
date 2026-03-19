from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from scann.native_annotation.app import app


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"SIMPLE FITS PLACEHOLDER")


def test_protected_api_requires_bearer_token(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    _touch(dataset_root / "new" / "PGC 17069.fts")

    monkeypatch.setenv("SCANN_NATIVE_DATASET_ROOT", str(dataset_root))
    client = TestClient(app)

    response = client.get("/api/tasks")

    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"


def test_login_returns_jwt_and_allows_access(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    _touch(dataset_root / "new" / "PGC 17069.fts")

    monkeypatch.setenv("SCANN_NATIVE_DATASET_ROOT", str(dataset_root))
    client = TestClient(app)

    login = client.post(
        "/api/login",
        json={"username": "annotator", "password": "scann123"},
    )

    assert login.status_code == 200
    token = login.json()["access_token"]
    assert token

    tasks = client.get("/api/tasks", headers={"Authorization": f"Bearer {token}"})
    assert tasks.status_code == 200
