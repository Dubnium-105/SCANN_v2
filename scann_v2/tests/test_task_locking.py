from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from scann.native_annotation.app import app


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"SIMPLE FITS PLACEHOLDER")


def _annotation_payload() -> dict:
    return {
        "bucket": "positive",
        "source_view": "new",
        "metadata": {"annotator": "tester"},
        "annotations": [
            {"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0, "label": "Positive"},
        ],
    }


def _auth_headers(client: TestClient, username: str = "annotator", password: str = "scann123") -> dict[str, str]:
    response = client.post(
        "/api/login",
        json={"username": username, "password": password},
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_claim_next_task_locks_task_between_clients(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"

    _touch(dataset_root / "new" / "PGC 17069.fts")
    _touch(dataset_root / "old" / "PGC 17069.fts")
    _touch(dataset_root / "new_marked" / "PGC 17069.fts")

    monkeypatch.setenv("SCANN_NATIVE_DATASET_ROOT", str(dataset_root))
    client = TestClient(app)
    headers = _auth_headers(client)

    response_a = client.get("/api/tasks/next", params={"client_id": "client-a"}, headers=headers)
    response_b = client.get("/api/tasks/next", params={"client_id": "client-b"}, headers=headers)

    assert response_a.status_code == 200
    assert response_a.json()["task_id"] == "PGC 17069"

    assert response_b.status_code == 404
    assert response_b.json()["detail"] == "No available task"


def test_annotation_submit_releases_lock_for_next_client(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"

    _touch(dataset_root / "new" / "PGC 17069.fts")
    _touch(dataset_root / "old" / "PGC 17069.fts")
    _touch(dataset_root / "new_marked" / "PGC 17069.fts")

    monkeypatch.setenv("SCANN_NATIVE_DATASET_ROOT", str(dataset_root))
    client = TestClient(app)
    headers = _auth_headers(client)

    claim_a = client.get("/api/tasks/next", params={"client_id": "client-a"}, headers=headers)
    assert claim_a.status_code == 200

    save_resp = client.post(
        "/api/annotations/PGC 17069",
        params={"client_id": "client-a", "release_after_save": "true"},
        json=_annotation_payload(),
        headers=headers,
    )
    assert save_resp.status_code == 200

    claim_b = client.get("/api/tasks/next", params={"client_id": "client-b"}, headers=headers)
    assert claim_b.status_code == 200
    assert claim_b.json()["task_id"] == "PGC 17069"


def test_claim_specific_task_blocks_other_clients_and_save_requires_owner(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"

    _touch(dataset_root / "new" / "PGC 17069.fts")
    _touch(dataset_root / "old" / "PGC 17069.fts")
    _touch(dataset_root / "new_marked" / "PGC 17069.fts")

    monkeypatch.setenv("SCANN_NATIVE_DATASET_ROOT", str(dataset_root))
    client = TestClient(app)
    headers = _auth_headers(client)

    claim_a = client.post("/api/tasks/PGC 17069/claim", params={"client_id": "client-a"}, headers=headers)
    assert claim_a.status_code == 200
    assert claim_a.json()["task_id"] == "PGC 17069"

    claim_b = client.post("/api/tasks/PGC 17069/claim", params={"client_id": "client-b"}, headers=headers)
    assert claim_b.status_code == 409
    assert claim_b.json()["detail"] == "Task locked by another client"

    save_without_owner = client.post(
        "/api/annotations/PGC 17069",
        json=_annotation_payload(),
        headers=headers,
    )
    assert save_without_owner.status_code == 409
    assert save_without_owner.json()["detail"] == "Task locked by another client"

    save_wrong_owner = client.post(
        "/api/annotations/PGC 17069",
        params={"client_id": "client-b"},
        json=_annotation_payload(),
        headers=headers,
    )
    assert save_wrong_owner.status_code == 409
    assert save_wrong_owner.json()["detail"] == "Task locked by another client"

    save_owner = client.post(
        "/api/annotations/PGC 17069",
        params={"client_id": "client-a", "release_after_save": "false"},
        json=_annotation_payload(),
        headers=headers,
    )
    assert save_owner.status_code == 200

    release_owner = client.post(
        "/api/tasks/PGC 17069/release",
        params={"client_id": "client-a"},
        headers=headers,
    )
    assert release_owner.status_code == 200


def test_release_endpoint_releases_lock_for_next_client(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"

    _touch(dataset_root / "new" / "PGC 17069.fts")
    _touch(dataset_root / "old" / "PGC 17069.fts")
    _touch(dataset_root / "new_marked" / "PGC 17069.fts")

    monkeypatch.setenv("SCANN_NATIVE_DATASET_ROOT", str(dataset_root))
    client = TestClient(app)
    headers = _auth_headers(client)

    claim_a = client.get("/api/tasks/next", params={"client_id": "client-a"}, headers=headers)
    assert claim_a.status_code == 200

    release_resp = client.post(
        "/api/tasks/PGC 17069/release",
        params={"client_id": "client-a"},
        headers=headers,
    )
    assert release_resp.status_code == 200
    assert release_resp.json() == {
        "task_id": "PGC 17069",
        "client_id": "client-a",
        "released": True,
    }

    claim_b = client.get("/api/tasks/next", params={"client_id": "client-b"}, headers=headers)
    assert claim_b.status_code == 200
    assert claim_b.json()["task_id"] == "PGC 17069"


def test_heartbeat_extends_lock_and_keeps_other_clients_blocked(tmp_path, monkeypatch) -> None:
    from scann.native_annotation import routes as native_routes

    dataset_root = tmp_path / "dataset"

    _touch(dataset_root / "new" / "PGC 17069.fts")
    _touch(dataset_root / "old" / "PGC 17069.fts")
    _touch(dataset_root / "new_marked" / "PGC 17069.fts")

    monkeypatch.setenv("SCANN_NATIVE_DATASET_ROOT", str(dataset_root))
    client = TestClient(app)
    headers = _auth_headers(client)

    claim_a = client.get("/api/tasks/next", params={"client_id": "client-a"}, headers=headers)
    assert claim_a.status_code == 200
    initial_expires_at = datetime.fromisoformat(claim_a.json()["lock_expires_at"])

    lock_service = native_routes.get_task_lock_service()
    existing_lock = lock_service.get_task_lock("PGC 17069")
    assert existing_lock is not None

    future_now = existing_lock.locked_at.replace(microsecond=0) + timedelta(seconds=1)
    monkeypatch.setattr(lock_service, "_now", lambda: future_now)

    heartbeat_resp = client.post(
        "/api/tasks/PGC 17069/heartbeat",
        params={"client_id": "client-a"},
        headers=headers,
    )
    assert heartbeat_resp.status_code == 200
    refreshed_expires_at = datetime.fromisoformat(heartbeat_resp.json()["lock_expires_at"])
    assert refreshed_expires_at > initial_expires_at

    claim_b = client.get("/api/tasks/next", params={"client_id": "client-b"}, headers=headers)
    assert claim_b.status_code == 404
    assert claim_b.json()["detail"] == "No available task"


def test_release_and_heartbeat_reject_non_owner(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"

    _touch(dataset_root / "new" / "PGC 17069.fts")
    _touch(dataset_root / "old" / "PGC 17069.fts")
    _touch(dataset_root / "new_marked" / "PGC 17069.fts")

    monkeypatch.setenv("SCANN_NATIVE_DATASET_ROOT", str(dataset_root))
    client = TestClient(app)
    headers = _auth_headers(client)

    claim_a = client.get("/api/tasks/next", params={"client_id": "client-a"}, headers=headers)
    assert claim_a.status_code == 200

    release_resp = client.post(
        "/api/tasks/PGC 17069/release",
        params={"client_id": "client-b"},
        headers=headers,
    )
    assert release_resp.status_code == 409
    assert release_resp.json()["detail"] == "Task locked by another client"

    heartbeat_resp = client.post(
        "/api/tasks/PGC 17069/heartbeat",
        params={"client_id": "client-b"},
        headers=headers,
    )
    assert heartbeat_resp.status_code == 409
    assert heartbeat_resp.json()["detail"] == "Task locked by another client"
