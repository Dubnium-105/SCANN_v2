from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from scann.native_annotation.app import app


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"SIMPLE FITS PLACEHOLDER")


def _auth_headers(client: TestClient, username: str = "admin", password: str = "admin123") -> dict[str, str]:
    response = client.post(
        "/api/login",
        json={"username": username, "password": password},
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _worker_headers() -> dict[str, str]:
    return {"X-SCANN-Worker-Token": "worker-secret"}


def test_prelabel_enqueue_claim_complete_and_list_status(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    _touch(dataset_root / "new" / "PGC 17069.fts")
    _touch(dataset_root / "old" / "PGC 17069.fts")

    monkeypatch.setenv("SCANN_NATIVE_DATASET_ROOT", str(dataset_root))
    monkeypatch.setenv("SCANN_PRELABEL_WORKER_TOKEN", "worker-secret")
    monkeypatch.setenv("SCANN_PRELABEL_JOB_TIMEOUT_SECONDS", "120")

    client = TestClient(app)
    admin_headers = _auth_headers(client)

    enqueue = client.post(
        "/api/prelabels/enqueue",
        json={"model_version": "detector-v1"},
        headers=admin_headers,
    )

    assert enqueue.status_code == 200
    enqueue_payload = enqueue.json()
    assert enqueue_payload["requested_count"] == 1
    assert enqueue_payload["enqueued_count"] == 1
    assert enqueue_payload["skipped_count"] == 0
    assert len(enqueue_payload["job_ids"]) == 1

    tasks_after_enqueue = client.get("/api/tasks", headers=admin_headers)
    assert tasks_after_enqueue.status_code == 200
    task_payload = tasks_after_enqueue.json()[0]
    assert task_payload["prelabel_status"] == "queued"
    assert task_payload["prelabel_model_version"] == "detector-v1"

    claim = client.post(
        "/api/prelabel-jobs/claim",
        json={
            "worker_id": "gpu-worker-1",
            "host_name": "pc-01",
            "device_label": "RTX-4090",
        },
        headers=_worker_headers(),
    )
    assert claim.status_code == 200
    claim_payload = claim.json()
    assert claim_payload["task_id"] == "PGC 17069"
    assert claim_payload["model_version"] == "detector-v1"
    assert claim_payload["paths"]["new"] == "new/PGC 17069.fts"
    assert claim_payload["paths"]["old"] == "old/PGC 17069.fts"

    fetched_fits = client.get(
        f"/api/prelabel-jobs/{claim_payload['job_id']}/fits/new",
        params={"worker_id": "gpu-worker-1"},
        headers=_worker_headers(),
    )
    assert fetched_fits.status_code == 200
    assert fetched_fits.content == (dataset_root / "new" / "PGC 17069.fts").read_bytes()

    heartbeat = client.post(
        f"/api/prelabel-jobs/{claim_payload['job_id']}/heartbeat",
        json={"worker_id": "gpu-worker-1"},
        headers=_worker_headers(),
    )
    assert heartbeat.status_code == 200
    assert heartbeat.json()["accepted"] is True

    complete = client.post(
        f"/api/prelabel-jobs/{claim_payload['job_id']}/complete",
        json={
            "worker_id": "gpu-worker-1",
            "source_view": "new",
            "ai_suggestion": "real",
            "ai_confidence": 0.97,
            "metadata": {"pipeline": "detector-v1"},
            "annotations": [
                {
                    "x": 10.0,
                    "y": 20.0,
                    "width": 30.0,
                    "height": 40.0,
                    "label": "Positive",
                    "detail_type": "candidate",
                    "confidence": 0.97,
                }
            ],
        },
        headers=_worker_headers(),
    )
    assert complete.status_code == 200
    complete_payload = complete.json()
    assert complete_payload["task_id"] == "PGC 17069"
    assert complete_payload["status"] == "available"
    assert complete_payload["box_count"] == 1
    assert complete_payload["annotations"][0]["confidence"] == 0.97

    prelabel = client.get("/api/prelabels/PGC%2017069", headers=admin_headers)
    assert prelabel.status_code == 200
    prelabel_payload = prelabel.json()
    assert prelabel_payload["model_version"] == "detector-v1"
    assert prelabel_payload["ai_suggestion"] == "real"
    assert prelabel_payload["box_count"] == 1

    tasks_after_complete = client.get("/api/tasks", headers=admin_headers)
    assert tasks_after_complete.status_code == 200
    final_task_payload = tasks_after_complete.json()[0]
    assert final_task_payload["prelabel_status"] == "available"
    assert final_task_payload["prelabel_model_version"] == "detector-v1"
    assert final_task_payload["prelabel_box_count"] == 1

    enqueue_again = client.post(
        "/api/prelabels/enqueue",
        json={"model_version": "detector-v1"},
        headers=admin_headers,
    )
    assert enqueue_again.status_code == 200
    assert enqueue_again.json()["enqueued_count"] == 0
    assert enqueue_again.json()["skipped_count"] == 1


def test_prelabel_claim_respects_supported_model_versions(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    _touch(dataset_root / "new" / "PGC 17069.fts")

    monkeypatch.setenv("SCANN_NATIVE_DATASET_ROOT", str(dataset_root))
    monkeypatch.setenv("SCANN_PRELABEL_WORKER_TOKEN", "worker-secret")

    client = TestClient(app)
    admin_headers = _auth_headers(client)

    enqueue = client.post(
        "/api/prelabels/enqueue",
        json={"model_version": "detector-v2"},
        headers=admin_headers,
    )
    assert enqueue.status_code == 200
    assert enqueue.json()["enqueued_count"] == 1

    claim = client.post(
        "/api/prelabel-jobs/claim",
        json={
            "worker_id": "gpu-worker-1",
            "capabilities": {"model_versions": ["detector-v1"]},
        },
        headers=_worker_headers(),
    )
    assert claim.status_code == 404
    assert claim.json()["detail"] == "No queued prelabel job"


def test_prelabel_worker_endpoints_require_token(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    _touch(dataset_root / "new" / "PGC 17069.fts")
    monkeypatch.setenv("SCANN_NATIVE_DATASET_ROOT", str(dataset_root))
    monkeypatch.setenv("SCANN_PRELABEL_WORKER_TOKEN", "worker-secret")

    client = TestClient(app)

    response = client.post(
        "/api/prelabel-jobs/claim",
        json={"worker_id": "gpu-worker-1"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid worker token"
