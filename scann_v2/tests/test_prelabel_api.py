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
        json={
            "model_version": "detector-v1",
            "model_id": "model-20260413-001",
            "model_backbone": "ViT_B_16",
            "candidate_limit": 12,
            "confidence_threshold": 0.42,
        },
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
    assert task_payload["prelabel_model_id"] == "model-20260413-001"
    assert task_payload["prelabel_model_backbone"] == "ViT_B_16"

    queued_prelabel = client.get("/api/prelabels/PGC%2017069", headers=admin_headers)
    assert queued_prelabel.status_code == 200
    queued_prelabel_payload = queued_prelabel.json()
    assert queued_prelabel_payload["status"] == "queued"
    assert queued_prelabel_payload["model_version"] == "detector-v1"
    assert queued_prelabel_payload.get("prelabel_id") is None
    assert queued_prelabel_payload["box_count"] == 0

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
    assert claim_payload["model_id"] == "model-20260413-001"
    assert claim_payload["model_backbone"] == "ViT_B_16"
    assert claim_payload["candidate_limit"] == 12
    assert claim_payload["confidence_threshold"] == 0.42
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
            "ai_suggestion": "asteroid",
            "ai_confidence": 0.97,
            "metadata": {"pipeline": "detector-v1"},
            "annotations": [
                {
                    "x": 10.0,
                    "y": 20.0,
                    "width": 30.0,
                    "height": 40.0,
                    "label": None,
                    "detail_type": "asteroid",
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
    assert prelabel_payload["model_id"] == "model-20260413-001"
    assert prelabel_payload["model_backbone"] == "ViT_B_16"
    assert prelabel_payload["candidate_limit"] == 12
    assert prelabel_payload["confidence_threshold"] == 0.42
    assert prelabel_payload["ai_suggestion"] == "asteroid"
    assert prelabel_payload["box_count"] == 1

    tasks_after_complete = client.get("/api/tasks", headers=admin_headers)
    assert tasks_after_complete.status_code == 200
    final_task_payload = tasks_after_complete.json()[0]
    assert final_task_payload["prelabel_status"] == "available"
    assert final_task_payload["prelabel_model_version"] == "detector-v1"
    assert final_task_payload["prelabel_model_id"] == "model-20260413-001"
    assert final_task_payload["prelabel_model_backbone"] == "ViT_B_16"
    assert final_task_payload["prelabel_box_count"] == 1

    enqueue_again = client.post(
        "/api/prelabels/enqueue",
        json={
            "model_version": "detector-v1",
            "candidate_limit": 12,
            "confidence_threshold": 0.42,
        },
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
    assert "No compatible queued prelabel job" in claim.json()["detail"]


def test_prelabel_claim_respects_model_id_and_backbone(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    _touch(dataset_root / "new" / "PGC 17069.fts")

    monkeypatch.setenv("SCANN_NATIVE_DATASET_ROOT", str(dataset_root))
    monkeypatch.setenv("SCANN_PRELABEL_WORKER_TOKEN", "worker-secret")

    client = TestClient(app)
    admin_headers = _auth_headers(client)

    enqueue = client.post(
        "/api/prelabels/enqueue",
        json={
            "model_version": "detector-v1",
            "model_id": "model-a",
            "model_backbone": "ViT_B_16",
        },
        headers=admin_headers,
    )
    assert enqueue.status_code == 200
    assert enqueue.json()["enqueued_count"] == 1

    claim = client.post(
        "/api/prelabel-jobs/claim",
        json={
            "worker_id": "gpu-worker-1",
            "capabilities": {
                "model_versions": ["detector-v1"],
                "model_ids": ["model-b"],
                "model_backbones": ["ResNet18"],
            },
        },
        headers=_worker_headers(),
    )
    assert claim.status_code == 404
    assert "No compatible queued prelabel job" in claim.json()["detail"]


def test_prelabel_claim_treats_auto_backbone_as_wildcard(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    _touch(dataset_root / "new" / "PGC 17069.fts")

    monkeypatch.setenv("SCANN_NATIVE_DATASET_ROOT", str(dataset_root))
    monkeypatch.setenv("SCANN_PRELABEL_WORKER_TOKEN", "worker-secret")

    client = TestClient(app)
    admin_headers = _auth_headers(client)

    enqueue = client.post(
        "/api/prelabels/enqueue",
        json={
            "model_version": "detector-v1",
            "model_id": "model-a",
            "model_backbone": "ViT_B_16",
        },
        headers=admin_headers,
    )
    assert enqueue.status_code == 200
    assert enqueue.json()["enqueued_count"] == 1

    claim = client.post(
        "/api/prelabel-jobs/claim",
        json={
            "worker_id": "gpu-worker-1",
            "capabilities": {
                "model_versions": ["detector-v1"],
                "model_ids": ["model-a"],
                "model_backbones": ["auto"],
            },
        },
        headers=_worker_headers(),
    )
    assert claim.status_code == 200
    claim_payload = claim.json()
    assert claim_payload["task_id"] == "PGC 17069"
    assert claim_payload["model_backbone"] == "ViT_B_16"


def test_prelabel_enqueue_treats_threshold_and_limit_as_distinct_config(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    _touch(dataset_root / "new" / "PGC 17069.fts")

    monkeypatch.setenv("SCANN_NATIVE_DATASET_ROOT", str(dataset_root))
    monkeypatch.setenv("SCANN_PRELABEL_WORKER_TOKEN", "worker-secret")

    client = TestClient(app)
    admin_headers = _auth_headers(client)

    first_enqueue = client.post(
        "/api/prelabels/enqueue",
        json={
            "model_version": "detector-v1",
            "candidate_limit": 10,
            "confidence_threshold": 0.30,
        },
        headers=admin_headers,
    )
    assert first_enqueue.status_code == 200
    assert first_enqueue.json()["enqueued_count"] == 1

    claim = client.post(
        "/api/prelabel-jobs/claim",
        json={"worker_id": "gpu-worker-1"},
        headers=_worker_headers(),
    )
    assert claim.status_code == 200
    claim_payload = claim.json()

    complete = client.post(
        f"/api/prelabel-jobs/{claim_payload['job_id']}/complete",
        json={
            "worker_id": "gpu-worker-1",
            "annotations": [],
        },
        headers=_worker_headers(),
    )
    assert complete.status_code == 200

    second_enqueue = client.post(
        "/api/prelabels/enqueue",
        json={
            "model_version": "detector-v1",
            "candidate_limit": 5,
            "confidence_threshold": 0.70,
        },
        headers=admin_headers,
    )
    assert second_enqueue.status_code == 200
    assert second_enqueue.json()["enqueued_count"] == 1
    assert second_enqueue.json()["skipped_count"] == 0


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


def test_annotation_save_accepts_applied_prelabel_and_skips_duplicate_enqueue(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    _touch(dataset_root / "new" / "PGC 17069.fts")
    _touch(dataset_root / "old" / "PGC 17069.fts")

    monkeypatch.setenv("SCANN_NATIVE_DATASET_ROOT", str(dataset_root))
    monkeypatch.setenv("SCANN_PRELABEL_WORKER_TOKEN", "worker-secret")

    client = TestClient(app)
    admin_headers = _auth_headers(client)
    annotator_headers = _auth_headers(client, username="annotator", password="scann123")

    enqueue = client.post(
        "/api/prelabels/enqueue",
        json={
            "model_version": "detector-v1",
            "model_id": "model-accept-001",
            "model_backbone": "ViT_B_16",
        },
        headers=admin_headers,
    )
    assert enqueue.status_code == 200
    assert enqueue.json()["enqueued_count"] == 1

    claim = client.post(
        "/api/prelabel-jobs/claim",
        json={"worker_id": "gpu-worker-1"},
        headers=_worker_headers(),
    )
    assert claim.status_code == 200
    claim_payload = claim.json()

    complete = client.post(
        f"/api/prelabel-jobs/{claim_payload['job_id']}/complete",
        json={
            "worker_id": "gpu-worker-1",
            "source_view": "new",
            "ai_suggestion": "asteroid",
            "ai_confidence": 0.97,
            "annotations": [
                {
                    "x": 12.0,
                    "y": 18.0,
                    "width": 32.0,
                    "height": 26.0,
                    "label": None,
                    "detail_type": "asteroid",
                    "confidence": 0.97,
                }
            ],
        },
        headers=_worker_headers(),
    )
    assert complete.status_code == 200
    prelabel_id = complete.json()["prelabel_id"]

    save = client.post(
        "/api/annotations/PGC 17069",
        json={
            "source_view": "new",
            "metadata": {
                "tool": "bbox",
                "applied_prelabel": {
                    "prelabel_id": prelabel_id,
                    "model_version": "detector-v1",
                    "model_id": "model-accept-001",
                    "model_backbone": "ViT_B_16",
                    "imported_annotation_count": 1,
                },
            },
            "annotations": [
                {
                    "x": 12.0,
                    "y": 18.0,
                    "width": 32.0,
                    "height": 26.0,
                    "label": "Positive",
                    "detail_type": "candidate",
                }
            ],
        },
        headers=annotator_headers,
    )
    assert save.status_code == 200
    save_payload = save.json()
    assert save_payload["accepted_prelabel_id"] == prelabel_id
    assert save_payload["revision_id"]

    prelabel = client.get("/api/prelabels/PGC%2017069", headers=annotator_headers)
    assert prelabel.status_code == 200
    prelabel_payload = prelabel.json()
    assert prelabel_payload["status"] == "accepted"
    assert prelabel_payload.get("prelabel_id") is None
    assert prelabel_payload["box_count"] == 1

    tasks = client.get("/api/tasks", headers=annotator_headers)
    assert tasks.status_code == 200
    task_payload = tasks.json()[0]
    assert task_payload["prelabel_status"] == "accepted"
    assert task_payload["prelabel_model_version"] == "detector-v1"
    assert task_payload["prelabel_model_id"] == "model-accept-001"
    assert task_payload["prelabel_model_backbone"] == "ViT_B_16"
    assert task_payload["prelabel_box_count"] == 1

    enqueue_again = client.post(
        "/api/prelabels/enqueue",
        json={"model_version": "detector-v1"},
        headers=admin_headers,
    )
    assert enqueue_again.status_code == 200
    assert enqueue_again.json()["enqueued_count"] == 0
    assert enqueue_again.json()["skipped_count"] == 1


def test_prelabel_management_list_cancel_and_worker_status(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    _touch(dataset_root / "new" / "PGC 17069.fts")
    _touch(dataset_root / "new" / "PGC 17070.fts")

    monkeypatch.setenv("SCANN_NATIVE_DATASET_ROOT", str(dataset_root))
    monkeypatch.setenv("SCANN_PRELABEL_WORKER_TOKEN", "worker-secret")

    client = TestClient(app)
    admin_headers = _auth_headers(client)

    enqueue = client.post(
        "/api/prelabels/enqueue",
        json={
            "model_version": "detector-v2",
            "model_id": "model-manage-001",
            "model_backbone": "ViT_B_16",
        },
        headers=admin_headers,
    )
    assert enqueue.status_code == 200
    assert enqueue.json()["enqueued_count"] == 2

    claim = client.post(
        "/api/prelabel-jobs/claim",
        json={
            "worker_id": "gpu-worker-9",
            "display_name": "GPU Worker",
            "host_name": "pc-09",
            "device_label": "RTX-4090",
            "capabilities": {
                "model_versions": ["detector-v2"],
                "model_ids": ["model-manage-001"],
                "model_backbones": ["ViT_B_16"],
            },
        },
        headers=_worker_headers(),
    )
    assert claim.status_code == 200

    listed_jobs = client.get("/api/prelabels/jobs?limit=10", headers=admin_headers)
    assert listed_jobs.status_code == 200
    listed_job_payload = listed_jobs.json()
    assert len(listed_job_payload) == 2
    assert {item["status"] for item in listed_job_payload} == {"queued", "claimed"}

    workers = client.get("/api/prelabels/workers?limit=10", headers=admin_headers)
    assert workers.status_code == 200
    workers_payload = workers.json()
    assert len(workers_payload) == 1
    assert workers_payload[0]["worker_id"] == "gpu-worker-9"
    assert workers_payload[0]["capabilities"]["model_ids"] == ["model-manage-001"]

    cancel = client.post(
        "/api/prelabels/jobs/cancel",
        json={
            "task_ids": ["PGC 17069", "PGC 17070"],
            "statuses": ["queued", "claimed"],
            "reason": "cancelled from test",
        },
        headers=admin_headers,
    )
    assert cancel.status_code == 200
    cancel_payload = cancel.json()
    assert cancel_payload["cancelled_count"] == 2
    assert all(item["status"] == "cancelled" for item in cancel_payload["jobs"])

    tasks = client.get("/api/tasks", headers=admin_headers)
    assert tasks.status_code == 200
    by_id = {item["task_id"]: item for item in tasks.json()}
    assert by_id["PGC 17069"]["prelabel_status"] == "cancelled"
    assert by_id["PGC 17070"]["prelabel_status"] == "cancelled"

    cancelled_summary = client.get("/api/prelabels/PGC%2017069", headers=admin_headers)
    assert cancelled_summary.status_code == 200
    cancelled_payload = cancelled_summary.json()
    assert cancelled_payload["status"] == "cancelled"
    assert cancelled_payload.get("prelabel_id") is None
