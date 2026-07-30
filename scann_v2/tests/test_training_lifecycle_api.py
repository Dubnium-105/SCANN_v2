from __future__ import annotations

import json
import hashlib
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


def _training_worker_headers() -> dict[str, str]:
    return {"X-SCANN-Worker-Token": "training-worker-secret"}


def _seed_annotation(client: TestClient, dataset_root: Path) -> None:
    _touch(dataset_root / "new" / "PGC 17069.fts")
    _touch(dataset_root / "old" / "PGC 17069.fts")
    annotator_headers = _auth_headers(client, username="annotator", password="scann123")
    save = client.post(
        "/api/annotations/PGC 17069",
        json={
            "source_view": "new",
            "metadata": {"tool": "bbox"},
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


def test_training_snapshot_job_complete_promote_and_enqueue_prelabels(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    monkeypatch.setenv("SCANN_NATIVE_DATASET_ROOT", str(dataset_root))
    monkeypatch.setenv("SCANN_TRAINING_WORKER_TOKEN", "training-worker-secret")

    client = TestClient(app)
    _seed_annotation(client, dataset_root)
    admin_headers = _auth_headers(client)

    snapshot = client.post(
        "/api/training/snapshots",
        json={
            "snapshot_name": "round-1",
            "task_ids": ["PGC 17069"],
            "metadata": {"stage": "bootstrap"},
        },
        headers=admin_headers,
    )
    assert snapshot.status_code == 200
    snapshot_payload = snapshot.json()
    assert snapshot_payload["snapshot_name"] == "round-1"
    assert snapshot_payload["task_count"] == 1
    assert snapshot_payload["annotation_count"] == 1

    snapshots = client.get("/api/training/snapshots", headers=admin_headers)
    assert snapshots.status_code == 200
    assert snapshots.json()[0]["snapshot_id"] == snapshot_payload["snapshot_id"]

    create_job = client.post(
        "/api/training/jobs",
        json={
            "snapshot_id": snapshot_payload["snapshot_id"],
            "task_type": "classification",
            "model_version": "cls-v3",
            "model_id": "cls-v3-run-001",
            "model_backbone": "ViT_B_16",
            "train_config": {"epochs": 2, "batch_size": 4},
            "promote_on_success": True,
            "enqueue_prelabels_on_success": True,
            "prelabel_task_ids": ["PGC 17069"],
        },
        headers=admin_headers,
    )
    assert create_job.status_code == 200
    create_job_payload = create_job.json()
    assert create_job_payload["status"] == "queued"
    assert create_job_payload["snapshot_id"] == snapshot_payload["snapshot_id"]
    assert create_job_payload["model_id"] == "cls-v3-run-001"
    assert create_job_payload["model_backbone"] == "ViT_B_16"

    jobs = client.get("/api/training/jobs", headers=admin_headers)
    assert jobs.status_code == 200
    assert jobs.json()[0]["job_id"] == create_job_payload["job_id"]

    claim = client.post(
        "/api/training-jobs/claim",
        json={
            "worker_id": "trainer-1",
            "host_name": "gpu-pc-01",
            "device_label": "RTX-4090",
            "capabilities": {
                "task_types": ["classification"],
                "model_backbones": ["ViT_B_16"],
            },
        },
        headers=_training_worker_headers(),
    )
    assert claim.status_code == 200
    claim_payload = claim.json()
    assert claim_payload["job_id"] == create_job_payload["job_id"]
    assert claim_payload["snapshot_id"] == snapshot_payload["snapshot_id"]
    assert claim_payload["model_version"] == "cls-v3"
    assert claim_payload["model_id"] == "cls-v3-run-001"
    assert claim_payload["model_backbone"] == "ViT_B_16"

    heartbeat = client.post(
        f"/api/training-jobs/{claim_payload['job_id']}/heartbeat",
        json={"worker_id": "trainer-1"},
        headers=_training_worker_headers(),
    )
    assert heartbeat.status_code == 200
    assert heartbeat.json()["accepted"] is True

    snapshot_doc = client.get(
        f"/api/training-jobs/{claim_payload['job_id']}/snapshot",
        params={"worker_id": "trainer-1"},
        headers=_training_worker_headers(),
    )
    assert snapshot_doc.status_code == 200
    payload = json.loads(snapshot_doc.content.decode("utf-8"))
    assert payload["version"] == "2.3"
    assert payload["images"][0]["id"] == "PGC 17069"

    upload = client.post(
        f"/api/training-jobs/{claim_payload['job_id']}/artifact",
        params={"worker_id": "trainer-1", "filename": "best_model.pth"},
        content=b"mock-checkpoint",
        headers={
            **_training_worker_headers(),
            "Content-Type": "application/octet-stream",
        },
    )
    assert upload.status_code == 200
    upload_payload = upload.json()
    assert upload_payload["artifact_path"].endswith("/best_model.pth")

    complete = client.post(
        f"/api/training-jobs/{claim_payload['job_id']}/complete",
        json={
            "worker_id": "trainer-1",
            "artifact_path": upload_payload["artifact_path"],
            "metrics": {"f1": 0.93, "val_loss": 0.12},
            "metadata": {"epochs": 2, "dataset_snapshot": snapshot_payload["snapshot_id"]},
        },
        headers=_training_worker_headers(),
    )
    assert complete.status_code == 200
    complete_payload = complete.json()
    assert complete_payload["job"]["status"] == "completed"
    assert complete_payload["job"]["run_id"]
    assert complete_payload["run"]["status"] == "completed"
    assert complete_payload["run"]["artifact_path"] == upload_payload["artifact_path"]
    assert complete_payload["model"]["model_id"] == "cls-v3-run-001"
    artifact_metadata = complete_payload["model"]["metadata"]["artifact"]
    assert artifact_metadata["path"] == upload_payload["artifact_path"]
    assert artifact_metadata["size_bytes"] == len(b"mock-checkpoint")
    assert artifact_metadata["sha256"] == hashlib.sha256(b"mock-checkpoint").hexdigest()
    assert complete_payload["model"]["is_promoted"] is False
    assert complete_payload["auto_promoted"] is False
    assert complete_payload["prelabel_enqueue"] is None
    assert any("zero_shot_classes_present" in item for item in complete_payload["promotion_warnings"])

    runs = client.get("/api/training/runs", headers=admin_headers)
    assert runs.status_code == 200
    run_payload = runs.json()[0]
    assert run_payload["job_id"] == claim_payload["job_id"]
    assert run_payload["metrics"]["f1"] == 0.93

    models = client.get("/api/training/models", headers=admin_headers)
    assert models.status_code == 200
    assert models.json()[0]["model_id"] == "cls-v3-run-001"

    promoted = client.get(
        "/api/training/models/promoted",
        params={"task_type": "classification"},
        headers=admin_headers,
    )
    assert promoted.status_code == 404

    artifact = client.get(
        "/api/training/models/cls-v3-run-001/artifact",
        headers=admin_headers,
    )
    assert artifact.status_code == 200
    assert artifact.content == b"mock-checkpoint"

    artifact_file = dataset_root / upload_payload["artifact_path"]
    artifact_file.unlink()
    rejected_promote = client.post(
        "/api/training/models/cls-v3-run-001/promote",
        headers=admin_headers,
    )
    assert rejected_promote.status_code == 409
    assert rejected_promote.json()["detail"] == "model artifact file does not exist"
    artifact_file.write_bytes(b"mock-checkpoint")

    manual_promote = client.post(
        "/api/training/models/cls-v3-run-001/promote",
        params={"enqueue_prelabels": "true", "force_prelabel": "true", "task_ids": "PGC 17069"},
        headers=admin_headers,
    )
    assert manual_promote.status_code == 200
    manual_payload = manual_promote.json()
    assert manual_payload["model"]["is_promoted"] is True
    assert manual_payload["prelabel_enqueue"]["enqueued_count"] == 1
    assert any("zero_shot_classes_present" in item for item in manual_payload["promotion_warnings"])

    promoted = client.get(
        "/api/training/models/promoted",
        params={"task_type": "classification"},
        headers=admin_headers,
    )
    assert promoted.status_code == 200
    promoted_payload = promoted.json()
    assert promoted_payload["model_id"] == "cls-v3-run-001"
    assert promoted_payload["artifact_path"] == upload_payload["artifact_path"]

    tasks = client.get("/api/tasks", headers=admin_headers)
    assert tasks.status_code == 200
    task_payload = tasks.json()[0]
    assert task_payload["prelabel_status"] == "queued"
    assert task_payload["prelabel_model_version"] == "cls-v3"
    assert task_payload["prelabel_model_id"] == "cls-v3-run-001"
    assert task_payload["prelabel_model_backbone"] == "ViT_B_16"


def test_training_job_can_create_snapshot_implicitly_and_requires_worker_token(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    monkeypatch.setenv("SCANN_NATIVE_DATASET_ROOT", str(dataset_root))
    monkeypatch.setenv("SCANN_TRAINING_WORKER_TOKEN", "training-worker-secret")

    client = TestClient(app)
    _seed_annotation(client, dataset_root)
    admin_headers = _auth_headers(client)

    create_job = client.post(
        "/api/training/jobs",
        json={
            "snapshot_name": "auto-snapshot",
            "snapshot_task_ids": ["PGC 17069"],
            "snapshot_metadata": {"round": 2},
            "task_type": "classification",
            "model_version": "cls-v4",
            "model_backbone": "ResNet18",
            "train_config": {"epochs": 1},
        },
        headers=admin_headers,
    )
    assert create_job.status_code == 200
    create_job_payload = create_job.json()
    assert create_job_payload["status"] == "queued"
    assert create_job_payload["snapshot_id"]
    assert create_job_payload["model_id"].startswith("cls-v4-")

    unauthorized_claim = client.post(
        "/api/training-jobs/claim",
        json={"worker_id": "trainer-1"},
    )
    assert unauthorized_claim.status_code == 401
    assert unauthorized_claim.json()["detail"] == "Invalid worker token"

    incompatible_claim = client.post(
        "/api/training-jobs/claim",
        json={
            "worker_id": "trainer-1",
            "capabilities": {
                "task_types": ["classification"],
                "model_backbones": ["ViT_B_16"],
            },
        },
        headers=_training_worker_headers(),
    )
    assert incompatible_claim.status_code == 404
    assert incompatible_claim.json()["detail"] == "No queued training job"


def test_low_support_training_completion_registers_but_does_not_auto_promote(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    monkeypatch.setenv("SCANN_NATIVE_DATASET_ROOT", str(dataset_root))
    monkeypatch.setenv("SCANN_TRAINING_WORKER_TOKEN", "training-worker-secret")

    client = TestClient(app)
    _seed_annotation(client, dataset_root)
    admin_headers = _auth_headers(client)

    snapshot = client.post(
        "/api/training/snapshots",
        json={"snapshot_name": "low-support", "task_ids": ["PGC 17069"]},
        headers=admin_headers,
    )
    assert snapshot.status_code == 200
    snapshot_payload = snapshot.json()
    assert "class_audit" in snapshot_payload["metadata"]

    create_job = client.post(
        "/api/training/jobs",
        json={
            "snapshot_id": snapshot_payload["snapshot_id"],
            "task_type": "classification",
            "model_version": "cls-low",
            "model_id": "cls-low-run",
            "model_backbone": "ResNet18",
            "promote_on_success": True,
            "enqueue_prelabels_on_success": True,
        },
        headers=admin_headers,
    )
    assert create_job.status_code == 200
    assert create_job.json()["train_config"]["imbalance_strategy"] == "class_balanced_focal"

    claim = client.post(
        "/api/training-jobs/claim",
        json={"worker_id": "trainer-low", "capabilities": {"task_types": ["classification"]}},
        headers=_training_worker_headers(),
    )
    assert claim.status_code == 200
    job_id = claim.json()["job_id"]

    upload = client.post(
        f"/api/training-jobs/{job_id}/artifact",
        params={"worker_id": "trainer-low", "filename": "best_model.pth"},
        content=b"mock-low-support-checkpoint",
        headers={**_training_worker_headers(), "Content-Type": "application/octet-stream"},
    )
    assert upload.status_code == 200

    complete = client.post(
        f"/api/training-jobs/{job_id}/complete",
        json={
            "worker_id": "trainer-low",
            "artifact_path": upload.json()["artifact_path"],
            "metrics": {
                "macro_f1_supported": 0.1,
                "promotion_warnings": ["train_support_below_minimum: asteroid"],
            },
        },
        headers=_training_worker_headers(),
    )
    assert complete.status_code == 200
    complete_payload = complete.json()
    assert complete_payload["model"]["is_promoted"] is False
    assert complete_payload["auto_promoted"] is False
    assert complete_payload["prelabel_enqueue"] is None
    assert "auto_promotion_suppressed_due_to_class_coverage" in complete_payload["promotion_warnings"]

    promoted = client.get(
        "/api/training/models/promoted",
        params={"task_type": "classification"},
        headers=admin_headers,
    )
    assert promoted.status_code == 404

    manual = client.post(
        "/api/training/models/cls-low-run/promote",
        headers=admin_headers,
    )
    assert manual.status_code == 200
    manual_payload = manual.json()
    assert manual_payload["model"]["is_promoted"] is True
    assert "train_support_below_minimum: asteroid" in manual_payload["promotion_warnings"]
