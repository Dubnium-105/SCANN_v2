from __future__ import annotations

import hashlib
import json
import sqlite3

from fastapi.testclient import TestClient

from scann.core.dataset_storage import (
    DatasetStorage,
    RawAssetRecord,
    TaskRecord,
)
from scann.native_annotation.app import app


def _auth_headers(
    client: TestClient,
    username: str = "admin",
    password: str = "admin123",
) -> dict[str, str]:
    response = client.post(
        "/api/login",
        json={"username": username, "password": password},
    )
    assert response.status_code == 200
    return {
        "Authorization": f"Bearer {response.json()['access_token']}"
    }


def _seed_discovery_dataset(dataset_root):
    storage = DatasetStorage(dataset_root)
    storage.ensure_schema()
    storage.upsert_raw_assets(
        [
            RawAssetRecord(
                asset_id="asset-1",
                asset_role="new",
                field_key="field-1",
                field_name="field-1",
                capture_key="capture-1",
                relpath="new/task-1.fts",
                file_name="task-1.fts",
                file_stem="task-1",
                suffix=".fts",
            )
        ]
    )
    storage.sync_tasks(
        [
            TaskRecord(
                task_id="task-1",
                field_key="field-1",
                field_name="field-1",
                capture_key="capture-1",
                new_asset_id="asset-1",
                preprocess_status="ready",
            )
        ]
    )
    storage.create_dataset_partition(
        partition_id="partition-1",
        partition_name="gold-v1",
        manifest_relpath=".scann_control/partitions/partition-1.json",
        manifest_sha256="a" * 64,
        taxonomy_version="taxonomy-v1",
        split_strategy="grouped",
        seed=42,
        task_count=1,
        train_task_count=0,
        validation_task_count=0,
        test_task_count=1,
        activate=False,
        created_by="admin",
    )

    artifact = (
        dataset_root
        / ".scann_control"
        / "models"
        / "model-1"
        / "model.pth"
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"valid model artifact")
    artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    with sqlite3.connect(storage.db_path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(
            """
            INSERT INTO model_registry (
                model_id, model_version, model_backbone, task_type,
                artifact_path, metrics_json, metadata_json, created_by,
                created_at, updated_at
            ) VALUES (
                'model-1', 'v1', 'test', 'classification',
                '.scann_control/models/model-1/model.pth',
                '{}', ?, 'admin',
                '2026-01-01T00:00:00+00:00',
                '2026-01-01T00:00:00+00:00'
            )
            """,
            (
                json.dumps(
                    {"artifact": {"sha256": artifact_sha256}}
                ),
            ),
        )
        connection.execute(
            """
            INSERT INTO task_ai_prelabels (
                prelabel_id, task_id, source_view, ai_suggestion,
                ai_confidence, model_version, model_id, model_backbone,
                status, box_count, metadata_json, created_at, updated_at
            ) VALUES (
                'prelabel-1', 'task-1', 'new', 'real',
                0.9, 'v1', 'model-1', 'test',
                'accepted', 1, '{}',
                '2026-01-01T00:00:00+00:00',
                '2026-01-01T00:00:00+00:00'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO task_ai_prelabel_boxes (
                prelabel_id, box_index, x, y, width, height,
                label, detail_type, confidence
            ) VALUES (
                'prelabel-1', 0, 10, 10, 8, 8,
                'real', 'supernova', 0.9
            )
            """
        )
    revision_id = storage.append_annotation_revision(
        task_id="task-1",
        source_view="new",
        parent_revision_id=None,
        rollback_of_revision_id=None,
        submitted_by="annotator",
        origin="online",
        saved_at="2026-01-01T00:01:00+00:00",
        metadata={},
        annotations=[
            {
                "x": 10,
                "y": 10,
                "width": 8,
                "height": 8,
                "label": "real",
                "detail_type": "supernova",
            }
        ],
        revision_id="revision-1",
    )
    return storage, revision_id


def test_discovery_lifecycle_api_contracts(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    monkeypatch.setenv("SCANN_NATIVE_DATASET_ROOT", str(dataset_root))
    _storage, revision_id = _seed_discovery_dataset(dataset_root)
    client = TestClient(app)
    admin_headers = _auth_headers(client)

    evaluation = client.post(
        "/api/evaluations",
        headers=admin_headers,
        json={
            "run_type": "candidate",
            "partition_id": "partition-1",
            "model_id": "model-1",
            "records": [
                {
                    "task_id": "task-1",
                    "truth": [
                        {"x": 10, "y": 10, "width": 8, "height": 8}
                    ],
                    "candidates": [
                        {"x": 10, "y": 10, "width": 8, "height": 8}
                    ],
                }
            ],
        },
    )
    assert evaluation.status_code == 200
    evaluation_payload = evaluation.json()
    assert evaluation_payload["status"] == "completed"
    assert evaluation_payload["metrics"]["recall"] == 1.0

    active_learning = client.post(
        "/api/active-learning/batches",
        headers=admin_headers,
        json={
            "batch_name": "round-1",
            "budget": 1,
            "model_id": "model-1",
            "partition_id": "partition-1",
            "items": [
                {
                    "task_id": "task-1",
                    "uncertainty": 0.9,
                    "group_key": "night-1/field-1",
                    "ood": True,
                }
            ],
        },
    )
    assert active_learning.status_code == 200
    assert active_learning.json()["items"][0]["dual_review"] is True

    feedback = client.post(
        "/api/review-feedback",
        headers=admin_headers,
        json={
            "task_id": "task-1",
            "prelabel_id": "prelabel-1",
            "revision_id": revision_id,
            "review_duration_seconds": 12.5,
        },
    )
    assert feedback.status_code == 200
    assert feedback.json()["outcome"] == "full_accept"

    shadow = client.post(
        "/api/training/models/model-1/deployments/shadow",
        headers=admin_headers,
        json={"evaluation_run_id": evaluation_payload["run_id"]},
    )
    assert shadow.status_code == 200
    assert shadow.json()["stage"] == "shadow"
    assert shadow.json()["config"]["affects_visible_prelabels"] is False

    rejected_canary = client.post(
        "/api/training/models/model-1/deployments/canary",
        headers=admin_headers,
        json={"traffic_fraction": 0.1, "human_approved": False},
    )
    assert rejected_canary.status_code == 409

    canary = client.post(
        "/api/training/models/model-1/deployments/canary",
        headers=admin_headers,
        json={"traffic_fraction": 0.1, "human_approved": True},
    )
    assert canary.status_code == 200
    assert canary.json()["stage"] == "canary"

    rejected_promotion = client.post(
        "/api/training/models/model-1/deployments/promote",
        headers=admin_headers,
        json={
            "evaluation_run_id": evaluation_payload["run_id"],
            "taxonomy_version": "taxonomy-v1",
            "partition_id": "partition-1",
            "required_metrics": {"recall": 0.9},
            "shadow_drift_ok": True,
            "canary_review_ok": True,
            "human_approved": False,
        },
    )
    assert rejected_promotion.status_code == 409

    promotion = client.post(
        "/api/training/models/model-1/deployments/promote",
        headers=admin_headers,
        json={
            "evaluation_run_id": evaluation_payload["run_id"],
            "taxonomy_version": "taxonomy-v1",
            "partition_id": "partition-1",
            "required_metrics": {"recall": 0.9},
            "shadow_drift_ok": True,
            "canary_review_ok": True,
            "human_approved": True,
        },
    )
    assert promotion.status_code == 200
    assert promotion.json()["stage"] == "promoted"
    assert promotion.json()["config"]["auto_promotion"] is False

    deployments = client.get(
        "/api/training/model-deployments",
        headers=admin_headers,
    )
    assert deployments.status_code == 200
    assert deployments.json()[0]["stage"] == "promoted"
