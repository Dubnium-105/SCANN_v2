from __future__ import annotations

import json

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
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _seed_partition_dataset(dataset_root) -> list[str]:
    storage = DatasetStorage(dataset_root)
    storage.ensure_schema()
    assets = []
    tasks = []
    task_ids = []
    for index in range(1, 13):
        date_token = f"202602{index:02d}T120000"
        field_key = f"field-{index % 3}"
        task_id = f"{date_token}__{field_key}"
        asset_id = f"asset-{index}"
        relpath = f"new/{task_id}.fts"
        task_ids.append(task_id)
        assets.append(
            RawAssetRecord(
                asset_id=asset_id,
                asset_role="new",
                field_key=field_key,
                field_name=field_key,
                capture_key=task_id.lower(),
                relpath=relpath,
                file_name=f"{task_id}.fts",
                file_stem=task_id,
                suffix=".fts",
                date_obs=f"2026-02-{index:02d}T12:00:00Z",
            )
        )
        tasks.append(
            TaskRecord(
                task_id=task_id,
                field_key=field_key,
                field_name=field_key,
                capture_key=task_id.lower(),
                new_asset_id=asset_id,
                preprocess_status="ready",
            )
        )

    storage.upsert_raw_assets(assets)
    storage.sync_tasks(tasks)
    for index, task_id in enumerate(task_ids):
        detail_type = (
            "supernova"
            if index == 0
            else ("asteroid" if index % 2 else "noise")
        )
        legacy_label = "real" if detail_type in {"supernova", "asteroid"} else "bogus"
        storage.upsert_current_annotation(
            task_id=task_id,
            source_view="new",
            label=legacy_label,
            detail_type=detail_type,
            ai_suggestion=None,
            ai_confidence=None,
            annotations=[
                {
                    "x": 10,
                    "y": 12,
                    "width": 8,
                    "height": 8,
                    "label": legacy_label,
                    "detail_type": detail_type,
                    "confidence": 1.0,
                }
            ],
            annotation_origin="online",
        )
    return task_ids


def test_partition_api_freezes_gold_test_and_snapshot_excludes_it(
    tmp_path,
    monkeypatch,
):
    dataset_root = tmp_path / "dataset"
    monkeypatch.setenv("SCANN_NATIVE_DATASET_ROOT", str(dataset_root))
    _seed_partition_dataset(dataset_root)
    client = TestClient(app)
    headers = _auth_headers(client)

    create = client.post(
        "/api/dataset-partitions",
        json={
            "partition_name": "gold-v1",
            "seed": 17,
            "activate": True,
        },
        headers=headers,
    )
    assert create.status_code == 200, create.text
    partition = create.json()
    assert partition["is_active"] is True
    assert partition["task_count"] == 12
    assert partition["train_task_count"] > 0
    assert partition["validation_task_count"] > 0
    assert partition["test_task_count"] > 0

    repeated = client.post(
        "/api/dataset-partitions",
        json={
            "partition_name": "gold-v1",
            "seed": 17,
            "activate": True,
        },
        headers=headers,
    )
    assert repeated.status_code == 200
    assert repeated.json()["partition_id"] == partition["partition_id"]

    listed = client.get("/api/dataset-partitions", headers=headers)
    assert listed.status_code == 200
    assert len(listed.json()) == 1

    manifest_path = dataset_root / partition["manifest_relpath"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    test_ids = {
        item["task_id"]
        for item in manifest["splits"]["test"]
    }

    snapshot = client.post(
        "/api/training/snapshots",
        json={"snapshot_name": "partitioned-training"},
        headers=headers,
    )
    assert snapshot.status_code == 200, snapshot.text
    snapshot_payload = snapshot.json()
    assert snapshot_payload["metadata"]["partition_id"] == partition["partition_id"]
    assert snapshot_payload["metadata"]["partition_splits"] == [
        "train",
        "validation",
    ]
    snapshot_document = json.loads(
        (dataset_root / snapshot_payload["document_relpath"]).read_text(
            encoding="utf-8"
        )
    )
    snapshot_ids = {
        item["id"]
        for item in snapshot_document["images"]
    }
    assert snapshot_document["version"] == "3.0"
    assert snapshot_document["taxonomy_version"] == "scann-discovery-v1"
    assert not (snapshot_ids & test_ids)
    assert len(snapshot_ids) == (
        partition["train_task_count"]
        + partition["validation_task_count"]
    )

    leaked = client.post(
        "/api/training/snapshots",
        json={
            "snapshot_name": "must-fail",
            "task_ids": [next(iter(test_ids))],
        },
        headers=headers,
    )
    assert leaked.status_code == 400
    assert "gold-test" in leaked.json()["detail"]
