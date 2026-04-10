from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from scann.core.dataset_storage import DatasetStorage
from scann.core.fits_annotation_storage import load_v2_annotation_document
from scann.native_annotation.app import app
from scann.native_annotation.annotation_service import AnnotationBox, AnnotationService


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"SIMPLE FITS PLACEHOLDER")


def _auth_headers(client: TestClient, username: str, password: str) -> dict[str, str]:
    response = client.post(
        "/api/login",
        json={"username": username, "password": password},
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _payload(label: str) -> dict:
    return {
        "bucket": "positive",
        "source_view": "new",
        "metadata": {"annotator": label},
        "annotations": [
            {"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0, "label": label},
        ],
    }


def test_annotation_history_and_revision_detail(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    _touch(dataset_root / "new" / "PGC 17069.fts")

    monkeypatch.setenv("SCANN_NATIVE_DATASET_ROOT", str(dataset_root))
    client = TestClient(app)

    headers_a = _auth_headers(client, username="annotator", password="scann123")
    headers_b = _auth_headers(client, username="admin", password="admin123")

    save_a = client.post(
        "/api/annotations/PGC 17069",
        json=_payload("A-label"),
        headers=headers_a,
    )
    save_b = client.post(
        "/api/annotations/PGC 17069",
        json=_payload("B-label"),
        headers=headers_b,
    )

    assert save_a.status_code == 200
    assert save_b.status_code == 200

    history = client.get("/api/annotations/PGC 17069/history", headers=headers_a)
    assert history.status_code == 200

    history_data = history.json()
    assert history_data["task_id"] == "PGC 17069"
    assert len(history_data["revisions"]) == 2
    assert history_data["revisions"][0]["submitted_by"] == "admin"
    assert history_data["revisions"][1]["submitted_by"] == "annotator"
    assert history_data["revisions"][0]["change_summary"]["modified"] == 1

    latest_revision_id = history_data["revisions"][0]["revision_id"]
    latest_revision = client.get(
        f"/api/annotations/PGC 17069/history/{latest_revision_id}",
        headers=headers_a,
    )
    assert latest_revision.status_code == 200

    revision_data = latest_revision.json()
    assert revision_data["revision_id"] == latest_revision_id
    assert revision_data["submitted_by"] == "admin"
    assert revision_data["annotations"][0]["label"] == "B-label"
    assert revision_data["parent_revision_id"] is not None
    assert revision_data["change_summary"]["modified"] == 1
    assert len(revision_data["changed_items"]) == 1
    assert revision_data["changed_items"][0]["change_type"] == "modified"
    assert "label" in revision_data["changed_items"][0]["changed_fields"]


def test_admin_only_rollback_and_creates_new_revision(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    _touch(dataset_root / "new" / "PGC 17069.fts")

    monkeypatch.setenv("SCANN_NATIVE_DATASET_ROOT", str(dataset_root))
    client = TestClient(app)

    headers_annotator = _auth_headers(client, username="annotator", password="scann123")
    headers_admin = _auth_headers(client, username="admin", password="admin123")

    client.post("/api/annotations/PGC 17069", json=_payload("A-label"), headers=headers_annotator)
    client.post("/api/annotations/PGC 17069", json=_payload("B-label"), headers=headers_admin)

    history = client.get("/api/annotations/PGC 17069/history", headers=headers_admin)
    assert history.status_code == 200
    history_data = history.json()
    older_revision_id = history_data["revisions"][1]["revision_id"]

    forbidden_rollback = client.post(
        f"/api/annotations/PGC 17069/rollback/{older_revision_id}",
        headers=headers_annotator,
    )
    assert forbidden_rollback.status_code == 403

    rollback = client.post(
        f"/api/annotations/PGC 17069/rollback/{older_revision_id}",
        headers=headers_admin,
    )
    assert rollback.status_code == 200
    rollback_data = rollback.json()
    assert rollback_data["rolled_back_to_revision_id"] == older_revision_id

    refreshed_history = client.get("/api/annotations/PGC 17069/history", headers=headers_admin)
    assert refreshed_history.status_code == 200
    refreshed_data = refreshed_history.json()
    assert len(refreshed_data["revisions"]) == 3
    assert refreshed_data["revisions"][0]["submitted_by"] == "admin"
    assert refreshed_data["revisions"][0]["rollback_of_revision_id"] == older_revision_id

    db_path = dataset_root / "scann_dataset.db"
    assert db_path.exists()

    storage = DatasetStorage(dataset_root)
    revisions = storage.list_annotation_revisions("PGC 17069")
    document = load_v2_annotation_document(dataset_root)

    assert len(revisions) == 3
    images = document.get("images", [])
    assert any(item.get("id") == "PGC 17069" for item in images if isinstance(item, dict))


def test_annotation_history_accepts_uploaded_aligned_crop_task_id(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    suffixed_task_id = "20260203T134946__NGC 918__aligned_crop"
    canonical_task_id = "20260203T134946__NGC 918"

    _touch(dataset_root / "new" / f"{suffixed_task_id}.fts")
    _touch(dataset_root / "old" / f"{suffixed_task_id}.fts")

    monkeypatch.setenv("SCANN_NATIVE_DATASET_ROOT", str(dataset_root))
    client = TestClient(app)
    headers = _auth_headers(client, username="annotator", password="scann123")

    save_response = client.post(
        f"/api/annotations/{suffixed_task_id}",
        json=_payload("A-label"),
        headers=headers,
    )
    assert save_response.status_code == 200
    assert save_response.json()["task_id"] == canonical_task_id

    history_response = client.get(f"/api/annotations/{suffixed_task_id}/history", headers=headers)
    assert history_response.status_code == 200
    assert history_response.json()["task_id"] == canonical_task_id
    assert len(history_response.json()["revisions"]) == 1


def test_large_annotation_diff_avoids_all_pairs_matching(tmp_path, monkeypatch) -> None:
    service = AnnotationService(tmp_path)
    before = [
        AnnotationBox(x=float(index * 20), y=10.0, width=6.0, height=6.0, label="real")
        for index in range(1200)
    ]
    after = [AnnotationBox.model_validate(item.model_dump()) for item in before]
    after[777] = AnnotationBox(x=before[777].x, y=before[777].y, width=6.0, height=6.0, label="bogus")

    iou_calls = 0
    original_iou = service._bbox_iou

    def counted_iou(left: AnnotationBox, right: AnnotationBox) -> float:
        nonlocal iou_calls
        iou_calls += 1
        return original_iou(left, right)

    monkeypatch.setattr(service, "_bbox_iou", counted_iou)

    summary, changed_items = service._build_diff(before, after)

    assert summary.added == 0
    assert summary.removed == 0
    assert summary.modified == 1
    assert len(changed_items) == 1
    assert changed_items[0].changed_fields == ["label"]
    assert iou_calls < 50
