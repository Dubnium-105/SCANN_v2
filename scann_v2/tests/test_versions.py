from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from scann.native_annotation.app import app


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
