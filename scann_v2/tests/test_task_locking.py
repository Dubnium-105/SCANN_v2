from __future__ import annotations

from pathlib import Path

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


def test_claim_next_task_locks_task_between_clients(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"

    _touch(dataset_root / "new" / "PGC 17069.fts")
    _touch(dataset_root / "old" / "PGC 17069.fts")
    _touch(dataset_root / "new_marked" / "PGC 17069.fts")

    monkeypatch.setenv("SCANN_NATIVE_DATASET_ROOT", str(dataset_root))
    client = TestClient(app)

    response_a = client.get("/api/tasks/next", params={"client_id": "client-a"})
    response_b = client.get("/api/tasks/next", params={"client_id": "client-b"})

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

    claim_a = client.get("/api/tasks/next", params={"client_id": "client-a"})
    assert claim_a.status_code == 200

    save_resp = client.post(
        "/api/annotations/PGC 17069",
        params={"client_id": "client-a"},
        json=_annotation_payload(),
    )
    assert save_resp.status_code == 200

    claim_b = client.get("/api/tasks/next", params={"client_id": "client-b"})
    assert claim_b.status_code == 200
    assert claim_b.json()["task_id"] == "PGC 17069"
