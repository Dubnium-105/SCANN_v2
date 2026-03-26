from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from scann.native_annotation.app import app


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"SIMPLE FITS PLACEHOLDER")


def _auth_headers(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/login",
        json={"username": "annotator", "password": "scann123"},
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_api_tasks_aggregates_triplet_paths(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"

    _touch(dataset_root / "new" / "PGC 17069.fts")
    _touch(dataset_root / "old" / "PGC 17069.fts")
    _touch(dataset_root / "new_marked" / "PGC 17069.fts")

    _touch(dataset_root / "new" / "PGC 35671.fts")
    _touch(dataset_root / "old" / "PGC 35671.fts")
    _touch(dataset_root / "new_marked" / "PGC 35671.fts")

    # This file should not be surfaced because it is missing a matching triplet.
    _touch(dataset_root / "old" / "ONLY_OLD.fts")

    monkeypatch.setenv("SCANN_NATIVE_DATASET_ROOT", str(dataset_root))
    client = TestClient(app)
    headers = _auth_headers(client)

    response = client.get("/api/tasks", headers=headers)

    assert response.status_code == 200
    assert response.json() == [
        {
            "task_id": "PGC 17069",
            "new_path": "new/PGC 17069.fts",
            "old_path": "old/PGC 17069.fts",
            "new_marked_path": "new_marked/PGC 17069.fts",
        },
        {
            "task_id": "PGC 35671",
            "new_path": "new/PGC 35671.fts",
            "old_path": "old/PGC 35671.fts",
            "new_marked_path": "new_marked/PGC 35671.fts",
        },
    ]
