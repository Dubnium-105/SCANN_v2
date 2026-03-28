from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from scann.native_annotation.app import app
from scann.core.fits_annotation_storage import load_v2_annotation_document


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"SIMPLE FITS PLACEHOLDER")


def _payload(bucket: str) -> dict:
    return {
        "bucket": bucket,
        "source_view": "new",
        "metadata": {"annotator": "tester"},
        "annotations": [
            {"x": 10.5, "y": 20.5, "width": 30.0, "height": 40.0, "label": "Positive"},
            {"x": 100.0, "y": 120.0, "width": 10.0, "height": 12.0, "label": "Artifact"},
        ],
    }


def _auth_headers(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/login",
        json={"username": "annotator", "password": "scann123"},
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_api_annotations_save_to_positive_and_negative(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    _touch(dataset_root / "new" / "PGC 17069.fts")
    _touch(dataset_root / "old" / "PGC 17069.fts")
    monkeypatch.setenv("SCANN_NATIVE_DATASET_ROOT", str(dataset_root))

    client = TestClient(app)
    headers = _auth_headers(client)

    resp_positive = client.post("/api/annotations/PGC 17069", json=_payload("positive"), headers=headers)
    resp_negative = client.post("/api/annotations/PGC 17069", json=_payload("negative"), headers=headers)

    assert resp_positive.status_code == 200
    assert resp_negative.status_code == 200

    dataset_db_file = dataset_root / "scann_dataset.db"

    assert dataset_db_file.exists()
    assert resp_positive.json()["saved_path"] == "scann_dataset.db"
    assert resp_negative.json()["saved_path"] == "scann_dataset.db"

    annotations_doc = load_v2_annotation_document(dataset_root)
    images = annotations_doc.get("images", [])
    assert len(images) == 1

    saved_image = images[0]
    assert saved_image["id"] == "PGC 17069"
    assert saved_image["source_view"] == "new"
    assert len(saved_image["annotations"]) == 2
