from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from scann.native_annotation.app import app
from scann.core.models import AlignResult


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"SIMPLE FITS PLACEHOLDER")


def _write_fits(path: Path, data: np.ndarray, *, date_obs: str) -> None:
    try:
        from astropy.io import fits as astro_fits
    except ImportError:
        pytest.skip("astropy not installed")

    path.parent.mkdir(parents=True, exist_ok=True)
    header = astro_fits.Header()
    header["DATE-OBS"] = date_obs
    astro_fits.writeto(str(path), data, header=header, overwrite=True)


def _fake_align(new_data, old_data, method="auto", max_shift=None):
    return AlignResult(
        aligned_old=old_data.astype(np.float32),
        dx=0.0,
        dy=0.0,
        success=True,
    )


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


def test_api_dataset_preprocess_exposes_aligned_tasks(tmp_path, monkeypatch) -> None:
    from scann.native_annotation import routes as native_routes
    from scann.native_annotation.dataset_service import DatasetService
    from scann.services.dataset_preprocess_service import DatasetPreprocessService

    dataset_root = tmp_path / "dataset"
    rng = np.random.default_rng(12)
    date_obs = "2026-01-15T20:30:00"

    data = rng.normal(loc=1000, scale=10, size=(48, 48)).astype(np.float32)
    marked = data.copy()
    marked[23:25, :] = 4095.0
    marked[:, 23:25] = 4095.0

    _write_fits(dataset_root / "new" / "field_001.fits", data, date_obs=date_obs)
    _write_fits(dataset_root / "old" / "field_001.fits", np.roll(data, shift=1, axis=0), date_obs=date_obs)
    _write_fits(dataset_root / "new_marked" / "field_001.fits", marked, date_obs=date_obs)

    monkeypatch.setenv("SCANN_NATIVE_DATASET_ROOT", str(dataset_root))
    monkeypatch.setattr(
        native_routes,
        "get_dataset_preprocess_service",
        lambda: DatasetPreprocessService(align_fn=_fake_align),
    )
    monkeypatch.setattr(
        native_routes,
        "get_dataset_service",
        lambda: DatasetService(
            dataset_root=dataset_root,
            preprocess_service=DatasetPreprocessService(align_fn=_fake_align),
        ),
    )
    client = TestClient(app)
    headers = _auth_headers(client)

    preprocess = client.post("/api/dataset/preprocess", headers=headers)

    assert preprocess.status_code == 200
    payload = preprocess.json()
    assert payload["standardized_files"] == 3
    assert payload["generated_aligned_pairs"] == 1
    assert payload["generated_marked_crops"] == 1
    assert payload["task_count"] == 1

    tasks = client.get("/api/tasks", headers=headers)

    assert tasks.status_code == 200
    assert tasks.json() == [
        {
            "task_id": "20260115T203000__field_001",
            "new_path": "new/20260115T203000__field_001__aligned_crop.fts",
            "old_path": "old/20260115T203000__field_001__aligned_crop.fts",
            "new_marked_path": "new_marked/20260115T203000__field_001__aligned_crop.fts",
        }
    ]
