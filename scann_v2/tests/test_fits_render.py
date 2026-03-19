from __future__ import annotations

import numpy as np
from astropy.io import fits
from fastapi.testclient import TestClient

from scann.native_annotation.app import app


def test_api_render_returns_png(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    new_dir = dataset_root / "new"
    new_dir.mkdir(parents=True, exist_ok=True)

    fits_path = new_dir / "test_target.fts"
    data = np.linspace(0, 1000, 64 * 64, dtype=np.float32).reshape(64, 64)
    fits.writeto(str(fits_path), data, overwrite=True)

    monkeypatch.setenv("SCANN_NATIVE_DATASET_ROOT", str(dataset_root))
    client = TestClient(app)

    response = client.get("/api/render/new/test_target.fts")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert response.content.startswith(b"\x89PNG\r\n\x1a\n")
