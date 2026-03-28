from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from scann.core.models import AlignResult
from scann.data.file_manager import match_new_old_pairs
from scann.services.dataset_preprocess_service import DatasetPreprocessService


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


def test_match_new_old_pairs_keeps_all_duplicate_normalized_names(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    new_dir = tmp_path / "new"
    old_dir = tmp_path / "old"

    _write_fits(new_dir / "20260115T203000__field_001.fits", rng.normal(size=(16, 16)).astype(np.float32), date_obs="2026-01-15T20:30:00")
    _write_fits(new_dir / "20260115T204500__field_001.fits", rng.normal(size=(16, 16)).astype(np.float32), date_obs="2026-01-15T20:45:00")
    _write_fits(old_dir / "20260115T203100__FW_field_001.fits", rng.normal(size=(16, 16)).astype(np.float32), date_obs="2026-01-15T20:31:00")
    _write_fits(old_dir / "20260115T204600__FW_field_001.fits", rng.normal(size=(16, 16)).astype(np.float32), date_obs="2026-01-15T20:46:00")

    pairs, only_new, only_old = match_new_old_pairs(str(new_dir), str(old_dir))

    assert len(pairs) == 2
    assert only_new == []
    assert only_old == []
    assert [pair.new_path.stem for pair in pairs] == [
        "20260115T203000__field_001",
        "20260115T204500__field_001",
    ]


def test_prepare_dataset_writes_reloadable_task_manifest(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    rng = np.random.default_rng(7)

    for name, date_obs in [
        ("field_001.fits", "2026-01-15T20:30:00"),
        ("field_002.fits", "2026-01-15T20:45:00"),
    ]:
        data = rng.normal(loc=1000, scale=20, size=(32, 32)).astype(np.float32)
        marked = data.copy()
        marked[15:17, :] = 4095.0
        marked[:, 15:17] = 4095.0
        _write_fits(dataset_root / "new" / name, data, date_obs=date_obs)
        _write_fits(dataset_root / "old" / name, np.roll(data, shift=1, axis=0), date_obs=date_obs)
        _write_fits(dataset_root / "new_marked" / name, marked, date_obs=date_obs)

    service = DatasetPreprocessService(align_fn=_fake_align)
    report = service.prepare_dataset(dataset_root)
    tasks = DatasetPreprocessService.load_task_manifest(dataset_root)

    assert report.task_count == 2
    assert len(tasks) == 2
    assert (dataset_root / "preprocessed_tasks.json").is_file()
    assert all(task.new_path.is_file() for task in tasks)
