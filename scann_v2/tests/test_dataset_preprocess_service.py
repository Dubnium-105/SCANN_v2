from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from scann.core.models import AlignResult
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


@pytest.fixture
def raw_dataset(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "dataset"
    rng = np.random.default_rng(21)
    date_obs = "2026-01-15T20:30:00"

    for idx in range(2):
        data = rng.normal(loc=1000, scale=20, size=(64, 64)).astype(np.float32)
        marked = data.copy()
        marked[31:33, :] = 4095.0
        marked[:, 31:33] = 4095.0

        name = f"field_{idx + 1:03d}.fits"
        _write_fits(dataset_root / "new" / name, data, date_obs=date_obs)
        _write_fits(dataset_root / "old" / name, np.roll(data, shift=(1, -1), axis=(0, 1)), date_obs=date_obs)
        _write_fits(dataset_root / "new_marked" / name, marked, date_obs=date_obs)

    return dataset_root


def _fake_align(new_data, old_data, method="auto", max_shift=None):
    return AlignResult(
        aligned_old=old_data.astype(np.float32),
        dx=0.0,
        dy=0.0,
        success=True,
    )


def test_prepare_dataset_standardizes_aligns_and_collects_tasks(raw_dataset: Path) -> None:
    service = DatasetPreprocessService(align_fn=_fake_align)

    report = service.prepare_dataset(raw_dataset)
    tasks = service.collect_preprocessed_tasks(raw_dataset)

    assert report.standardized_files == 6
    assert report.generated_aligned_pairs == 2
    assert report.generated_marked_crops == 2
    assert report.task_count == 2

    assert (raw_dataset / "dataset_raw" / "new" / "field_001.fits").exists()
    assert (raw_dataset / "dataset_raw" / "old" / "field_001.fits").exists()
    assert (raw_dataset / "dataset_raw" / "new_marked" / "field_001.fits").exists()
    assert (raw_dataset / "new" / "20260115T203000__field_001__aligned_crop.fts").exists()
    assert (raw_dataset / "old" / "20260115T203000__field_001__aligned_crop.fts").exists()
    assert (raw_dataset / "new_marked" / "20260115T203000__field_001__aligned_crop.fts").exists()

    assert [task.task_id for task in tasks] == [
        "20260115T203000__field_001",
        "20260115T203000__field_002",
    ]


def test_prepare_dataset_is_idempotent_after_first_run(raw_dataset: Path) -> None:
    service = DatasetPreprocessService(align_fn=_fake_align)

    first = service.prepare_dataset(raw_dataset)
    second = service.prepare_dataset(raw_dataset)

    assert first.generated_aligned_pairs == 2
    assert second.standardized_files == 0
    assert second.generated_aligned_pairs == 0
    assert second.generated_marked_crops == 0
    assert second.reused_aligned_pairs == 2
    assert second.task_count == 2


def test_prepare_dataset_aligns_pairs_in_parallel_when_multiple_workers_enabled(raw_dataset: Path) -> None:
    barrier = threading.Barrier(2, timeout=5)
    seen_threads: set[int] = set()
    seen_threads_lock = threading.Lock()

    def _parallel_align(new_data, old_data, method="auto", max_shift=None):
        with seen_threads_lock:
            seen_threads.add(threading.get_ident())
        barrier.wait()
        return AlignResult(
            aligned_old=old_data.astype(np.float32),
            dx=0.0,
            dy=0.0,
            success=True,
        )

    service = DatasetPreprocessService(
        align_fn=_parallel_align,
        max_workers=2,
    )

    report = service.prepare_dataset(raw_dataset)

    assert report.generated_aligned_pairs == 2
    assert len(seen_threads) == 2
