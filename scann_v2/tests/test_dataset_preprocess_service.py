from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scann.core.dataset_storage import DatasetStorage
from scann.core.models import AlignResult
from scann.services.dataset_preprocess_service import DatasetPreprocessService


def _write_fits(path: Path, data: np.ndarray, *, date_obs: str | None = None) -> None:
    try:
        from astropy.io import fits as astro_fits
    except ImportError:
        pytest.skip("astropy not installed")

    path.parent.mkdir(parents=True, exist_ok=True)
    header = astro_fits.Header()
    if date_obs is not None:
        header["DATE-OBS"] = date_obs
    astro_fits.writeto(str(path), data, header=header, overwrite=True)


def _fake_align(new_data, old_data, method="auto", max_shift=None):
    return AlignResult(
        aligned_old=old_data.astype(np.float32),
        dx=0.0,
        dy=0.0,
        success=True,
    )


@pytest.fixture
def dataset_raw_root(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "dataset"
    rng = np.random.default_rng(21)

    for idx in range(2):
        date_obs = f"2026-01-15T20:{30 + idx:02d}:00"
        data = rng.normal(loc=1000, scale=20, size=(64, 64)).astype(np.float32)
        marked = data.copy()
        marked[31:33, :] = 4095.0
        marked[:, 31:33] = 4095.0
        name = f"field_{idx + 1:03d}.fits"

        _write_fits(dataset_root / "dataset_raw" / "new" / name, data, date_obs=date_obs)
        _write_fits(
            dataset_root / "dataset_raw" / "old" / name,
            np.roll(data, shift=(1, -1), axis=(0, 1)),
            date_obs=date_obs,
        )
        _write_fits(dataset_root / "dataset_raw" / "new_marked" / name, marked, date_obs=date_obs)

    return dataset_root


def test_prepare_dataset_from_dataset_raw_creates_db_tasks_and_artifacts(dataset_raw_root: Path) -> None:
    service = DatasetPreprocessService(align_fn=_fake_align)

    report = service.prepare_dataset(dataset_raw_root)
    tasks = service.collect_preprocessed_tasks(dataset_raw_root)
    storage = DatasetStorage(dataset_raw_root)

    assert report.standardized_files == 6
    assert report.generated_aligned_pairs == 2
    assert report.generated_marked_crops == 2
    assert report.task_count == 2

    assert (dataset_raw_root / "scann_dataset.db").exists()
    assert [task.task_id for task in tasks] == [
        "20260115T203000__field_001",
        "20260115T203100__field_002",
    ]
    assert all(task.new_path.exists() for task in tasks)
    assert all(task.old_path is not None and task.old_path.exists() for task in tasks)

    db_tasks = storage.list_tasks()
    assert len(db_tasks) == 2
    assert all(task.preprocess_status == "ready" for task in db_tasks)


def test_prepare_dataset_is_idempotent_for_dataset_raw_inputs(dataset_raw_root: Path) -> None:
    service = DatasetPreprocessService(align_fn=_fake_align)

    first = service.prepare_dataset(dataset_raw_root)
    second = service.prepare_dataset(dataset_raw_root)

    assert first.generated_aligned_pairs == 2
    assert second.standardized_files == 6
    assert second.generated_aligned_pairs == 0
    assert second.generated_marked_crops == 0
    assert second.reused_aligned_pairs == 2
    assert second.task_count == 2


def test_prepare_dataset_supports_date_typed_new_and_marked_raw_dirs(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    data = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
    marked = data + 1000.0

    _write_fits(
        dataset_root / "dataset_raw" / "old" / "FW_NGC 1003.fit",
        data.copy(),
        date_obs="2026-02-01T20:00:00",
    )
    _write_fits(
        dataset_root / "dataset_raw" / "20260216_new" / "FW_NGC 1003.fit",
        data,
        date_obs="2026-02-16T21:00:00",
    )
    _write_fits(
        dataset_root / "dataset_raw" / "20260216_mark" / "FW_NGC 1003.fit",
        marked,
        date_obs="2026-02-16T21:00:00",
    )

    service = DatasetPreprocessService(align_fn=_fake_align)
    report = service.prepare_dataset(dataset_root)
    tasks = service.collect_preprocessed_tasks(dataset_root)
    storage = DatasetStorage(dataset_root)

    assert report.standardized_files == 3
    assert report.generated_aligned_pairs == 1
    assert report.generated_marked_crops == 1
    assert [task.task_id for task in tasks] == ["20260216T210000__NGC 1003"]
    assert tasks[0].new_path.parent == dataset_root / "new"
    assert tasks[0].old_path is not None and tasks[0].old_path.parent == dataset_root / "old"
    assert tasks[0].new_marked_path is not None
    assert tasks[0].new_marked_path.parent == dataset_root / "new_marked"

    new_assets = storage.list_raw_assets("new")
    marked_assets = storage.list_raw_assets("new_marked")
    assert [asset.relpath for asset in new_assets] == [
        "dataset_raw/20260216_new/FW_NGC 1003.fit",
    ]
    assert [asset.relpath for asset in marked_assets] == [
        "dataset_raw/20260216_mark/FW_NGC 1003.fit",
    ]


def test_prepare_dataset_scopes_date_typed_marked_inputs_by_date(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    data = np.arange(24 * 24, dtype=np.float32).reshape(24, 24)

    _write_fits(dataset_root / "dataset_raw" / "old" / "field_001.fits", data.copy())
    _write_fits(dataset_root / "dataset_raw" / "20260216_new" / "field_001.fits", data)
    _write_fits(dataset_root / "dataset_raw" / "20260217_new" / "field_001.fits", data + 10.0)
    _write_fits(dataset_root / "dataset_raw" / "20260217_marked" / "field_001.fits", data + 500.0)

    service = DatasetPreprocessService(align_fn=_fake_align)
    report = service.prepare_dataset(dataset_root)
    storage = DatasetStorage(dataset_root)
    db_tasks = storage.list_tasks()
    raw_assets = {
        asset.asset_id: asset
        for role in ("new", "old", "new_marked")
        for asset in storage.list_raw_assets(role)
    }

    assert report.task_count == 2
    assert report.generated_aligned_pairs == 2
    assert report.generated_marked_crops == 1
    assert [task.task_id for task in db_tasks] == [
        "20260216__field_001",
        "20260217__field_001",
    ]

    first, second = db_tasks
    assert "dataset_raw/20260216_new/" in raw_assets[first.new_asset_id].relpath
    assert first.new_marked_asset_id is None
    assert "dataset_raw/20260217_new/" in raw_assets[second.new_asset_id].relpath
    assert second.new_marked_asset_id is not None
    assert "dataset_raw/20260217_marked/" in raw_assets[second.new_marked_asset_id].relpath


def test_prepare_dataset_reuses_single_old_asset_for_multiple_new_tasks(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    rng = np.random.default_rng(7)

    base = rng.normal(loc=1000, scale=10, size=(48, 48)).astype(np.float32)
    marked = base.copy()
    marked[23:25, :] = 4095.0
    marked[:, 23:25] = 4095.0

    _write_fits(
        dataset_root / "dataset_raw" / "old" / "field_001.fits",
        np.roll(base, shift=1, axis=0),
        date_obs="2026-01-15T20:29:00",
    )
    _write_fits(
        dataset_root / "dataset_raw" / "new" / "field_001.fits",
        base,
        date_obs="2026-01-15T20:30:00",
    )
    _write_fits(
        dataset_root / "dataset_raw" / "new" / "field_001 (2).fits",
        base + 5,
        date_obs="2026-01-15T20:45:00",
    )
    _write_fits(
        dataset_root / "dataset_raw" / "new_marked" / "field_001.fits",
        marked,
        date_obs="2026-01-15T20:30:00",
    )
    _write_fits(
        dataset_root / "dataset_raw" / "new_marked" / "field_001 (2).fits",
        marked + 3,
        date_obs="2026-01-15T20:45:00",
    )

    service = DatasetPreprocessService(align_fn=_fake_align)
    report = service.prepare_dataset(dataset_root)
    storage = DatasetStorage(dataset_root)
    tasks = storage.list_tasks()

    assert report.task_count == 2
    assert len(tasks) == 2
    assert len({task.new_asset_id for task in tasks}) == 2
    assert len({task.old_asset_id for task in tasks if task.old_asset_id is not None}) == 1
    assert len({task.new_marked_asset_id for task in tasks if task.new_marked_asset_id is not None}) == 2

    prepared = storage.list_prepared_task_paths()
    assert len(prepared) == 2
    assert [item["task_id"] for item in prepared] == [
        "20260115T203000__field_001",
        "20260115T204500__field_001",
    ]


def test_prepare_dataset_aligns_marked_new_to_new_before_cropping(tmp_path: Path) -> None:
    try:
        from astropy.io import fits as astro_fits
    except ImportError:
        pytest.skip("astropy not installed")

    dataset_root = tmp_path / "dataset"
    new_data = np.arange(8 * 8, dtype=np.float32).reshape(8, 8)
    marked_aligned = new_data + 500.0
    marked_rotated = np.rot90(marked_aligned, 2)

    _write_fits(
        dataset_root / "dataset_raw" / "new" / "field_001.fits",
        new_data,
        date_obs="2026-01-15T20:30:00",
    )
    _write_fits(
        dataset_root / "dataset_raw" / "old" / "field_001.fits",
        new_data.copy(),
        date_obs="2026-01-15T20:29:00",
    )
    _write_fits(
        dataset_root / "dataset_raw" / "new_marked" / "field_001.fits",
        marked_rotated,
        date_obs="2026-01-15T20:30:00",
    )

    align_inputs: list[np.ndarray] = []

    def _align_with_marked_step(reference, moving, method="auto", max_shift=None):
        align_inputs.append(np.asarray(moving, dtype=np.float32))
        if np.array_equal(moving, marked_aligned):
            return AlignResult(
                aligned_old=marked_aligned,
                dx=0.0,
                dy=0.0,
                success=True,
            )
        return AlignResult(
            aligned_old=np.asarray(moving, dtype=np.float32),
            dx=0.0,
            dy=0.0,
            success=not np.array_equal(moving, marked_rotated),
            error_message="alignment failed" if np.array_equal(moving, marked_rotated) else "",
        )

    service = DatasetPreprocessService(align_fn=_align_with_marked_step)
    report = service.prepare_dataset(dataset_root)
    tasks = service.collect_preprocessed_tasks(dataset_root)

    assert report.generated_marked_crops == 1
    assert len(tasks) == 1
    assert tasks[0].new_marked_path is not None

    with astro_fits.open(tasks[0].new_path, memmap=False) as hdul:
        generated_new = np.asarray(hdul[0].data, dtype=np.float32)
    with astro_fits.open(tasks[0].new_marked_path, memmap=False) as hdul:
        generated = np.asarray(hdul[0].data, dtype=np.float32)

    assert len(align_inputs) == 2
    np.testing.assert_array_equal(align_inputs[1], marked_aligned)
    np.testing.assert_array_equal(generated, generated_new + 500.0)
