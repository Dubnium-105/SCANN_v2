from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PyQt5.QtWidgets import QWidget

from scann.core.models import AppConfig, Candidate


@pytest.fixture
def fits_dataset(tmp_dir: Path) -> Path:
    try:
        from astropy.io import fits as astro_fits
    except ImportError:
        pytest.skip("astropy not installed")

    ds = tmp_dir / "fits_data"
    new_dir = ds / "new"
    old_dir = ds / "old"
    new_dir.mkdir(parents=True)
    old_dir.mkdir(parents=True)

    rng = np.random.default_rng(7)
    hdr = astro_fits.Header()
    hdr["OBJECT"] = "DialogField"
    hdr["DATE-OBS"] = "2026-01-15T20:30:00"

    for i in range(2):
        data = rng.normal(loc=1000, scale=50, size=(128, 128)).astype(np.uint16)
        astro_fits.writeto(
            str(new_dir / f"field_{i + 1:03d}.fits"),
            data,
            header=hdr,
            overwrite=True,
        )
        astro_fits.writeto(
            str(old_dir / f"field_{i + 1:03d}.fits"),
            np.roll(data, shift=(1, -1), axis=(0, 1)),
            header=hdr,
            overwrite=True,
        )

    return ds


def test_ai_prelabel_button_batches_v2_samples_and_persists_results(qapp, fits_dataset: Path):
    from scann.gui.dialogs.annotation_dialog import AnnotationDialog
    from scann.core.models import AlignResult

    class FakePipeline:
        def __init__(self, *args, **kwargs):
            self.patch_size = kwargs["patch_size"]
            self.inference_engine = kwargs["inference_engine"]

        def process_pair(self, pair_name, new_data, old_data, **kwargs):
            assert pair_name.startswith("field_")
            assert new_data.shape == old_data.shape == (128, 128)
            return SimpleNamespace(
                candidates=[Candidate(x=48, y=64, ai_score=0.91)],
                error="",
            )

    parent = QWidget()
    parent._inference_engine = SimpleNamespace(is_ready=True, is_v1=False, threshold=0.5)
    parent._show_message = Mock()

    config = AppConfig()
    config.slice_size = 20

    dialog = AnnotationDialog(parent=parent, config=config)
    dialog.set_mode("v2")

    def fake_align(new_data, old_data, method="auto", max_shift=None):
        return AlignResult(
            aligned_old=old_data.astype(np.float32),
            dx=0.0,
            dy=0.0,
            success=True,
        )

    with patch("scann.core.fits_annotation_backend.align", fake_align):
        dialog.load_dataset(str(fits_dataset))

    assert all(sample.display_name.endswith("__aligned_crop.fts") for sample in dialog._samples)

    with patch("scann.gui.dialogs.annotation_dialog.DetectionPipeline", FakePipeline):
        dialog._btn_ai_prelabel.click()

    reloaded_backend = dialog._backend.__class__()
    reloaded_samples = reloaded_backend.load_samples(str(fits_dataset))

    for sample in reloaded_samples:
        assert sample.ai_suggestion == "real"
        assert sample.ai_confidence == pytest.approx(0.91)
        assert len(sample.bboxes) == 1
        bbox = sample.bboxes[0]
        assert bbox.x == 38
        assert bbox.y == 54
        assert bbox.width == 20
        assert bbox.height == 20
        assert bbox.confidence == pytest.approx(0.91)

    parent._show_message.assert_called()
    dialog.close()
    parent.close()
    dialog.deleteLater()
    parent.deleteLater()
    qapp.processEvents()