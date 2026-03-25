from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PyQt5.QtWidgets import QWidget

from scann.core.models import AppConfig


@pytest.fixture
def triplet_dataset(tmp_dir: Path) -> Path:
    ds = tmp_dir / "triplet_dataset"
    (ds / "positive").mkdir(parents=True)
    (ds / "negative").mkdir(parents=True)
    unlabeled = ds / "unlabeled"
    unlabeled.mkdir(parents=True)

    rng = np.random.default_rng(11)
    for i in range(2):
        image = Image.fromarray(rng.integers(0, 255, size=(80, 240), dtype=np.uint8))
        image.save(unlabeled / f"img_{i + 1:03d}.png")

    return ds


def test_triplet_preview_displays_visible_ai_suggestion(qapp):
    from scann.gui.widgets.triplet_preview import TripletPreviewPanel

    panel = TripletPreviewPanel()

    panel.set_file_info("img_001.png")
    panel.set_ai_suggestion("real", 0.91)

    assert panel._ai_hint_label.isHidden() is False
    assert "AI 建议" in panel._ai_hint_label.text()
    assert "A.真" in panel._ai_hint_label.text()
    assert "91.0%" in panel._ai_hint_label.text()
    assert "img_001.png" in panel.toolTip()

    panel.deleteLater()
    qapp.processEvents()


def test_triplet_preview_clear_hides_ai_suggestion(qapp):
    from scann.gui.widgets.triplet_preview import TripletPreviewPanel

    panel = TripletPreviewPanel()
    panel.set_ai_suggestion("bogus", 0.73)

    panel.clear_ai_suggestion()

    assert panel._ai_hint_label.isHidden() is True
    assert panel._ai_hint_label.text() == ""
    assert "AI 建议" not in panel.toolTip()

    panel.deleteLater()
    qapp.processEvents()


def test_annotation_dialog_v1_clears_stale_ai_suggestion(qapp, triplet_dataset: Path):
    from scann.gui.dialogs.annotation_dialog import AnnotationDialog

    parent = QWidget()
    dialog = AnnotationDialog(parent=parent, config=AppConfig())
    dialog.set_mode("v1")
    dialog.load_dataset(str(triplet_dataset))

    dialog._samples[0].ai_suggestion = "real"
    dialog._samples[0].ai_confidence = 0.88
    dialog._current_index = 0
    dialog._update_display()

    assert dialog._triplet_preview._ai_hint_label.isHidden() is False
    assert "88.0%" in dialog._triplet_preview._ai_hint_label.text()

    dialog._current_index = 1
    dialog._update_display()

    assert dialog._triplet_preview._ai_hint_label.isHidden() is True
    assert dialog._triplet_preview._ai_hint_label.text() == ""

    dialog.close()
    parent.close()
    dialog.deleteLater()
    parent.deleteLater()
    qapp.processEvents()
