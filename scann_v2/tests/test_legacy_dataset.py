from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scann.experiments.legacy_dataset import LegacyTripletExperimentDataset


def _write_semantic_triplet_png(path: Path, *, diff: int, new: int, old: int) -> None:
    image = np.zeros((80, 240), dtype=np.uint8)
    image[:, :80] = diff
    image[:, 80:160] = new
    image[:, 160:] = old
    Image.fromarray(image).save(path)


@pytest.fixture
def legacy_manifest_doc(tmp_path):
    positive = tmp_path / "positive"
    positive.mkdir()
    sample_path = positive / "20260313_REAL_PGC 1111.fts2_cand1_S1.2.png"
    _write_semantic_triplet_png(sample_path, diff=10, new=120, old=250)

    return {
        "dataset_root": str(tmp_path),
        "entries": [
            {
                "relative_path": "positive/20260313_REAL_PGC 1111.fts2_cand1_S1.2.png",
                "bucket": "positive",
                "label": 1,
                "label_name": "real",
                "group_key": "pgc 1111.fts2",
                "split": "train",
            }
        ],
    }


def test_dataset_reorders_channels_by_semantic_input_mode(legacy_manifest_doc):
    dataset = LegacyTripletExperimentDataset(
        legacy_manifest_doc,
        split="train",
        image_size=80,
        resize_mode="keep",
        normalize=False,
        augment=False,
        input_mode="new_old_diff",
    )

    x, y = dataset[0]
    assert y.item() == 1
    assert x.shape == (3, 80, 80)
    assert x[0].mean().item() == pytest.approx(120 / 255.0, rel=1e-4)
    assert x[1].mean().item() == pytest.approx(250 / 255.0, rel=1e-4)
    assert x[2].mean().item() == pytest.approx(10 / 255.0, rel=1e-4)


def test_dataset_supports_stored_order_and_diff_only_modes(legacy_manifest_doc):
    stored_dataset = LegacyTripletExperimentDataset(
        legacy_manifest_doc,
        split="train",
        image_size=80,
        resize_mode="keep",
        normalize=False,
        augment=False,
        input_mode="stored_triplet",
    )
    stored_x, _ = stored_dataset[0]
    assert stored_x[0].mean().item() == pytest.approx(10 / 255.0, rel=1e-4)
    assert stored_x[1].mean().item() == pytest.approx(120 / 255.0, rel=1e-4)
    assert stored_x[2].mean().item() == pytest.approx(250 / 255.0, rel=1e-4)

    diff_only_dataset = LegacyTripletExperimentDataset(
        legacy_manifest_doc,
        split="train",
        image_size=80,
        resize_mode="keep",
        normalize=False,
        augment=False,
        input_mode="diff_only",
    )
    diff_only_x, _ = diff_only_dataset[0]
    assert diff_only_x[0].mean().item() == pytest.approx(10 / 255.0, rel=1e-4)
    assert diff_only_x[1].mean().item() == pytest.approx(10 / 255.0, rel=1e-4)
    assert diff_only_x[2].mean().item() == pytest.approx(10 / 255.0, rel=1e-4)
