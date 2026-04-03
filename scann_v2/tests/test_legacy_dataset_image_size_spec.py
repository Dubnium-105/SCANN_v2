from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scann.experiments.legacy_dataset import LegacyTripletExperimentDataset
from scann.experiments import legacy_runner


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


def test_load_experiment_config_normalizes_image_size_variants():
    resized = legacy_runner.load_experiment_config(
        {
            "experiment_name": "fullres_resize",
            "image_size": (96, 128),
            "resize_mode": "resize",
        }
    )
    kept = legacy_runner.load_experiment_config(
        {
            "experiment_name": "keep_native",
            "image_size": "keep",
            "resize_mode": "keep",
        }
    )

    assert resized.image_size == [96, 128]
    assert kept.image_size == "keep"


def test_dataset_supports_non_square_resize_shape(legacy_manifest_doc):
    dataset = LegacyTripletExperimentDataset(
        legacy_manifest_doc,
        split="train",
        image_size=[96, 128],
        resize_mode="resize",
        normalize=False,
        augment=False,
        input_mode="new_old_diff",
    )

    x, _ = dataset[0]
    assert x.shape == (3, 96, 128)


def test_dataset_supports_keep_image_size_string(legacy_manifest_doc):
    dataset = LegacyTripletExperimentDataset(
        legacy_manifest_doc,
        split="train",
        image_size="keep",
        resize_mode="keep",
        normalize=False,
        augment=False,
        input_mode="new_old_diff",
    )

    x, _ = dataset[0]
    assert x.shape == (3, 80, 80)


def test_vit_model_factory_accepts_square_list_and_rejects_non_square():
    model = legacy_runner.create_experiment_model(
        "vit_b_16",
        pretrained=False,
        image_size=[80, 80],
    )
    assert model.image_size == 80

    with pytest.raises(ValueError, match="square image_size"):
        legacy_runner.create_experiment_model(
            "vit_b_16",
            pretrained=False,
            image_size=[80, 96],
        )
