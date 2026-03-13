import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
from unittest.mock import patch

from scann.ai.training_worker import TrainingWorker


def _write_triplet_png(path) -> None:
    image = np.zeros((80, 240), dtype=np.uint8)
    image[:, :80] = 64
    image[:, 80:160] = 128
    image[:, 160:] = 192
    Image.fromarray(image).save(path)


class TestTrainingWorkerDatasetParsing:
    def test_build_sample_pool_v1_from_dataset_root(self, tmp_path):
        positive_dir = tmp_path / "positive"
        negative_dir = tmp_path / "negative"
        positive_dir.mkdir()
        negative_dir.mkdir()
        _write_triplet_png(positive_dir / "pos.png")
        _write_triplet_png(negative_dir / "neg.png")

        worker = TrainingWorker({"dataset_dir": str(tmp_path), "dataset_format": "v1"})
        samples, sample_kind = worker._build_sample_pool()

        assert sample_kind == "file"
        assert len(samples) == 2
        assert {label for _, label in samples} == {0, 1}

    def test_collect_v2_samples_from_root_uses_annotations(self, tmp_path):
        new_dir = tmp_path / "new"
        old_dir = tmp_path / "old"
        new_dir.mkdir()
        old_dir.mkdir()

        (new_dir / "target.fts").write_bytes(b"new")
        (old_dir / "target.fts").write_bytes(b"old")

        annotations = {
            "images": [
                {
                    "id": "target",
                    "file_name": "target.fts",
                    "annotations": [
                        {
                            "x": 12,
                            "y": 14,
                            "width": 20,
                            "height": 24,
                            "detail_type": "asteroid",
                        },
                        {
                            "x": 42,
                            "y": 48,
                            "width": 18,
                            "height": 18,
                            "label": "bogus",
                        },
                    ],
                }
            ]
        }
        (tmp_path / "annotations.json").write_text(json.dumps(annotations), encoding="utf-8")

        new_image = np.linspace(0, 1, 128 * 128, dtype=np.float32).reshape(128, 128)
        old_image = np.flipud(new_image)

        def fake_read_fits(path):
            if str(path).endswith("new\\target.fts") or str(path).endswith("new/target.fts"):
                return SimpleNamespace(data=new_image)
            return SimpleNamespace(data=old_image)

        worker = TrainingWorker({"dataset_dir": str(tmp_path), "dataset_format": "v2"})
        with patch("scann.ai.training_worker.read_fits", side_effect=fake_read_fits):
            samples = worker._collect_v2_samples_from_root(tmp_path)

        assert len(samples) == 2
        assert {label for _, label in samples} == {0, 1}
        for triplet, _label in samples:
            assert triplet.shape == (3, 80, 80)
            assert triplet.dtype == np.float32
            assert np.all(triplet >= 0.0)
            assert np.all(triplet <= 1.0)