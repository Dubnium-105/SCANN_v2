import json
from types import SimpleNamespace
import torch

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

    def test_collect_v2_samples_maps_disappeared_detail_types_to_bogus(self, tmp_path):
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
                            "detail_type": "disappeared_asteroid",
                        },
                        {
                            "x": 32,
                            "y": 36,
                            "width": 16,
                            "height": 18,
                            "detail_type": "disappeared_star",
                        },
                        {
                            "x": 52,
                            "y": 56,
                            "width": 18,
                            "height": 18,
                            "detail_type": "asteroid",
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

        labels = [label for _, label in samples]
        assert labels.count(0) == 2
        assert labels.count(1) == 1


class TestTrainingWorkerDetectionTask:
    def test_resolve_task_type_default_and_detection(self):
        worker_default = TrainingWorker({})
        assert worker_default._resolve_task_type() == "classification"

        worker_detection = TrainingWorker({"task_type": "detection"})
        assert worker_detection._resolve_task_type() == "detection"

    def test_compute_dense_detection_loss_returns_positive(self):
        pred_dense = torch.zeros((1, 5, 2, 2), dtype=torch.float32)
        target_heatmap = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        target_heatmap[0, 0, 0, 0] = 1.0
        target_bbox = torch.zeros((1, 4, 2, 2), dtype=torch.float32)
        target_bbox_mask = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        target_bbox_mask[0, 0, 0, 0] = 1.0

        total_loss, heatmap_loss, bbox_loss = TrainingWorker._compute_dense_detection_loss(
            pred_dense,
            target_heatmap,
            target_bbox,
            target_bbox_mask,
            heatmap_pos_weight=4.0,
            bbox_loss_weight=2.0,
        )

        assert total_loss.item() >= 0.0
        assert heatmap_loss.item() >= 0.0
        assert bbox_loss.item() >= 0.0

    def test_save_detection_checkpoint_contains_metadata(self, tmp_path):
        model = torch.nn.Conv2d(3, 5, kernel_size=1)
        output_path = tmp_path / "dense_ckpt.pth"

        TrainingWorker._save_detection_checkpoint(
            model=model,
            save_path=str(output_path),
            best_epoch=2,
            best_val_loss=0.123,
            heatmap_threshold=0.35,
            bbox_loss_weight=2.0,
            heatmap_pos_weight=4.0,
            patch_size=16,
        )

        ckpt = torch.load(str(output_path), map_location="cpu")
        assert ckpt["task_type"] == "detection"
        assert ckpt["model_format"] == "v2_detector"
        assert ckpt["heatmap_threshold"] == 0.35
        assert ckpt["bbox_loss_weight"] == 2.0
        assert ckpt["patch_size"] == 16
