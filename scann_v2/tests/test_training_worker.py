import json
from typing import Optional
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
                            "detail_type": "noise",
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

    def test_collect_v2_samples_respects_manual_crop_metadata(self, tmp_path):
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
                    "metadata": {
                        "manual_crop": {
                            "x": 0,
                            "y": 0,
                            "width": 64,
                            "height": 64,
                        }
                    },
                    "annotations": [
                        {
                            "x": 10,
                            "y": 10,
                            "width": 12,
                            "height": 12,
                            "detail_type": "asteroid",
                        },
                        {
                            "x": 96,
                            "y": 96,
                            "width": 12,
                            "height": 12,
                            "detail_type": "noise",
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

        assert len(samples) == 1
        assert samples[0][1] == 1

    def test_collect_v2_samples_uses_snapshot_paths_even_without_directory_pair_matches(self, tmp_path):
        new_dir = tmp_path / "new"
        old_dir = tmp_path / "old"
        new_dir.mkdir()
        old_dir.mkdir()

        new_file = new_dir / "aligned_task_001.fts"
        old_file = old_dir / "cropped_task_001.fts"
        new_file.write_bytes(b"new")
        old_file.write_bytes(b"old")

        annotations = {
            "images": [
                {
                    "id": "task-001",
                    "paths": {
                        "new": "new/aligned_task_001.fts",
                        "old": "old/cropped_task_001.fts",
                    },
                    "annotations": [
                        {
                            "x": 12,
                            "y": 14,
                            "width": 20,
                            "height": 24,
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
            if str(path).endswith("aligned_task_001.fts"):
                return SimpleNamespace(data=new_image)
            return SimpleNamespace(data=old_image)

        worker = TrainingWorker({"dataset_dir": str(tmp_path), "dataset_format": "v2"})
        with patch("scann.ai.training_worker.match_new_old_pairs", return_value=([], [], [])):
            with patch("scann.ai.training_worker.read_fits", side_effect=fake_read_fits):
                samples = worker._collect_v2_samples_from_root(tmp_path)

        assert len(samples) == 1
        assert samples[0][1] == 1


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

    def test_detection_training_uses_snapshot_annotation_file(self, tmp_path, monkeypatch):
        snapshot_path = tmp_path / "snapshot.json"
        snapshot_path.write_text(json.dumps({"version": "2.3", "images": []}), encoding="utf-8")

        captured: dict[str, object] = {}

        class _FakeDenseDataset:
            def __init__(self, dataset_root: str, annotation_file: Optional[str] = None, patch_size: int = 16):
                captured["dataset_root"] = dataset_root
                captured["annotation_file"] = annotation_file
                captured["patch_size"] = patch_size

            def __len__(self) -> int:
                return 1

        monkeypatch.setattr("scann.ai.dataset.FitsDenseDetectionDataset", _FakeDenseDataset)

        worker = TrainingWorker(
            {
                "dataset_dir": str(tmp_path),
                "dataset_format": "v2",
                "task_type": "detection",
                "annotations_document_path": str(snapshot_path),
                "epochs": 1,
                "batch_size": 1,
                "lr": 1e-3,
                "backbone": "ResNet18",
                "save_path": str(tmp_path / "dense-best.pth"),
            }
        )
        errors: list[str] = []
        worker.error.connect(errors.append)

        worker.run()

        assert captured["dataset_root"] == str(tmp_path)
        assert captured["annotation_file"] == str(snapshot_path)
        assert errors
