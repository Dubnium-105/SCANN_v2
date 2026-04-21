import json
from types import SimpleNamespace
import torch
import pytest

import numpy as np
from PIL import Image
from unittest.mock import patch

from scann.ai.class_balance import build_class_audit, merge_imbalance_config
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
        with pytest.raises(ValueError):
            worker._build_sample_pool()

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
        assert {label for _, label in samples} == {0, 4}
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
        assert labels.count(8) == 1
        assert labels.count(9) == 1
        assert labels.count(0) == 1

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
        assert samples[0][1] == 0

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
        assert samples[0][1] == 0


class TestTrainingWorkerDetectionTask:
    def test_resolve_task_type_default_and_detection(self):
        worker_default = TrainingWorker({})
        assert worker_default._resolve_task_type() == "classification"

        worker_detection = TrainingWorker({"task_type": "detection"})
        assert worker_detection._resolve_task_type() == "classification"

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

    def test_frozen_feature_training_smoke_writes_metadata(self, tmp_path):
        records = []
        for index in range(3):
            records.append(
                {
                    "task_id": f"tail-{index}",
                    "annotation_index": 0,
                    "detail_type": "asteroid",
                    "label": 0,
                    "data": np.full((3, 8 + index, 8 + index), 0.2 + index * 0.01, dtype=np.float32),
                    "quality_score": 1.0,
                }
            )
        for index in range(12):
            records.append(
                {
                    "task_id": f"head-{index}",
                    "annotation_index": 0,
                    "detail_type": "corresponding",
                    "label": 7,
                    "data": np.full((3, 10 + (index % 3), 10 + (index % 3)), 0.7 + index * 0.001, dtype=np.float32),
                    "quality_score": 1.0,
                }
            )

        train_idx = list(range(2)) + list(range(3, 13))
        val_idx = [2, 13, 14]
        split_support = {
            "train": [2, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0],
            "val": [1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
        }
        class_support = build_class_audit(records, split_support=split_support)
        output_path = tmp_path / "frozen_feature.pth"
        config = merge_imbalance_config(
            {
                "feature_encoder": "scann_test_identity",
                "variance_transfer": {
                    "enabled": True,
                    "synthetic_per_tail": 3,
                    "tail_max_support": 2,
                    "donor_min_support": 10,
                    "covariance_mode": "full",
                },
                "prior_logit_correction": {"enabled": True, "tau": 1.0},
            }
        )
        worker = TrainingWorker(
            {
                "feature_encoder": "scann_test_identity",
                "epochs": 2,
                "batch_size": 4,
                "lr": 0.01,
                "seed": 4,
            }
        )

        worker._run_frozen_feature_training(
            dataset_root=tmp_path,
            all_samples=records,
            train_idx=train_idx,
            val_idx=val_idx,
            split_support=split_support,
            class_support=class_support,
            imbalance_config=config,
            device=torch.device("cpu"),
            epochs=2,
            batch_size=4,
            lr=0.01,
            save_path=str(output_path),
            backbone_name="ViT_B_16",
        )

        ckpt = torch.load(output_path, map_location="cpu", weights_only=False)
        assert ckpt["model_format"] == "frozen_feature_classifier"
        assert ckpt["feature_encoder"] == "scann_test_identity"
        assert ckpt["variance_transfer_summary"]["synthetic_counts"]["asteroid"] == 3
        assert ckpt["class_log_prior"][1] == 0.0
        assert "macro_f1_supported" in ckpt["best_metrics"]
        assert "tail_recall@1" in ckpt["best_metrics"]
        assert "macro_ap" in ckpt["best_metrics"]
        assert "long_tail_score" in ckpt["best_metrics"]
        assert "selection_score" in ckpt["best_metrics"]

    def test_long_tail_selection_score_and_log_summary_are_deduplicated(self):
        metrics = {
            "macro_f1_supported": 0.10,
            "tail_recall@1": 0.40,
            "macro_ap": 0.20,
        }
        config = merge_imbalance_config(
            {
                "selection_metric": "long_tail_score",
                "selection_metric_weights": {
                    "macro_f1_supported": 0.5,
                    "tail_recall@1": 0.3,
                    "macro_ap": 0.2,
                },
            }
        )

        score = TrainingWorker._feature_selection_value(metrics, "long_tail_score", config)
        summary = TrainingWorker._format_feature_metric_summary(
            metrics,
            leading=("tail_recall@1", metrics["tail_recall@1"]),
        )

        assert score == pytest.approx(0.21)
        assert metrics["long_tail_score"] == pytest.approx(0.21)
        assert summary.count("tail_recall@1") == 1

    def test_selection_constraints_penalize_epochs_below_tail_floor(self):
        metrics = {
            "macro_f1_supported": 0.22,
            "tail_recall@1": 0.05,
            "macro_ap": 0.26,
        }
        config = merge_imbalance_config(
            {
                "selection_metric": "macro_f1_supported",
                "selection_constraints": {"tail_recall@1": 0.15},
            }
        )

        score = TrainingWorker._feature_selection_value(metrics, "macro_f1_supported", config)

        assert score < 0.0
        assert metrics["selection_raw_score"] == pytest.approx(0.22)
        assert metrics["selection_constraint_penalty"] == pytest.approx(0.10)
        assert metrics["selection_constraints_met"] is False

    def test_training_uses_snapshot_annotation_document(self, tmp_path):
        snapshot_path = tmp_path / "snapshot.json"
        snapshot_path.write_text(
            json.dumps(
                {
                    "version": "2.3",
                    "images": [
                        {
                            "id": "from-snapshot",
                            "annotations": [
                                {"x": 1, "y": 2, "width": 3, "height": 4, "detail_type": "asteroid"}
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        worker = TrainingWorker(
            {
                "dataset_dir": str(tmp_path),
                "dataset_format": "v2",
                "task_type": "classification",
                "annotations_document_path": str(snapshot_path),
            }
        )

        payload = worker._load_v2_annotations_document(tmp_path)

        assert payload["images"][0]["id"] == "from-snapshot"
