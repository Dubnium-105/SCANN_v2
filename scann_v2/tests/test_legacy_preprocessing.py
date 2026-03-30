from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scann.experiments import legacy_runner


def test_create_experiment_model_supports_custom_vit_image_size():
    model = legacy_runner.create_experiment_model("vit_b_16", pretrained=False, image_size=80)
    assert model.image_size == 80


def test_run_legacy_preprocessing_comparison_reuses_existing_and_writes_csv(tmp_path, monkeypatch):
    output_root = tmp_path / "experiments"
    results_dir = output_root / "results"
    results_dir.mkdir(parents=True)

    existing_summary = {
        "experiment_name": "legacy_vit_b16_pretrained_gpu",
        "model_name": "vit_b_16",
        "input_mode": "new_old_diff",
        "pretrained": True,
        "seed": 42,
        "image_size": 224,
        "resize_mode": "resize",
        "normalize": True,
        "batch_size": 16,
        "epochs_requested": 100,
        "epochs_ran": 25,
        "best_epoch": 15,
        "best_threshold": 0.75,
        "val_accuracy": 0.95,
        "val_recall": 0.96,
        "val_f1": 0.95,
        "val_roc_auc": 0.99,
        "test_accuracy": 0.95,
        "test_recall": 0.96,
        "test_f1": 0.94,
        "test_roc_auc": 0.99,
        "avg_epoch_seconds": 11.2,
        "peak_gpu_memory_mb": 2222.0,
        "checkpoint_path": str(output_root / "checkpoints" / "legacy_vit_b16_pretrained_gpu_best.pt"),
    }
    (results_dir / "legacy_vit_b16_pretrained_gpu_summary.json").write_text(
        json.dumps(existing_summary),
        encoding="utf-8",
    )

    train_calls: list[tuple[int, str, bool]] = []

    def _fake_train(config):
        train_calls.append((config["image_size"], config["resize_mode"], config["normalize"]))
        score_map = {
            (80, "keep", True): 0.90,
            (224, "pad_resize", True): 0.91,
            (224, "resize", False): 0.82,
        }
        experiment_name = config["experiment_name"]
        return {
            "experiment_name": experiment_name,
            "model_name": "vit_b_16",
            "input_mode": "new_old_diff",
            "pretrained": True,
            "seed": 42,
            "image_size": config["image_size"],
            "resize_mode": config["resize_mode"],
            "normalize": config["normalize"],
            "batch_size": 16,
            "epochs_requested": 100,
            "epochs_ran": 12,
            "best_epoch": 8,
            "best_threshold": 0.5,
            "val_accuracy": 0.9,
            "val_recall": 0.9,
            "val_f1": score_map[(config["image_size"], config["resize_mode"], config["normalize"])],
            "val_roc_auc": 0.95,
            "test_accuracy": 0.9,
            "test_recall": 0.9,
            "test_f1": score_map[(config["image_size"], config["resize_mode"], config["normalize"])],
            "test_roc_auc": 0.95,
            "avg_epoch_seconds": 10.0 + float(config["image_size"]) / 100.0,
            "peak_gpu_memory_mb": 1000.0 + float(config["image_size"]),
            "checkpoint_path": str(output_root / "checkpoints" / f"{experiment_name}_best.pt"),
        }

    monkeypatch.setattr(legacy_runner, "train_legacy_classifier", _fake_train)

    summary = legacy_runner.run_legacy_preprocessing_comparison(
        {
            "experiment_name": "legacy_vit_b16_pretrained_gpu",
            "output_root": str(output_root),
            "model_name": "vit_b_16",
            "pretrained": True,
            "input_mode": "new_old_diff",
            "image_size": 224,
            "resize_mode": "resize",
            "normalize": True,
            "batch_size": 16,
            "epochs": 100,
            "seed": 42,
        }
    )

    assert train_calls == [
        (80, "keep", True),
        (224, "pad_resize", True),
        (224, "resize", False),
    ]
    assert summary["best_variant"] == "resize_224"
    assert summary["fastest_variant"] == "native_keep_80"
    assert summary["lowest_memory_variant"] == "native_keep_80"
    csv_path = Path(summary["comparison_csv_path"])
    assert csv_path.exists()

    with csv_path.open("r", encoding="utf-8", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))

    assert len(rows) == 4
    assert {row["variant"] for row in rows} == {
        "native_keep_80",
        "resize_224",
        "pad_resize_224",
        "resize_224_no_norm",
    }
    by_variant = {row["variant"]: row for row in rows}
    assert by_variant["resize_224"]["result_source"] == "reused"
    assert by_variant["resize_224"]["normalize"] == "True"
