from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scann.experiments import legacy_runner


def test_run_legacy_input_fusion_comparison_reuses_existing_and_writes_csv(tmp_path, monkeypatch):
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
        "checkpoint_path": str(output_root / "checkpoints" / "legacy_vit_b16_pretrained_gpu_best.pt"),
    }
    (results_dir / "legacy_vit_b16_pretrained_gpu_summary.json").write_text(
        json.dumps(existing_summary),
        encoding="utf-8",
    )

    train_calls: list[str] = []

    def _fake_train(config):
        train_calls.append(config["input_mode"])
        score_map = {
            "diff_only": 0.81,
            "diff_new": 0.89,
            "diff_old": 0.87,
        }
        experiment_name = config["experiment_name"]
        return {
            "experiment_name": experiment_name,
            "model_name": "vit_b_16",
            "input_mode": config["input_mode"],
            "pretrained": True,
            "seed": 42,
            "image_size": 224,
            "resize_mode": "resize",
            "batch_size": 16,
            "epochs_requested": 100,
            "epochs_ran": 12,
            "best_epoch": 8,
            "best_threshold": 0.5,
            "val_accuracy": 0.9,
            "val_recall": 0.9,
            "val_f1": score_map[config["input_mode"]],
            "val_roc_auc": 0.95,
            "test_accuracy": 0.9,
            "test_recall": 0.9,
            "test_f1": score_map[config["input_mode"]],
            "test_roc_auc": 0.95,
            "checkpoint_path": str(output_root / "checkpoints" / f"{experiment_name}_best.pt"),
        }

    monkeypatch.setattr(legacy_runner, "train_legacy_classifier", _fake_train)

    summary = legacy_runner.run_legacy_input_fusion_comparison(
        {
            "experiment_name": "legacy_vit_b16_pretrained_gpu",
            "output_root": str(output_root),
            "model_name": "vit_b_16",
            "pretrained": True,
            "input_mode": "new_old_diff",
            "image_size": 224,
            "resize_mode": "resize",
            "batch_size": 16,
            "epochs": 100,
            "seed": 42,
        }
    )

    assert train_calls == ["diff_only", "diff_new", "diff_old"]
    assert summary["best_input_mode"] == "new_old_diff"
    csv_path = Path(summary["comparison_csv_path"])
    assert csv_path.exists()

    with csv_path.open("r", encoding="utf-8", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))

    assert len(rows) == 4
    assert rows[0]["input_mode"] == "new_old_diff"
    assert rows[0]["result_source"] == "reused"
    assert {row["input_mode"] for row in rows} == {"new_old_diff", "diff_only", "diff_new", "diff_old"}
