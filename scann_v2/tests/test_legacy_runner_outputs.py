from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scann.experiments.legacy_runner import (
    _save_analysis_figure,
    _save_learning_curve_plot,
    _save_predictions_csv,
    _save_split_distribution_plot,
)


def test_legacy_runner_plot_and_prediction_outputs(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "summary": {
                    "split_counts": {
                        "train": {"real": 7, "bogus": 9},
                        "val": {"real": 2, "bogus": 2},
                        "test": {"real": 3, "bogus": 1},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    split_plot = tmp_path / "split.png"
    _save_split_distribution_plot(manifest_path, split_plot)
    assert split_plot.exists()
    assert split_plot.stat().st_size > 0

    history_plot = tmp_path / "history.png"
    _save_learning_curve_plot(
        [
            {
                "epoch": 1,
                "train_loss": 0.9,
                "val_loss": 0.8,
                "val_f1": 0.6,
                "val_f2": 0.7,
                "val_roc_auc": 0.75,
                "val_pr_auc": 0.72,
            },
            {
                "epoch": 2,
                "train_loss": 0.7,
                "val_loss": 0.6,
                "val_f1": 0.7,
                "val_f2": 0.78,
                "val_roc_auc": 0.81,
                "val_pr_auc": 0.8,
            },
        ],
        history_plot,
    )
    assert history_plot.exists()
    assert history_plot.stat().st_size > 0

    labels = np.array([0, 0, 1, 1], dtype=np.int32)
    probs = np.array([0.1, 0.35, 0.7, 0.95], dtype=np.float64)
    analysis_plot = tmp_path / "analysis.png"
    _save_analysis_figure(labels, probs, threshold=0.5, split_name="test", output_path=analysis_plot)
    assert analysis_plot.exists()
    assert analysis_plot.stat().st_size > 0

    prediction_csv = tmp_path / "predictions.csv"
    _save_predictions_csv(
        prediction_csv,
        [
            {"relative_path": "positive/a.png", "group_key": "a", "candidate_id": 1, "is_manual": False},
            {"relative_path": "negative/b.png", "group_key": "b", "candidate_id": 2, "is_manual": True},
            {"relative_path": "positive/c.png", "group_key": "c", "candidate_id": 3, "is_manual": False},
            {"relative_path": "positive/d.png", "group_key": "d", "candidate_id": 4, "is_manual": False},
        ],
        labels,
        probs,
        threshold=0.5,
        split_name="test",
    )
    content = prediction_csv.read_text(encoding="utf-8")
    assert "prob_real" in content
    assert "pred_label_name" in content
