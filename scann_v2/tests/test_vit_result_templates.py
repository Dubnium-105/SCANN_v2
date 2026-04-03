from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scann.experiments import legacy_runner


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.reader(file_obj)
        return next(reader)


def test_vit_attention_benchmark_template_matches_summary_columns():
    template_path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "results"
        / "vit_attention_benchmark_template.csv"
    )

    header = _read_header(template_path)

    assert header == legacy_runner.SUMMARY_COLUMNS


def test_vit_attention_ablation_template_matches_ablation_columns():
    template_path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "results"
        / "vit_attention_ablation_template.csv"
    )

    header = _read_header(template_path)

    assert header == legacy_runner.VIT_ATTENTION_ABLATION_COLUMNS
