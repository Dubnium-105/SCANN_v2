"""CLI entry point for legacy v1 Exp-8 quantization smoke validation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scann.experiments.legacy_runner import run_legacy_quantization_smoke


def main() -> int:
    default_config = PROJECT_ROOT / "experiments" / "configs" / "legacy_swin_t_pretrained_gpu.json"
    parser = argparse.ArgumentParser(description="Run legacy v1 Exp-8 quantization smoke validation")
    parser.add_argument(
        "--config",
        default=str(default_config),
        help="Base json/yaml experiment config. Defaults to the pretrained Swin-T baseline.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path. Defaults to <output_root>/checkpoints/<experiment_name>_best.pt.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output csv path. Defaults to experiments/results/quantization_smoke_results.csv.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    summary = run_legacy_quantization_smoke(
        args.config,
        checkpoint_path=args.checkpoint,
        comparison_csv_path=args.output,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
