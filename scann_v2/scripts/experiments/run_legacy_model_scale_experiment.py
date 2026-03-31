"""CLI entry point for legacy v1 Exp-7 model-scale comparison."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scann.experiments.legacy_runner import run_legacy_model_scale_comparison


def main() -> int:
    default_config = PROJECT_ROOT / "experiments" / "configs" / "legacy_swin_t_pretrained_gpu.json"
    parser = argparse.ArgumentParser(description="Run legacy v1 Exp-7 model-scale comparison")
    parser.add_argument(
        "--config",
        default=str(default_config),
        help="Base json/yaml experiment config. Defaults to the pretrained Swin-T baseline.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output csv path for the comparison table.",
    )
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Retrain even if a summary json for that scale variant already exists.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    summary = run_legacy_model_scale_comparison(
        args.config,
        comparison_csv_path=args.output,
        skip_existing=not args.rerun_existing,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
