"""CLI entry point for legacy v1 classifier experiments."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scann.experiments.legacy_runner import train_legacy_classifier


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a legacy v1 triplet classifier from config")
    parser.add_argument("--config", required=True, help="Path to a json/yaml experiment config")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    summary = train_legacy_classifier(args.config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
