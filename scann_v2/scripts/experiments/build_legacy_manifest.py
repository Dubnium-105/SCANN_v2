"""CLI entry point for building a fixed legacy v1 manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scann.experiments.legacy_manifest import build_legacy_triplet_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a fixed split manifest for the v1 triplet dataset")
    parser.add_argument("--dataset-dir", required=True, help="Dataset root that contains positive/negative")
    parser.add_argument("--output", required=True, help="Output manifest json path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for grouped split assignment")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio")
    args = parser.parse_args()

    manifest = build_legacy_triplet_manifest(
        args.dataset_dir,
        args.output,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    summary = manifest["summary"]
    print(
        "manifest written to",
        args.output,
        f"| total={summary['total_samples']}",
        f"| groups={summary['group_count']}",
        f"| real={summary['label_counts']['real']}",
        f"| bogus={summary['label_counts']['bogus']}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
