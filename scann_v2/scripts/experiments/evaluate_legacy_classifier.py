"""CLI entry point for evaluating a saved legacy v1 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scann.experiments.legacy_runner import evaluate_legacy_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a saved legacy v1 classifier checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path produced by train_legacy_classifier")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--manifest", default=None, help="Optional manifest override")
    parser.add_argument("--dataset-dir", default=None, help="Optional dataset root override")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size override")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional num_workers override")
    parser.add_argument("--device", default=None, help="Optional device override")
    args = parser.parse_args()

    metrics = evaluate_legacy_checkpoint(
        args.checkpoint,
        split=args.split,
        manifest_path=args.manifest,
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
