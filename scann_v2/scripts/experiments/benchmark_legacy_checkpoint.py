"""CLI entry point for benchmarking a saved legacy checkpoint, optionally with config overrides."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scann.experiments import benchmark_legacy_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark a saved legacy checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path produced by train_legacy_classifier")
    parser.add_argument("--config", default=None, help="Optional experiment config override for inference benchmarking")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to benchmark")
    parser.add_argument("--device", default="cpu", help="Benchmark device, for example cpu or cuda")
    parser.add_argument("--threshold", type=float, default=None, help="Optional decision threshold override")
    args = parser.parse_args()

    metrics = benchmark_legacy_checkpoint(
        args.checkpoint,
        config_override=args.config,
        split=args.split,
        device=args.device,
        threshold=args.threshold,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
