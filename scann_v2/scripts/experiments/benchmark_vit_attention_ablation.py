"""CLI entry point for running the ViT packed-KV attention ablation matrix on a saved checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scann.experiments import run_vit_attention_ablation


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ViT attention compression ablation matrix")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path produced by train_legacy_classifier")
    parser.add_argument("--config", default=None, help="Optional base config override for the ablation matrix")
    parser.add_argument("--comparison-csv", default=None, help="Optional CSV output path")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to benchmark")
    parser.add_argument("--device", default="cpu", help="Benchmark device, for example cpu or cuda")
    parser.add_argument("--threshold", type=float, default=None, help="Optional decision threshold override")
    args = parser.parse_args()

    summary = run_vit_attention_ablation(
        args.checkpoint,
        base_config=args.config,
        comparison_csv_path=args.comparison_csv,
        split=args.split,
        device=args.device,
        threshold=args.threshold,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
