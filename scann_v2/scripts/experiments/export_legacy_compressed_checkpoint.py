"""Export one or more portable compressed legacy checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scann.experiments import (
    export_legacy_compressed_checkpoint,
    export_legacy_compressed_checkpoint_variants,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export compressed legacy checkpoint variants")
    parser.add_argument("--checkpoint", required=True, help="Dense legacy checkpoint to compress")
    parser.add_argument(
        "--mode",
        default="all",
        choices=["all", "custom_int8_weight_only", "custom_int4_weight_only", "packed_int4_weight_only"],
        help="Compression mode to export. Use 'all' to emit both supported variants.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path for single-mode export, or output directory when --mode=all.",
    )
    args = parser.parse_args()

    if args.mode == "all":
        summary = export_legacy_compressed_checkpoint_variants(
            args.checkpoint,
            output_dir=args.output,
        )
    else:
        summary = export_legacy_compressed_checkpoint(
            args.checkpoint,
            checkpoint_compression_mode=args.mode,
            output_path=args.output,
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
