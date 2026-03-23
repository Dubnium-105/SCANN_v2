"""Manual diagnostic for FITS new/old pairing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scann.data.file_manager import match_new_old_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect how SCANN matches FITS files between new/old folders.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "dataset",
        help="Dataset root containing 'new' and 'old' folders.",
    )
    parser.add_argument(
        "--new-name",
        default="20221227T214251__SAC NGC 3813.fts",
        help="Optional filename fragment to inspect in the matched pairs.",
    )
    parser.add_argument(
        "--old-name",
        default="20260308T215954__SAC NGC 3813.fts",
        help="Optional filename fragment to inspect in the matched pairs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset.resolve()
    new_dir = dataset_dir / "new"
    old_dir = dataset_dir / "old"

    if not new_dir.exists() or not old_dir.exists():
        print(f"Dataset folders not found under: {dataset_dir}")
        return 1

    print(f"Scanning new folder: {new_dir}")
    print(f"Scanning old folder: {old_dir}")

    pairs, only_new, only_old = match_new_old_pairs(str(new_dir), str(old_dir))

    print("\nPairing summary:")
    print(f"  matched : {len(pairs)}")
    print(f"  new only: {len(only_new)}")
    print(f"  old only: {len(only_old)}")

    print("\nSample pairs:")
    for index, pair in enumerate(pairs[:10], start=1):
        print(f"  {index}. {pair.name}")
        print(f"     new: {pair.new_path.name}")
        print(f"     old: {pair.old_path.name}")

    if only_new:
        print("\nSample new-only names:")
        for name in only_new[:5]:
            print(f"  - {name}")

    if only_old:
        print("\nSample old-only names:")
        for name in only_old[:5]:
            print(f"  - {name}")

    print("\nTarget pair check:")
    new_found = False
    old_found = False
    matched = False

    for pair in pairs:
        if args.new_name in pair.new_path.name:
            new_found = True
            print(f"  found new file: {pair.new_path.name}")
        if args.old_name in pair.old_path.name:
            old_found = True
            print(f"  found old file: {pair.old_path.name}")
        if args.new_name in pair.new_path.name and args.old_name in pair.old_path.name:
            matched = True
            print(f"  matched together as: {pair.name}")

    if not new_found:
        print(f"  new file fragment not found: {args.new_name}")
    if not old_found:
        print(f"  old file fragment not found: {args.old_name}")
    if not matched:
        print("  target files were not paired together.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
