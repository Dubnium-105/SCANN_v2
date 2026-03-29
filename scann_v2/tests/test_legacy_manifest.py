from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scann.experiments.legacy_manifest import build_legacy_triplet_manifest


def _write_triplet_png(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    image = np.zeros((80, 240), dtype=np.uint8)
    image[:, :80] = rng.integers(0, 40, size=(80, 80), dtype=np.uint8)
    image[:, 80:160] = rng.integers(80, 160, size=(80, 80), dtype=np.uint8)
    image[:, 160:] = rng.integers(180, 255, size=(80, 80), dtype=np.uint8)
    Image.fromarray(image).save(path)


def test_build_legacy_manifest_keeps_same_group_in_single_split(tmp_path):
    positive = tmp_path / "positive"
    negative = tmp_path / "negative"
    positive.mkdir()
    negative.mkdir()

    file_specs = [
        ("positive", "20260313_REAL_PGC 1111.fts2_cand1_S1.2.png"),
        ("negative", "20260313_BOGUS_PGC 1111.fts2_cand2_S1.1.png"),
        ("positive", "20260313_REAL_PGC 2222.fts4_cand1_S1.0.png"),
        ("negative", "20260313_BOGUS_PGC 3333.fts5_cand1_S1.0.png"),
        ("positive", "20260313_REAL_MANUAL_PGC 4444.fts6_cand3_S9.9.png"),
        ("negative", "20260313_BOGUS_PGC 5555.fts8_cand1_S1.0.png"),
    ]
    for index, (bucket, file_name) in enumerate(file_specs):
        target_dir = positive if bucket == "positive" else negative
        _write_triplet_png(target_dir / file_name, seed=100 + index)

    manifest = build_legacy_triplet_manifest(tmp_path, seed=7)

    group_to_split: dict[str, str] = {}
    for entry in manifest["entries"]:
        group_key = entry["group_key"]
        split_name = entry["split"]
        previous = group_to_split.setdefault(group_key, split_name)
        assert previous == split_name

    split_counts = manifest["summary"]["split_counts"]
    assert split_counts["train"]["total"] > split_counts["val"]["total"]
    assert split_counts["train"]["total"] > split_counts["test"]["total"]
    assert manifest["summary"]["group_overlap_detected"] is False
    assert manifest["summary"]["total_samples"] == len(file_specs)
    assert manifest["summary"]["label_counts"] == {"real": 3, "bogus": 3}
