"""Build fixed manifests for the legacy v1 triplet PNG dataset."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SPLIT_NAMES = ("train", "val", "test")

_DATE_PREFIX_RE = re.compile(r"^(?P<date>\d{8}|\d{4}-\d{2}-\d{2})[-_]")
_LABEL_PREFIX_RE = re.compile(r"^(?P<label>real|bogus)[-_]", re.IGNORECASE)
_MANUAL_PREFIX_RE = re.compile(r"^manual[-_]", re.IGNORECASE)
_CANDIDATE_RE = re.compile(r"[_-]cand(?P<candidate>\d+)\b", re.IGNORECASE)


@dataclass(frozen=True)
class LegacyTripletSample:
    """One labeled legacy triplet sample."""

    relative_path: str
    bucket: str
    label: int
    label_name: str
    file_name: str
    file_stem: str
    source_name: str
    group_key: str
    candidate_id: int | None
    is_manual: bool
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "bucket": self.bucket,
            "label": self.label,
            "label_name": self.label_name,
            "file_name": self.file_name,
            "file_stem": self.file_stem,
            "source_name": self.source_name,
            "group_key": self.group_key,
            "candidate_id": self.candidate_id,
            "is_manual": self.is_manual,
            "sha256": self.sha256,
        }


@dataclass
class _GroupedSamples:
    key: str
    samples: list[LegacyTripletSample]

    @property
    def total(self) -> int:
        return len(self.samples)

    @property
    def positive(self) -> int:
        return sum(1 for sample in self.samples if sample.label == 1)

    @property
    def negative(self) -> int:
        return sum(1 for sample in self.samples if sample.label == 0)


def _normalize_group_key(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value).strip().lower()
    normalized = normalized.rstrip("_- ")
    return normalized


def parse_legacy_triplet_name(file_name: str) -> dict[str, Any]:
    """Extract source and grouping metadata from a v1 file name."""

    working = Path(file_name).stem.strip()

    date_match = _DATE_PREFIX_RE.match(working)
    if date_match:
        working = working[date_match.end() :]

    label_name: str | None = None
    label_match = _LABEL_PREFIX_RE.match(working)
    if label_match:
        label_name = label_match.group("label").lower()
        working = working[label_match.end() :]

    is_manual = False
    manual_match = _MANUAL_PREFIX_RE.match(working)
    if manual_match:
        is_manual = True
        working = working[manual_match.end() :]

    candidate_id: int | None = None
    source_name = working
    candidate_match = _CANDIDATE_RE.search(working)
    if candidate_match:
        candidate_id = int(candidate_match.group("candidate"))
        source_name = working[: candidate_match.start()]

    source_name = source_name.rstrip("_- ").strip()
    group_key = _normalize_group_key(source_name or working or Path(file_name).stem)

    return {
        "label_name_from_file": label_name,
        "source_name": source_name or Path(file_name).stem,
        "group_key": group_key or _normalize_group_key(Path(file_name).stem),
        "candidate_id": candidate_id,
        "is_manual": is_manual,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_legacy_triplet_dataset(dataset_root: str | Path) -> list[LegacyTripletSample]:
    """Scan `positive/negative` folders into structured sample metadata."""

    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    all_samples: list[LegacyTripletSample] = []
    for bucket, label in (("negative", 0), ("positive", 1)):
        bucket_dir = root / bucket
        if not bucket_dir.is_dir():
            continue

        for path in sorted(bucket_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
                continue

            parsed = parse_legacy_triplet_name(path.name)
            label_name = "real" if label == 1 else "bogus"
            declared_label = parsed.get("label_name_from_file")
            if declared_label and declared_label != label_name:
                raise ValueError(
                    f"Label conflict for {path.name}: folder={label_name}, file={declared_label}"
                )

            all_samples.append(
                LegacyTripletSample(
                    relative_path=path.relative_to(root).as_posix(),
                    bucket=bucket,
                    label=label,
                    label_name=label_name,
                    file_name=path.name,
                    file_stem=path.stem,
                    source_name=str(parsed["source_name"]),
                    group_key=str(parsed["group_key"]),
                    candidate_id=parsed["candidate_id"],
                    is_manual=bool(parsed["is_manual"]),
                    sha256=_sha256_file(path),
                )
            )

    if not all_samples:
        raise ValueError(f"No legacy triplet samples found under: {root}")
    return all_samples


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> dict[str, float]:
    ratios = {
        "train": float(train_ratio),
        "val": float(val_ratio),
        "test": float(test_ratio),
    }
    for split_name, value in ratios.items():
        if value <= 0.0:
            raise ValueError(f"Split ratio must be positive: {split_name}={value}")
    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios}")
    return ratios


def _group_samples(samples: Iterable[LegacyTripletSample]) -> list[_GroupedSamples]:
    grouped: dict[str, list[LegacyTripletSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.group_key].append(sample)

    return [
        _GroupedSamples(key=key, samples=sorted(items, key=lambda item: item.relative_path))
        for key, items in grouped.items()
    ]


def _assignment_score(
    split_name: str,
    group: _GroupedSamples,
    current_counts: dict[str, dict[str, int]],
    target_counts: dict[str, dict[str, float]],
) -> tuple[float, float, float, float]:
    next_total = current_counts[split_name]["total"] + group.total
    next_positive = current_counts[split_name]["positive"] + group.positive
    next_negative = current_counts[split_name]["negative"] + group.negative

    total_target = max(target_counts[split_name]["total"], 1.0)
    positive_target = max(target_counts[split_name]["positive"], 1.0)
    negative_target = max(target_counts[split_name]["negative"], 1.0)

    total_fill = next_total / total_target
    positive_fill = next_positive / positive_target
    negative_fill = next_negative / negative_target
    overflow_penalty = max(0.0, total_fill - 1.0) + max(0.0, positive_fill - 1.0) + max(0.0, negative_fill - 1.0)

    return (
        max(total_fill, positive_fill, negative_fill),
        total_fill + positive_fill + negative_fill + (overflow_penalty * 2.0),
        abs(total_fill - 1.0) + abs(positive_fill - 1.0) + abs(negative_fill - 1.0),
        current_counts[split_name]["total"],
    )


def assign_grouped_splits(
    samples: Iterable[LegacyTripletSample],
    *,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, list[LegacyTripletSample]]:
    """Assign groups to fixed train/val/test splits without leakage."""

    ratios = _validate_ratios(train_ratio, val_ratio, test_ratio)
    groups = _group_samples(samples)
    if not groups:
        raise ValueError("No groups available for split assignment")

    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=lambda group: (group.total, abs(group.positive - group.negative)), reverse=True)

    total_samples = sum(group.total for group in groups)
    total_positive = sum(group.positive for group in groups)
    total_negative = sum(group.negative for group in groups)

    target_counts = {
        split_name: {
            "total": ratios[split_name] * total_samples,
            "positive": ratios[split_name] * total_positive,
            "negative": ratios[split_name] * total_negative,
        }
        for split_name in SPLIT_NAMES
    }
    current_counts = {
        split_name: {"total": 0, "positive": 0, "negative": 0}
        for split_name in SPLIT_NAMES
    }
    assignments = {split_name: [] for split_name in SPLIT_NAMES}

    for index, group in enumerate(groups):
        empty_splits = [split_name for split_name in SPLIT_NAMES if not assignments[split_name]]
        remaining_groups = len(groups) - index
        if empty_splits and remaining_groups == len(empty_splits):
            chosen_split = empty_splits[0]
        else:
            chosen_split = min(
                SPLIT_NAMES,
                key=lambda split_name: _assignment_score(
                    split_name,
                    group,
                    current_counts,
                    target_counts,
                ),
            )

        assignments[chosen_split].extend(group.samples)
        current_counts[chosen_split]["total"] += group.total
        current_counts[chosen_split]["positive"] += group.positive
        current_counts[chosen_split]["negative"] += group.negative

    for split_name in SPLIT_NAMES:
        assignments[split_name].sort(key=lambda sample: sample.relative_path)
    return assignments


def _duplicates_by_hash(samples: Iterable[LegacyTripletSample]) -> list[list[str]]:
    hash_to_paths: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        hash_to_paths[sample.sha256].append(sample.relative_path)

    duplicates = [sorted(paths) for paths in hash_to_paths.values() if len(paths) > 1]
    duplicates.sort(key=lambda paths: tuple(paths))
    return duplicates


def _build_summary(
    samples: list[LegacyTripletSample],
    assignments: dict[str, list[LegacyTripletSample]],
) -> dict[str, Any]:
    split_group_keys = {
        split_name: {sample.group_key for sample in split_samples}
        for split_name, split_samples in assignments.items()
    }
    group_overlap = (
        split_group_keys["train"] & split_group_keys["val"]
        or split_group_keys["train"] & split_group_keys["test"]
        or split_group_keys["val"] & split_group_keys["test"]
    )

    label_counts = {
        "real": sum(1 for sample in samples if sample.label == 1),
        "bogus": sum(1 for sample in samples if sample.label == 0),
    }
    split_counts = {
        split_name: {
            "total": len(split_samples),
            "real": sum(1 for sample in split_samples if sample.label == 1),
            "bogus": sum(1 for sample in split_samples if sample.label == 0),
            "groups": len({sample.group_key for sample in split_samples}),
        }
        for split_name, split_samples in assignments.items()
    }

    return {
        "total_samples": len(samples),
        "label_counts": label_counts,
        "group_count": len({sample.group_key for sample in samples}),
        "manual_samples": sum(1 for sample in samples if sample.is_manual),
        "duplicate_hash_groups": _duplicates_by_hash(samples),
        "split_counts": split_counts,
        "group_overlap_detected": bool(group_overlap),
    }


def build_legacy_triplet_manifest(
    dataset_root: str | Path,
    output_path: str | Path | None = None,
    *,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, Any]:
    """Scan the legacy dataset and persist a fixed split manifest."""

    root = Path(dataset_root).resolve()
    samples = scan_legacy_triplet_dataset(root)
    assignments = assign_grouped_splits(
        samples,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    split_lookup = {
        sample.relative_path: split_name
        for split_name, split_samples in assignments.items()
        for sample in split_samples
    }
    entries = []
    for sample in sorted(samples, key=lambda item: item.relative_path):
        sample_dict = sample.to_dict()
        sample_dict["split"] = split_lookup[sample.relative_path]
        entries.append(sample_dict)

    manifest = {
        "manifest_version": 1,
        "dataset_type": "legacy_v1_triplet",
        "dataset_root": str(root),
        "triplet_storage_order": ["diff", "new", "old"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "ratios": _validate_ratios(train_ratio, val_ratio, test_ratio),
        "summary": _build_summary(samples, assignments),
        "entries": entries,
    }

    if output_path is not None:
        target_path = Path(output_path).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return manifest


def load_legacy_manifest(manifest: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Load a manifest from disk or return a manifest-like mapping."""

    if isinstance(manifest, dict):
        return manifest

    manifest_path = Path(manifest).resolve()
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def entries_for_split(
    manifest: str | Path | dict[str, Any],
    split: str,
) -> list[dict[str, Any]]:
    """Return all manifest entries for a split."""

    normalized_split = str(split).strip().lower()
    if normalized_split not in SPLIT_NAMES:
        raise ValueError(f"Unsupported split: {split}")

    manifest_doc = load_legacy_manifest(manifest)
    return [
        dict(entry)
        for entry in manifest_doc.get("entries", [])
        if str(entry.get("split", "")).lower() == normalized_split
    ]
