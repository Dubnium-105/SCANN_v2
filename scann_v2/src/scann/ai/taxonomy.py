"""Versioned, non-destructive taxonomy derivation for SCANN annotations."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

from scann.core.annotation_models import AnnotationLabel, DETAIL_TYPE_TO_LABEL, DetailType


TAXONOMY_VERSION = "scann-discovery-v1"


class ReviewAction(str, Enum):
    KEEP = "keep"
    REJECT = "reject"
    UNKNOWN = "unknown"


class PhenomenonFamily(str, Enum):
    APPEARANCE = "appearance"
    VARIABILITY = "variability"
    MOVING = "moving"
    DISAPPEARANCE = "disappearance"
    PERSISTENT_MISMATCH = "persistent_mismatch"
    INSTRUMENT_ARTIFACT = "instrument_artifact"
    UNKNOWN = "unknown"


DETAIL_TYPE_TO_FAMILY: dict[DetailType, PhenomenonFamily] = {
    DetailType.ASTEROID: PhenomenonFamily.MOVING,
    DetailType.SUPERNOVA: PhenomenonFamily.APPEARANCE,
    DetailType.VARIABLE_STAR: PhenomenonFamily.VARIABILITY,
    DetailType.SATELLITE_TRAIL: PhenomenonFamily.INSTRUMENT_ARTIFACT,
    DetailType.NOISE: PhenomenonFamily.INSTRUMENT_ARTIFACT,
    DetailType.DIFFRACTION_SPIKE: PhenomenonFamily.INSTRUMENT_ARTIFACT,
    DetailType.CMOS_CONDENSATION: PhenomenonFamily.INSTRUMENT_ARTIFACT,
    DetailType.CORRESPONDING: PhenomenonFamily.PERSISTENT_MISMATCH,
    DetailType.DISAPPEARED_ASTEROID: PhenomenonFamily.DISAPPEARANCE,
    DetailType.DISAPPEARED_STAR: PhenomenonFamily.DISAPPEARANCE,
    DetailType.DISAPPEARED_GALAXY: PhenomenonFamily.DISAPPEARANCE,
}


LABEL_TO_REVIEW_ACTION: dict[AnnotationLabel, ReviewAction] = {
    AnnotationLabel.REAL: ReviewAction.KEEP,
    AnnotationLabel.BOGUS: ReviewAction.REJECT,
}


@dataclass(frozen=True)
class TaxonomyAssignment:
    taxonomy_version: str
    detail_type: str | None
    legacy_label: str | None
    expected_legacy_label: str | None
    review_action: str
    phenomenon_family: str
    label_origin: str
    label_conflict: bool
    valid_detail_type: bool
    valid_explicit_label: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_detail_type(value: Any) -> DetailType | None:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return None
    try:
        return DetailType(normalized)
    except ValueError:
        return None


def normalize_legacy_label(value: Any) -> AnnotationLabel | None:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return None
    try:
        return AnnotationLabel(normalized)
    except ValueError:
        return None


def derive_taxonomy(
    *,
    detail_type: Any,
    explicit_label: Any = None,
) -> TaxonomyAssignment:
    """Derive taxonomy fields without modifying the source annotation.

    A valid explicit coarse label is preserved.  If it is missing or invalid,
    the legacy coarse label is derived from ``detail_type``.  Conflicts are
    surfaced rather than silently rewritten.
    """

    raw_detail_type = str(detail_type or "").strip().lower()
    raw_explicit_label = str(explicit_label or "").strip().lower()
    normalized_detail_type = normalize_detail_type(raw_detail_type)
    normalized_explicit_label = normalize_legacy_label(raw_explicit_label)
    expected_label = (
        DETAIL_TYPE_TO_LABEL.get(normalized_detail_type)
        if normalized_detail_type is not None
        else None
    )
    label_conflict = (
        normalized_explicit_label is not None
        and expected_label is not None
        and normalized_explicit_label != expected_label
    )

    if normalized_explicit_label is not None:
        legacy_label = normalized_explicit_label
        label_origin = "explicit_label"
    elif expected_label is not None:
        legacy_label = expected_label
        label_origin = "derived_from_detail_type"
    else:
        legacy_label = None
        label_origin = "unknown"

    review_action = (
        LABEL_TO_REVIEW_ACTION.get(legacy_label, ReviewAction.UNKNOWN)
        if legacy_label is not None
        else ReviewAction.UNKNOWN
    )
    phenomenon_family = (
        DETAIL_TYPE_TO_FAMILY.get(
            normalized_detail_type,
            PhenomenonFamily.UNKNOWN,
        )
        if normalized_detail_type is not None
        else PhenomenonFamily.UNKNOWN
    )

    return TaxonomyAssignment(
        taxonomy_version=TAXONOMY_VERSION,
        detail_type=(
            normalized_detail_type.value
            if normalized_detail_type is not None
            else (raw_detail_type or None)
        ),
        legacy_label=legacy_label.value if legacy_label is not None else None,
        expected_legacy_label=expected_label.value if expected_label is not None else None,
        review_action=review_action.value,
        phenomenon_family=phenomenon_family.value,
        label_origin=label_origin,
        label_conflict=label_conflict,
        valid_detail_type=normalized_detail_type is not None,
        valid_explicit_label=(
            normalized_explicit_label is not None
            if raw_explicit_label
            else True
        ),
    )


def enrich_annotation(annotation: Mapping[str, Any]) -> dict[str, Any]:
    """Return a snapshot-safe copy with derived taxonomy fields."""

    enriched = dict(annotation)
    assignment = derive_taxonomy(
        detail_type=annotation.get("detail_type"),
        explicit_label=annotation.get("label"),
    )
    enriched["original_label"] = annotation.get("label")
    enriched["label"] = assignment.legacy_label
    enriched["review_action"] = assignment.review_action
    enriched["phenomenon_family"] = assignment.phenomenon_family
    enriched["taxonomy_version"] = assignment.taxonomy_version
    enriched["label_origin"] = assignment.label_origin
    enriched["label_conflict"] = assignment.label_conflict
    return enriched


def build_taxonomy_audit(
    annotations: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    detail_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    origin_counts: Counter[str] = Counter()
    invalid_detail_types: Counter[str] = Counter()
    invalid_explicit_labels: Counter[str] = Counter()
    conflict_counts: Counter[str] = Counter()
    total = 0

    for annotation in annotations:
        total += 1
        raw_detail_type = str(annotation.get("detail_type") or "").strip().lower()
        raw_label = str(annotation.get("label") or "").strip().lower()
        assignment = derive_taxonomy(
            detail_type=raw_detail_type,
            explicit_label=raw_label,
        )
        detail_counts[assignment.detail_type or "missing"] += 1
        label_counts[assignment.legacy_label or "unknown"] += 1
        action_counts[assignment.review_action] += 1
        family_counts[assignment.phenomenon_family] += 1
        origin_counts[assignment.label_origin] += 1
        if raw_detail_type and not assignment.valid_detail_type:
            invalid_detail_types[raw_detail_type] += 1
        if raw_label and not assignment.valid_explicit_label:
            invalid_explicit_labels[raw_label] += 1
        if assignment.label_conflict:
            conflict_counts[
                f"{assignment.detail_type}:{raw_label}->{assignment.expected_legacy_label}"
            ] += 1

    return {
        "taxonomy_version": TAXONOMY_VERSION,
        "total_annotations": total,
        "detail_type_counts": dict(sorted(detail_counts.items())),
        "legacy_label_counts": dict(sorted(label_counts.items())),
        "review_action_counts": dict(sorted(action_counts.items())),
        "phenomenon_family_counts": dict(sorted(family_counts.items())),
        "label_origin_counts": dict(sorted(origin_counts.items())),
        "invalid_detail_types": dict(sorted(invalid_detail_types.items())),
        "invalid_explicit_labels": dict(sorted(invalid_explicit_labels.items())),
        "label_conflicts": dict(sorted(conflict_counts.items())),
    }
