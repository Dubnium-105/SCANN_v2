from __future__ import annotations

from scann.ai.taxonomy import (
    DETAIL_TYPE_TO_FAMILY,
    TAXONOMY_VERSION,
    PhenomenonFamily,
    ReviewAction,
    build_taxonomy_audit,
    derive_taxonomy,
    enrich_annotation,
)
from scann.core.annotation_models import DETAIL_TYPE_TO_LABEL, DetailType


def test_every_detail_type_has_a_family_and_legacy_label():
    assert set(DETAIL_TYPE_TO_FAMILY) == set(DetailType)
    assert set(DETAIL_TYPE_TO_LABEL) == set(DetailType)


def test_missing_coarse_label_is_derived_without_mutating_source():
    source = {
        "x": 1,
        "y": 2,
        "width": 3,
        "height": 4,
        "label": None,
        "detail_type": "asteroid",
    }

    enriched = enrich_annotation(source)

    assert source["label"] is None
    assert enriched["original_label"] is None
    assert enriched["label"] == "real"
    assert enriched["review_action"] == ReviewAction.KEEP.value
    assert enriched["phenomenon_family"] == PhenomenonFamily.MOVING.value
    assert enriched["label_origin"] == "derived_from_detail_type"
    assert enriched["taxonomy_version"] == TAXONOMY_VERSION


def test_explicit_conflicting_label_is_preserved_and_reported():
    assignment = derive_taxonomy(
        detail_type="supernova",
        explicit_label="bogus",
    )

    assert assignment.legacy_label == "bogus"
    assert assignment.expected_legacy_label == "real"
    assert assignment.review_action == ReviewAction.REJECT.value
    assert assignment.phenomenon_family == PhenomenonFamily.APPEARANCE.value
    assert assignment.label_conflict is True
    assert assignment.label_origin == "explicit_label"


def test_unknown_values_do_not_gain_a_synthetic_known_label():
    assignment = derive_taxonomy(
        detail_type="unrecognized",
        explicit_label="maybe",
    )

    assert assignment.detail_type == "unrecognized"
    assert assignment.legacy_label is None
    assert assignment.review_action == ReviewAction.UNKNOWN.value
    assert assignment.phenomenon_family == PhenomenonFamily.UNKNOWN.value
    assert assignment.valid_detail_type is False
    assert assignment.valid_explicit_label is False


def test_taxonomy_audit_counts_derivation_and_invalid_values():
    report = build_taxonomy_audit(
        [
            {"detail_type": "asteroid", "label": None},
            {"detail_type": "noise", "label": "bogus"},
            {"detail_type": "supernova", "label": "bogus"},
            {"detail_type": "unknown-detail", "label": "maybe"},
        ]
    )

    assert report["total_annotations"] == 4
    assert report["label_origin_counts"]["derived_from_detail_type"] == 1
    assert report["label_origin_counts"]["explicit_label"] == 2
    assert report["label_origin_counts"]["unknown"] == 1
    assert report["invalid_detail_types"] == {"unknown-detail": 1}
    assert report["invalid_explicit_labels"] == {"maybe": 1}
    assert report["label_conflicts"] == {"supernova:bogus->real": 1}
