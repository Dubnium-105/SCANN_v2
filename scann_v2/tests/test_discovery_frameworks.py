from __future__ import annotations

import math

import pytest

from scann.ai.active_learning import select_active_learning_batch
from scann.ai.model_governance import (
    DeploymentStage,
    canary_selected,
    evaluate_promotion_gate,
    validate_stage_transition,
)
from scann.ai.object_association import associate_observations
from scann.ai.ood import (
    MahalanobisReference,
    rank_anomaly_queue,
    score_ood_item,
)
from scann.core.discovery_storage import DiscoveryStorage
from scann.native_annotation.review_feedback_service import (
    compare_review_boxes,
)
from scann.services.operational_monitoring import (
    aggregate_detection_metrics,
)


def test_active_learning_is_deterministic_and_enforces_constraints():
    items = [
        {
            "task_id": "a",
            "uncertainty": 1.0,
            "group_key": "night-1",
            "embedding": [1.0, 0.0],
        },
        {
            "task_id": "b",
            "uncertainty": 0.9,
            "group_key": "night-1",
            "embedding": [0.999, 0.001],
        },
        {
            "task_id": "c",
            "uncertainty": 0.8,
            "group_key": "night-2",
            "embedding": [0.0, 1.0],
            "ood": True,
        },
    ]
    first = select_active_learning_batch(
        items,
        budget=2,
        max_per_group=1,
        seed=7,
    )
    second = select_active_learning_batch(
        items,
        budget=2,
        max_per_group=1,
        seed=7,
    )

    assert first == second
    assert [item["task_id"] for item in first["items"]] == ["a", "c"]
    assert first["items"][1]["dual_review"] is True


def test_ood_scoring_and_queue_are_bounded_and_never_auto_reject():
    reference = MahalanobisReference.fit(
        [[0.0, 0.0], [0.1, -0.1], [-0.1, 0.1]]
    )
    score = score_ood_item(
        probabilities=[0.5, 0.5],
        embedding=[3.0, 3.0],
        reference=reference,
    )
    queue = rank_anomaly_queue(
        [
            {"task_id": "a", "ood_score": score["score"]},
            {"task_id": "b", "ood_score": 0.99, "artifact_risk": 0.9},
        ],
        top_k=1,
    )

    assert 0.0 <= score["score"] <= 1.0
    assert score["auto_reject_allowed"] is False
    assert [item["task_id"] for item in queue] == ["a"]
    assert queue[0]["auto_reject_allowed"] is False


def test_object_association_obeys_wcs_gate_and_time_window():
    observations = [
        {
            "observation_id": "a",
            "ra_deg": 10.0,
            "dec_deg": 20.0,
            "observed_at": "2026-01-01T00:00:00Z",
        },
        {
            "observation_id": "b",
            "ra_deg": 10.0001,
            "dec_deg": 20.0,
            "observed_at": "2026-01-01T00:01:00Z",
        },
    ]
    gated = associate_observations(
        observations,
        maximum_separation_arcsec=2.0,
        maximum_time_delta_seconds=120.0,
        wcs_valid_fraction=0.9,
    )
    enabled = associate_observations(
        observations,
        maximum_separation_arcsec=2.0,
        maximum_time_delta_seconds=120.0,
        wcs_valid_fraction=0.99,
    )

    assert gated["enabled"] is False
    assert enabled["enabled"] is True
    assert len(enabled["associations"]) == 1


def test_governance_accepts_enum_values_and_requires_all_gates():
    current, target = validate_stage_transition(
        DeploymentStage.SHADOW,
        DeploymentStage.CANARY,
    )
    gate = evaluate_promotion_gate(
        artifact_valid=True,
        taxonomy_version="taxonomy-v1",
        partition_id="partition-1",
        gold_metrics={"recall": 0.9},
        required_metrics={"recall": 0.8},
        shadow_drift_ok=True,
        canary_review_ok=True,
        human_approved=False,
    )

    assert current is DeploymentStage.SHADOW
    assert target is DeploymentStage.CANARY
    assert gate["passed"] is False
    assert "human_approval_required" in gate["failures"]
    assert canary_selected(
        "task-1",
        "deployment-1",
        traffic_fraction=0.1,
    ) == canary_selected(
        "task-1",
        "deployment-1",
        traffic_fraction=0.1,
    )


def test_review_feedback_and_monitoring_summaries():
    feedback = compare_review_boxes(
        [
            {
                "x": 10,
                "y": 10,
                "width": 10,
                "height": 10,
                "label": "real",
                "detail_type": "candidate",
                "confidence": 0.8,
            }
        ],
        [
            {
                "x": 11,
                "y": 10,
                "width": 10,
                "height": 10,
                "label": "real",
                "detail_type": "supernova",
            },
            {
                "x": 50,
                "y": 50,
                "width": 8,
                "height": 8,
            },
        ],
    )
    metrics = aggregate_detection_metrics(
        [
            {
                "duration_ms": 10,
                "final_candidate_count": 0,
                "stage_counts": {"raw": 5, "final": 0},
                "fallback_reasons": ["relaxed"],
            },
            {
                "duration_ms": 20,
                "final_candidate_count": 2,
                "stage_counts": {"raw": 8, "final": 2},
            },
        ]
    )

    assert feedback["outcome"] == "partial_accept"
    assert feedback["human_added_count"] == 1
    assert feedback["geometry_correction_count"] == 1
    assert feedback["reclassification_count"] == 1
    assert math.isclose(metrics["empty_result_rate"], 0.5)
    assert metrics["fallback_reasons"] == {"relaxed": 1}


def test_discovery_storage_crud_is_idempotent_for_review_events(tmp_path):
    storage = DiscoveryStorage(tmp_path)
    evaluation = storage.create_evaluation(
        run_id="evaluation-1",
        run_type="candidate",
        status="registered",
        partition_id=None,
        model_id=None,
        config={"threshold": 1},
        created_by="admin",
    )
    completed = storage.update_evaluation(
        "evaluation-1",
        status="completed",
        metrics={"recall": 1.0},
    )
    first = storage.create_review_event(
        event_id="review-1",
        task_id="task-1",
        prelabel_id="prelabel-1",
        revision_id="revision-1",
        model_id="model-1",
        outcome="full_accept",
        match_algorithm_version="review-match-v1",
        result={"matched": 1},
        created_by="annotator",
    )
    second = storage.create_review_event(
        event_id="review-1",
        task_id="task-1",
        prelabel_id="prelabel-1",
        revision_id="revision-1",
        model_id="model-1",
        outcome="full_accept",
        match_algorithm_version="review-match-v1",
        result={"matched": 1},
        created_by="annotator",
    )

    assert evaluation["status"] == "registered"
    assert completed["metrics"]["recall"] == pytest.approx(1.0)
    assert first == second
    assert storage.list_review_events(task_id="task-1") == [first]
