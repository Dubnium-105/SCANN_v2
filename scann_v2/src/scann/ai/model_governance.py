"""Pure model deployment state machine and promotion gates."""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Mapping


MODEL_GOVERNANCE_VERSION = "model-governance-v1"


class DeploymentStage(str, Enum):
    REGISTERED = "registered"
    OFFLINE_PASSED = "offline_passed"
    SHADOW = "shadow"
    CANARY = "canary"
    PROMOTED = "promoted"
    RETIRED = "retired"
    INVALID_ARTIFACT = "invalid_artifact"


ALLOWED_STAGE_TRANSITIONS: dict[DeploymentStage, set[DeploymentStage]] = {
    DeploymentStage.REGISTERED: {
        DeploymentStage.OFFLINE_PASSED,
        DeploymentStage.INVALID_ARTIFACT,
        DeploymentStage.RETIRED,
    },
    DeploymentStage.OFFLINE_PASSED: {
        DeploymentStage.SHADOW,
        DeploymentStage.RETIRED,
        DeploymentStage.INVALID_ARTIFACT,
    },
    DeploymentStage.SHADOW: {
        DeploymentStage.CANARY,
        DeploymentStage.RETIRED,
        DeploymentStage.INVALID_ARTIFACT,
    },
    DeploymentStage.CANARY: {
        DeploymentStage.PROMOTED,
        DeploymentStage.SHADOW,
        DeploymentStage.RETIRED,
        DeploymentStage.INVALID_ARTIFACT,
    },
    DeploymentStage.PROMOTED: {
        DeploymentStage.RETIRED,
        DeploymentStage.INVALID_ARTIFACT,
    },
    DeploymentStage.RETIRED: {
        DeploymentStage.SHADOW,
    },
    DeploymentStage.INVALID_ARTIFACT: set(),
}


def validate_stage_transition(
    current: str | DeploymentStage,
    target: str | DeploymentStage,
) -> tuple[DeploymentStage, DeploymentStage]:
    current_stage = (
        current
        if isinstance(current, DeploymentStage)
        else DeploymentStage(str(current))
    )
    target_stage = (
        target
        if isinstance(target, DeploymentStage)
        else DeploymentStage(str(target))
    )
    if target_stage not in ALLOWED_STAGE_TRANSITIONS[current_stage]:
        raise ValueError(
            f"invalid model deployment transition: "
            f"{current_stage.value}->{target_stage.value}"
        )
    return current_stage, target_stage


def canary_selected(
    task_id: str,
    deployment_id: str,
    *,
    traffic_fraction: float,
) -> bool:
    fraction = max(0.0, min(1.0, float(traffic_fraction)))
    digest = hashlib.sha256(
        f"{deployment_id}:{task_id}".encode("utf-8")
    ).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return bucket < fraction


def evaluate_promotion_gate(
    *,
    artifact_valid: bool,
    taxonomy_version: str | None,
    partition_id: str | None,
    gold_metrics: Mapping[str, Any],
    required_metrics: Mapping[str, float],
    unsupported_required_classes: list[str] | None = None,
    shadow_drift_ok: bool = False,
    canary_review_ok: bool = False,
    human_approved: bool = False,
) -> dict[str, Any]:
    failures: list[str] = []
    if not artifact_valid:
        failures.append("artifact_invalid")
    if not str(taxonomy_version or "").strip():
        failures.append("taxonomy_missing")
    if not str(partition_id or "").strip():
        failures.append("partition_missing")
    for metric_name, threshold in required_metrics.items():
        value = gold_metrics.get(metric_name)
        if value is None or float(value) < float(threshold):
            failures.append(f"metric_failed:{metric_name}")
    if unsupported_required_classes:
        failures.append("required_classes_unverifiable")
    if not shadow_drift_ok:
        failures.append("shadow_gate_not_passed")
    if not canary_review_ok:
        failures.append("canary_gate_not_passed")
    if not human_approved:
        failures.append("human_approval_required")
    return {
        "version": MODEL_GOVERNANCE_VERSION,
        "passed": not failures,
        "failures": failures,
        "auto_promotion_allowed": False,
    }
