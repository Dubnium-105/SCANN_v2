"""Deterministic constrained active-learning batch selection."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ACTIVE_LEARNING_STRATEGY_VERSION = "active-learning-v1"


def _bounded(value: Any) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(normalized):
        return 0.0
    return max(0.0, min(1.0, normalized))


def _stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(
        f"{int(seed)}:{value}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.shape != right_values.shape or left_values.size == 0:
        return 0.0
    denominator = float(
        np.linalg.norm(left_values) * np.linalg.norm(right_values)
    )
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(left_values, right_values) / denominator)


@dataclass(frozen=True)
class ActiveLearningScore:
    task_id: str
    score: float
    uncertainty: float
    model_disagreement: float
    embedding_diversity: float
    rare_class_value: float
    recency_or_domain_shift: float
    group_key: str
    embedding: tuple[float, ...] = ()
    ood: bool = False
    high_business_value: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_ACTIVE_LEARNING_WEIGHTS: dict[str, float] = {
    "uncertainty": 0.40,
    "model_disagreement": 0.20,
    "embedding_diversity": 0.20,
    "rare_class_value": 0.15,
    "recency_or_domain_shift": 0.05,
}


def score_active_learning_item(
    payload: Mapping[str, Any],
    *,
    weights: Mapping[str, float] | None = None,
) -> ActiveLearningScore:
    normalized_weights = {
        **DEFAULT_ACTIVE_LEARNING_WEIGHTS,
        **dict(weights or {}),
    }
    components = {
        name: _bounded(payload.get(name))
        for name in DEFAULT_ACTIVE_LEARNING_WEIGHTS
    }
    total_weight = sum(
        max(0.0, float(normalized_weights[name]))
        for name in components
    )
    if total_weight <= 0.0:
        raise ValueError("active-learning weights must have positive mass")
    score = sum(
        components[name]
        * max(0.0, float(normalized_weights[name]))
        for name in components
    ) / total_weight
    return ActiveLearningScore(
        task_id=str(payload.get("task_id") or "").strip(),
        score=score,
        uncertainty=components["uncertainty"],
        model_disagreement=components["model_disagreement"],
        embedding_diversity=components["embedding_diversity"],
        rare_class_value=components["rare_class_value"],
        recency_or_domain_shift=components["recency_or_domain_shift"],
        group_key=(
            str(payload.get("group_key") or "").strip()
            or str(payload.get("task_id") or "").strip()
        ),
        embedding=tuple(
            float(value)
            for value in payload.get("embedding") or ()
        ),
        ood=bool(payload.get("ood")),
        high_business_value=bool(payload.get("high_business_value")),
        metadata=(
            dict(payload.get("metadata") or {})
            if isinstance(payload.get("metadata"), Mapping)
            else {}
        ),
    )


def select_active_learning_batch(
    items: Iterable[Mapping[str, Any] | ActiveLearningScore],
    *,
    budget: int,
    seed: int = 42,
    max_per_group: int = 3,
    duplicate_similarity: float = 0.98,
    dual_review_fraction: float = 0.10,
    weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    if int(budget) <= 0:
        raise ValueError("active-learning budget must be positive")
    scored = [
        item
        if isinstance(item, ActiveLearningScore)
        else score_active_learning_item(item, weights=weights)
        for item in items
    ]
    if any(not item.task_id for item in scored):
        raise ValueError("active-learning items require task_id")
    scored.sort(
        key=lambda item: (
            -item.score,
            _stable_fraction(item.task_id, seed),
            item.task_id,
        )
    )
    selected: list[ActiveLearningScore] = []
    group_counts: dict[str, int] = {}
    rejected_duplicate = 0
    rejected_group_limit = 0
    for item in scored:
        if len(selected) >= int(budget):
            break
        if group_counts.get(item.group_key, 0) >= max(1, int(max_per_group)):
            rejected_group_limit += 1
            continue
        if item.embedding and any(
            existing.embedding
            and cosine_similarity(item.embedding, existing.embedding)
            >= float(duplicate_similarity)
            for existing in selected
        ):
            rejected_duplicate += 1
            continue
        group_counts[item.group_key] = group_counts.get(item.group_key, 0) + 1
        selected.append(item)

    selected_payload: list[dict[str, Any]] = []
    for rank, item in enumerate(selected, start=1):
        dual_review = (
            item.ood
            or item.high_business_value
            or _stable_fraction(item.task_id, seed + 1)
            < max(0.0, min(1.0, float(dual_review_fraction)))
        )
        selected_payload.append(
            {
                **item.to_dict(),
                "rank": rank,
                "dual_review": dual_review,
                "reasons": [
                    name
                    for name in DEFAULT_ACTIVE_LEARNING_WEIGHTS
                    if getattr(item, name) >= 0.5
                ]
                + (["ood"] if item.ood else [])
                + (
                    ["high_business_value"]
                    if item.high_business_value
                    else []
                ),
            }
        )
    return {
        "strategy_version": ACTIVE_LEARNING_STRATEGY_VERSION,
        "seed": int(seed),
        "budget": int(budget),
        "selected_count": len(selected_payload),
        "max_per_group": int(max_per_group),
        "duplicate_similarity": float(duplicate_similarity),
        "dual_review_fraction": float(dual_review_fraction),
        "weights": {
            **DEFAULT_ACTIVE_LEARNING_WEIGHTS,
            **dict(weights or {}),
        },
        "rejected_duplicate_count": rejected_duplicate,
        "rejected_group_limit_count": rejected_group_limit,
        "items": selected_payload,
    }
