"""Explainable OOD scoring and bounded anomaly queue ranking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


OOD_SCORER_VERSION = "ood-score-v1"


def predictive_entropy(probabilities: Sequence[float]) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    values = np.clip(values, 1e-12, 1.0)
    values = values / max(float(np.sum(values)), 1e-12)
    entropy = -float(np.sum(values * np.log(values)))
    maximum = math.log(max(2, values.size))
    return entropy / maximum if maximum > 0.0 else 0.0


def ensemble_disagreement(
    probability_rows: Sequence[Sequence[float]],
) -> float:
    values = np.asarray(probability_rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        return 0.0
    return float(np.mean(np.var(values, axis=0)))


@dataclass(frozen=True)
class MahalanobisReference:
    mean: tuple[float, ...]
    precision: tuple[tuple[float, ...], ...]
    version: str = OOD_SCORER_VERSION

    @classmethod
    def fit(
        cls,
        embeddings: Sequence[Sequence[float]],
        *,
        regularization: float = 1e-4,
    ) -> "MahalanobisReference":
        values = np.asarray(embeddings, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 2:
            raise ValueError("OOD reference requires at least two embeddings")
        mean = np.mean(values, axis=0)
        covariance = np.cov(values, rowvar=False)
        if covariance.ndim == 0:
            covariance = np.asarray([[float(covariance)]])
        covariance += np.eye(covariance.shape[0]) * max(
            float(regularization),
            1e-12,
        )
        precision = np.linalg.pinv(covariance)
        return cls(
            mean=tuple(float(value) for value in mean),
            precision=tuple(
                tuple(float(value) for value in row)
                for row in precision
            ),
        )

    def distance(self, embedding: Sequence[float]) -> float:
        value = np.asarray(embedding, dtype=np.float64)
        mean = np.asarray(self.mean, dtype=np.float64)
        precision = np.asarray(self.precision, dtype=np.float64)
        if value.shape != mean.shape:
            raise ValueError("embedding dimension does not match OOD reference")
        delta = value - mean
        squared = float(delta.T @ precision @ delta)
        return math.sqrt(max(0.0, squared))


def score_ood_item(
    *,
    probabilities: Sequence[float],
    embedding: Sequence[float],
    reference: MahalanobisReference,
    ensemble_probabilities: Sequence[Sequence[float]] = (),
    structured_image_disagreement: float = 0.0,
    distance_scale: float = 5.0,
) -> dict[str, Any]:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("OOD probabilities must be a non-empty vector")
    values = values / max(float(np.sum(values)), 1e-12)
    entropy = predictive_entropy(values)
    max_softmax_gap = 1.0 - float(np.max(values))
    distance = reference.distance(embedding)
    normalized_distance = 1.0 - math.exp(
        -distance / max(float(distance_scale), 1e-6)
    )
    disagreement = min(
        1.0,
        max(0.0, ensemble_disagreement(ensemble_probabilities) * 10.0),
    )
    cross_modal = min(
        1.0,
        max(0.0, float(structured_image_disagreement)),
    )
    total = (
        0.30 * entropy
        + 0.20 * max_softmax_gap
        + 0.30 * normalized_distance
        + 0.10 * disagreement
        + 0.10 * cross_modal
    )
    return {
        "version": OOD_SCORER_VERSION,
        "score": min(1.0, max(0.0, total)),
        "entropy": entropy,
        "max_softmax_gap": max_softmax_gap,
        "mahalanobis_distance": distance,
        "normalized_distance": normalized_distance,
        "ensemble_disagreement": disagreement,
        "cross_modal_disagreement": cross_modal,
        "auto_reject_allowed": False,
    }


def rank_anomaly_queue(
    items: Iterable[Mapping[str, Any]],
    *,
    top_k: int,
    maximum_artifact_risk: float = 0.7,
) -> list[dict[str, Any]]:
    eligible = [
        dict(item)
        for item in items
        if float(item.get("artifact_risk") or 0.0)
        <= float(maximum_artifact_risk)
    ]
    eligible.sort(
        key=lambda item: (
            -float(item.get("ood_score") or item.get("score") or 0.0),
            -float(item.get("keep_probability") or 0.0),
            str(item.get("task_id") or ""),
        )
    )
    return [
        {
            **item,
            "rank": rank,
            "auto_reject_allowed": False,
        }
        for rank, item in enumerate(
            eligible[: max(0, int(top_k))],
            start=1,
        )
    ]
