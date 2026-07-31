"""Frozen-encoder hierarchical classification framework."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from scann.ai.feature_classifier import (
    FeatureHeadClassifier,
    feature_encoder_spec,
    forward_feature_encoder,
    load_feature_encoder,
    preprocess_feature_batch,
)
from scann.ai.taxonomy import (
    DETAIL_TYPE_TO_FAMILY,
    LABEL_TO_REVIEW_ACTION,
    TAXONOMY_VERSION,
    PhenomenonFamily,
    ReviewAction,
)
from scann.core.annotation_models import (
    DETAIL_TYPE_TO_LABEL,
    DetailType,
)


HIERARCHICAL_MODEL_FORMAT = "hierarchical_v1"
ACTION_CLASSES: tuple[str, ...] = (
    ReviewAction.KEEP.value,
    ReviewAction.REJECT.value,
)
FAMILY_CLASSES: tuple[str, ...] = tuple(
    item.value
    for item in PhenomenonFamily
    if item is not PhenomenonFamily.UNKNOWN
)
DETAIL_CLASSES: tuple[str, ...] = tuple(item.value for item in DetailType)


def _head(
    feature_dim: int,
    class_count: int,
    *,
    hidden_dim: int,
    dropout: float,
) -> FeatureHeadClassifier:
    return FeatureHeadClassifier(
        feature_dim=feature_dim,
        num_classes=class_count,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


class HierarchicalHeads(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        *,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.review_action_head = _head(
            feature_dim,
            len(ACTION_CLASSES),
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.phenomenon_family_head = _head(
            feature_dim,
            len(FAMILY_CLASSES),
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.detail_type_head = _head(
            feature_dim,
            len(DETAIL_CLASSES),
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "review_action_logits": self.review_action_head(features),
            "phenomenon_family_logits": self.phenomenon_family_head(features),
            "detail_type_logits": self.detail_type_head(features),
        }


class FrozenFeatureHierarchicalClassifier(nn.Module):
    """Frozen image encoder with independently masked hierarchy heads."""

    uses_internal_preprocessing = True
    is_hierarchical = True

    def __init__(
        self,
        *,
        feature_encoder: str,
        device: torch.device,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        heads: HierarchicalHeads | None = None,
        temperatures: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.encoder, self.spec = load_feature_encoder(
            feature_encoder,
            device=device,
        )
        self.heads = (
            heads
            if heads is not None
            else HierarchicalHeads(
                self.spec.feature_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        ).to(device)
        self.hidden_dim = int(self.heads.hidden_dim)
        self.dropout = float(self.heads.dropout)
        self.device = device
        self.temperatures = {
            "review_action": max(
                1e-4,
                float((temperatures or {}).get("review_action", 1.0)),
            ),
            "phenomenon_family": max(
                1e-4,
                float((temperatures or {}).get("phenomenon_family", 1.0)),
            ),
            "detail_type": max(
                1e-4,
                float((temperatures or {}).get("detail_type", 1.0)),
            ),
        }

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        prepared = preprocess_feature_batch(
            x.to(self.device),
            input_size=self.spec.input_size,
        )
        return forward_feature_encoder(
            self.encoder,
            prepared,
            family=self.spec.family,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.heads(self.extract_features(x))
        return {
            "review_action_logits": (
                outputs["review_action_logits"]
                / self.temperatures["review_action"]
            ),
            "phenomenon_family_logits": (
                outputs["phenomenon_family_logits"]
                / self.temperatures["phenomenon_family"]
            ),
            "detail_type_logits": (
                outputs["detail_type_logits"]
                / self.temperatures["detail_type"]
            ),
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Mapping[str, Any],
        *,
        device: torch.device,
    ) -> "FrozenFeatureHierarchicalClassifier":
        feature_encoder = str(
            checkpoint.get("feature_encoder")
            or "auto"
        )
        spec = feature_encoder_spec(feature_encoder, device=device)
        head_config = (
            checkpoint.get("head_config")
            if isinstance(checkpoint.get("head_config"), Mapping)
            else {}
        )
        heads = HierarchicalHeads(
            spec.feature_dim,
            hidden_dim=int(head_config.get("hidden_dim") or 256),
            dropout=float(head_config.get("dropout") or 0.1),
        )
        states = checkpoint.get("head_states")
        if not isinstance(states, Mapping):
            raise RuntimeError(
                "hierarchical checkpoint is missing head_states"
            )
        required = {
            "review_action_head": heads.review_action_head,
            "phenomenon_family_head": heads.phenomenon_family_head,
            "detail_type_head": heads.detail_type_head,
        }
        for name, module in required.items():
            state = states.get(name)
            if not isinstance(state, Mapping):
                raise RuntimeError(
                    f"hierarchical checkpoint is missing {name}"
                )
            module.load_state_dict(dict(state), strict=True)
        return cls(
            feature_encoder=feature_encoder,
            device=device,
            heads=heads,
            temperatures=(
                checkpoint.get("temperatures")
                if isinstance(checkpoint.get("temperatures"), Mapping)
                else None
            ),
        )


def _masked_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    class_weights: torch.Tensor | None,
    gamma: float,
) -> torch.Tensor:
    mask = targets >= 0
    if not bool(mask.any()):
        return logits.sum() * 0.0
    selected_logits = logits[mask]
    selected_targets = targets[mask].long()
    cross_entropy = F.cross_entropy(
        selected_logits,
        selected_targets,
        weight=class_weights,
        reduction="none",
    )
    probabilities = torch.softmax(selected_logits, dim=-1)
    target_probabilities = probabilities.gather(
        1,
        selected_targets.unsqueeze(1),
    ).squeeze(1)
    return (
        ((1.0 - target_probabilities).clamp_min(0.0) ** float(gamma))
        * cross_entropy
    ).mean()


def hierarchical_loss(
    outputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    *,
    action_weight: float = 1.0,
    family_weight: float = 1.0,
    detail_weight: float = 1.0,
    focal_gamma: float = 2.0,
    class_weights: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    weights = class_weights or {}
    action_loss = _masked_focal_loss(
        outputs["review_action_logits"],
        targets["review_action"],
        class_weights=weights.get("review_action"),
        gamma=focal_gamma,
    )
    family_loss = _masked_focal_loss(
        outputs["phenomenon_family_logits"],
        targets["phenomenon_family"],
        class_weights=weights.get("phenomenon_family"),
        gamma=focal_gamma,
    )
    detail_loss = _masked_focal_loss(
        outputs["detail_type_logits"],
        targets["detail_type"],
        class_weights=weights.get("detail_type"),
        gamma=focal_gamma,
    )
    total = (
        float(action_weight) * action_loss
        + float(family_weight) * family_loss
        + float(detail_weight) * detail_loss
    )
    return {
        "loss": total,
        "review_action_loss": action_loss,
        "phenomenon_family_loss": family_loss,
        "detail_type_loss": detail_loss,
    }


def taxonomy_target_indices(
    detail_type: str | None,
) -> dict[str, int]:
    try:
        normalized_detail = DetailType(str(detail_type or "").strip().lower())
    except ValueError:
        return {
            "review_action": -1,
            "phenomenon_family": -1,
            "detail_type": -1,
        }
    family = DETAIL_TYPE_TO_FAMILY[normalized_detail].value
    label = DETAIL_TYPE_TO_LABEL[normalized_detail]
    action = LABEL_TO_REVIEW_ACTION[label].value
    return {
        "review_action": ACTION_CLASSES.index(action),
        "phenomenon_family": FAMILY_CLASSES.index(family),
        "detail_type": DETAIL_CLASSES.index(normalized_detail.value),
    }


def hierarchical_predictions(
    outputs: Mapping[str, torch.Tensor],
) -> list[dict[str, Any]]:
    action_probs = torch.softmax(
        outputs["review_action_logits"],
        dim=-1,
    )
    family_probs = torch.softmax(
        outputs["phenomenon_family_logits"],
        dim=-1,
    )
    detail_probs = torch.softmax(
        outputs["detail_type_logits"],
        dim=-1,
    )
    results: list[dict[str, Any]] = []
    for row_index in range(detail_probs.shape[0]):
        action_index = int(torch.argmax(action_probs[row_index]).item())
        family_index = int(torch.argmax(family_probs[row_index]).item())
        detail_index = int(torch.argmax(detail_probs[row_index]).item())
        action = ACTION_CLASSES[action_index]
        family = FAMILY_CLASSES[family_index]
        detail = DETAIL_CLASSES[detail_index]
        results.append(
            {
                "review_action": action,
                "review_action_confidence": float(
                    action_probs[row_index, action_index].item()
                ),
                "phenomenon_family": family,
                "phenomenon_family_confidence": float(
                    family_probs[row_index, family_index].item()
                ),
                "detail_type": detail,
                "detail_type_confidence": float(
                    detail_probs[row_index, detail_index].item()
                ),
                "label": (
                    "real"
                    if action == ReviewAction.KEEP.value
                    else "bogus"
                ),
                "score": float(
                    action_probs[
                        row_index,
                        ACTION_CLASSES.index(ReviewAction.KEEP.value),
                    ].item()
                ),
            }
        )
    return results


def build_hierarchical_checkpoint(
    model: FrozenFeatureHierarchicalClassifier,
    *,
    partition_id: str,
    partition_manifest_sha256: str,
    feature_version: str,
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "model_format": HIERARCHICAL_MODEL_FORMAT,
        "taxonomy_version": TAXONOMY_VERSION,
        "partition_id": str(partition_id),
        "partition_manifest_sha256": str(partition_manifest_sha256),
        "feature_encoder": model.spec.name,
        "feature_dim": model.spec.feature_dim,
        "input_size": model.spec.input_size,
        "feature_version": str(feature_version),
        "head_config": {
            "hidden_dim": model.hidden_dim,
            "dropout": model.dropout,
        },
        "head_states": {
            "review_action_head": model.heads.review_action_head.state_dict(),
            "phenomenon_family_head": model.heads.phenomenon_family_head.state_dict(),
            "detail_type_head": model.heads.detail_type_head.state_dict(),
        },
        "classes": {
            "review_action": list(ACTION_CLASSES),
            "phenomenon_family": list(FAMILY_CLASSES),
            "detail_type": list(DETAIL_CLASSES),
        },
        "temperatures": dict(model.temperatures),
        "metrics": dict(metrics or {}),
    }


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    accuracy: float | None
    confidence: float | None


def calibration_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    bin_count: int = 10,
) -> dict[str, Any]:
    mask = targets >= 0
    probabilities = probabilities[mask]
    targets = targets[mask].long()
    if probabilities.numel() == 0:
        return {
            "support": 0,
            "brier_score": None,
            "ece": None,
            "bins": [],
        }
    confidence, predictions = probabilities.max(dim=1)
    accuracy = predictions.eq(targets).float()
    one_hot = F.one_hot(
        targets,
        num_classes=probabilities.shape[1],
    ).float()
    brier = torch.mean(
        torch.sum((probabilities - one_hot) ** 2, dim=1)
    )
    bins: list[ReliabilityBin] = []
    ece = torch.zeros((), dtype=probabilities.dtype)
    for index in range(max(1, int(bin_count))):
        lower = index / max(1, int(bin_count))
        upper = (index + 1) / max(1, int(bin_count))
        in_bin = (confidence >= lower) & (
            confidence <= upper
            if index == int(bin_count) - 1
            else confidence < upper
        )
        count = int(in_bin.sum().item())
        if count:
            bin_accuracy = float(accuracy[in_bin].mean().item())
            bin_confidence = float(confidence[in_bin].mean().item())
            ece += (
                float(count) / probabilities.shape[0]
                * abs(bin_accuracy - bin_confidence)
            )
        else:
            bin_accuracy = None
            bin_confidence = None
        bins.append(
            ReliabilityBin(
                lower=lower,
                upper=upper,
                count=count,
                accuracy=bin_accuracy,
                confidence=bin_confidence,
            )
        )
    return {
        "support": int(probabilities.shape[0]),
        "brier_score": float(brier.item()),
        "ece": float(ece.item()),
        "bins": [
            {
                "lower": item.lower,
                "upper": item.upper,
                "count": item.count,
                "accuracy": item.accuracy,
                "confidence": item.confidence,
            }
            for item in bins
        ],
    }


def fit_temperature_scaling(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    maximum_steps: int = 50,
) -> float:
    """Fit one positive scalar on validation logits only."""

    mask = targets >= 0
    selected_logits = logits.detach()[mask].float()
    selected_targets = targets.detach()[mask].long()
    if selected_targets.numel() < 2:
        return 1.0
    log_temperature = torch.zeros(
        (),
        dtype=selected_logits.dtype,
        device=selected_logits.device,
        requires_grad=True,
    )
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=max(1, int(maximum_steps)),
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.clamp(-4.0, 4.0).exp()
        loss = F.cross_entropy(
            selected_logits / temperature,
            selected_targets,
        )
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except RuntimeError:
        return 1.0
    temperature = float(
        log_temperature.detach().clamp(-4.0, 4.0).exp().item()
    )
    return max(1e-4, min(100.0, temperature))
