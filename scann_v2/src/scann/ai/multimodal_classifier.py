"""Shared-encoder new/old/difference late-fusion framework."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from scann.ai.feature_classifier import (
    forward_feature_encoder,
    load_feature_encoder,
    preprocess_feature_batch,
)
from scann.ai.hierarchical_classifier import HierarchicalHeads


MULTIMODAL_MODEL_FORMAT = "multimodal_hierarchical_v1"
MULTIMODAL_FEATURE_VERSION = "candidate-structured-v1"
DEFAULT_STRUCTURED_FEATURE_NAMES: tuple[str, ...] = (
    "snr",
    "flux_difference",
    "fwhm",
    "ellipticity",
    "positive_fraction",
    "negative_fraction",
    "dipole_score",
    "background_gradient",
    "edge_distance",
    "saturated_fraction",
    "centroid_shift",
    "area",
    "sharpness",
    "contrast",
    "polarity",
)


def build_structured_feature_matrix(
    records: list[Mapping[str, Any]],
    *,
    feature_names: tuple[str, ...] = DEFAULT_STRUCTURED_FEATURE_NAMES,
) -> tuple[np.ndarray, np.ndarray]:
    """Build values plus an explicit missingness mask from candidate records."""

    values = np.zeros(
        (len(records), len(feature_names)),
        dtype=np.float32,
    )
    mask = np.zeros_like(values, dtype=bool)
    for row_index, record in enumerate(records):
        nested = record.get("structured_features")
        if not isinstance(nested, Mapping):
            nested = record.get("candidate_features")
        source = nested if isinstance(nested, Mapping) else record
        for column_index, name in enumerate(feature_names):
            raw = source.get(name)
            try:
                normalized = float(raw)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(normalized):
                continue
            values[row_index, column_index] = normalized
            mask[row_index, column_index] = True
    return values, mask


@dataclass(frozen=True)
class FeatureNormalization:
    mean: tuple[float, ...]
    std: tuple[float, ...]
    feature_names: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        mask: np.ndarray,
        *,
        feature_names: tuple[str, ...],
    ) -> "FeatureNormalization":
        values = np.asarray(values, dtype=np.float64)
        mask = np.asarray(mask, dtype=bool)
        if values.ndim != 2 or mask.shape != values.shape:
            raise ValueError("structured values and mask must be matching matrices")
        if values.shape[1] != len(feature_names):
            raise ValueError("feature_names do not match structured feature width")
        means: list[float] = []
        standard_deviations: list[float] = []
        for index in range(values.shape[1]):
            observed = values[mask[:, index], index]
            if observed.size:
                mean = float(np.mean(observed))
                std = float(np.std(observed))
            else:
                mean = 0.0
                std = 1.0
            means.append(mean)
            standard_deviations.append(max(std, 1e-6))
        return cls(
            mean=tuple(means),
            std=tuple(standard_deviations),
            feature_names=tuple(feature_names),
        )

    def transform(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if values.shape != mask.shape:
            raise ValueError("structured values and mask must have matching shapes")
        mean = torch.as_tensor(
            self.mean,
            dtype=values.dtype,
            device=values.device,
        )
        std = torch.as_tensor(
            self.std,
            dtype=values.dtype,
            device=values.device,
        )
        normalized = (values - mean) / std
        return torch.where(mask.bool(), normalized, torch.zeros_like(normalized))


class SharedEncoderLateFusionClassifier(nn.Module):
    """Encode three aligned views with one frozen encoder and fuse metadata."""

    is_hierarchical = True
    is_multimodal = True
    uses_internal_preprocessing = True

    def __init__(
        self,
        *,
        encoder: nn.Module,
        encoder_family: str,
        image_feature_dim: int,
        input_size: int,
        structured_feature_dim: int,
        feature_normalization: FeatureNormalization,
        structured_hidden_dim: int = 64,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.1,
        temperatures: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.encoder_family = str(encoder_family)
        self.image_feature_dim = int(image_feature_dim)
        self.input_size = int(input_size)
        self.structured_feature_dim = int(structured_feature_dim)
        self.feature_normalization = feature_normalization
        self.structured_hidden_dim = int(structured_hidden_dim)
        self.fusion_hidden_dim = int(fusion_hidden_dim)
        self.dropout = float(dropout)
        self.temperatures = {
            name: max(
                1e-4,
                float((temperatures or {}).get(name, 1.0)),
            )
            for name in (
                "review_action",
                "phenomenon_family",
                "detail_type",
            )
        }
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.structured_mlp = nn.Sequential(
            nn.Linear(self.structured_feature_dim * 2, structured_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(structured_hidden_dim),
        )
        concatenated_dim = self.image_feature_dim * 3 + structured_hidden_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(concatenated_dim),
            nn.Linear(concatenated_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = HierarchicalHeads(
            fusion_hidden_dim,
            hidden_dim=fusion_hidden_dim,
            dropout=dropout,
        )

    def _encode_views(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5 or images.shape[1] != 3:
            raise ValueError("multimodal images must have shape [B,3,C,H,W]")
        batch, view_count, channels, height, width = images.shape
        flattened = images.reshape(
            batch * view_count,
            channels,
            height,
            width,
        )
        prepared = preprocess_feature_batch(
            flattened,
            input_size=self.input_size,
        )
        self.encoder.eval()
        with torch.no_grad():
            encoded = forward_feature_encoder(
                self.encoder,
                prepared,
                family=self.encoder_family,
            )
        return encoded.reshape(batch, view_count * self.image_feature_dim)

    def forward(
        self,
        images: torch.Tensor,
        structured_values: torch.Tensor,
        structured_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if structured_values.shape[-1] != self.structured_feature_dim:
            raise ValueError("structured feature width does not match model")
        normalized = self.feature_normalization.transform(
            structured_values.float(),
            structured_mask,
        )
        structured_input = torch.cat(
            [normalized, structured_mask.float()],
            dim=-1,
        )
        structured_embedding = self.structured_mlp(structured_input)
        fused = self.fusion(
            torch.cat(
                [self._encode_views(images), structured_embedding],
                dim=-1,
            )
        )
        outputs = self.heads(fused)
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
    ) -> "SharedEncoderLateFusionClassifier":
        feature_encoder = str(
            checkpoint.get("feature_encoder") or "auto"
        )
        encoder, spec = load_feature_encoder(
            feature_encoder,
            device=device,
        )
        normalization = checkpoint.get("structured_normalization")
        if not isinstance(normalization, Mapping):
            raise RuntimeError(
                "multimodal checkpoint is missing structured normalization"
            )
        feature_names = tuple(
            str(item)
            for item in checkpoint.get("structured_features") or ()
        )
        if not feature_names:
            raise RuntimeError(
                "multimodal checkpoint is missing structured feature names"
            )
        mean = tuple(float(item) for item in normalization.get("mean") or ())
        std = tuple(float(item) for item in normalization.get("std") or ())
        if len(mean) != len(feature_names) or len(std) != len(feature_names):
            raise RuntimeError(
                "multimodal normalization width does not match features"
            )
        fusion_config = (
            checkpoint.get("fusion_config")
            if isinstance(checkpoint.get("fusion_config"), Mapping)
            else {}
        )
        model = cls(
            encoder=encoder,
            encoder_family=spec.family,
            image_feature_dim=spec.feature_dim,
            input_size=spec.input_size,
            structured_feature_dim=len(feature_names),
            feature_normalization=FeatureNormalization(
                mean=mean,
                std=std,
                feature_names=feature_names,
            ),
            structured_hidden_dim=int(
                fusion_config.get("structured_hidden_dim") or 64
            ),
            fusion_hidden_dim=int(
                fusion_config.get("fusion_hidden_dim") or 256
            ),
            dropout=float(fusion_config.get("dropout") or 0.1),
            temperatures=(
                checkpoint.get("temperatures")
                if isinstance(checkpoint.get("temperatures"), Mapping)
                else None
            ),
        ).to(device)
        states = checkpoint.get("trainable_states")
        if not isinstance(states, Mapping):
            raise RuntimeError(
                "multimodal checkpoint is missing trainable states"
            )
        for name, module in (
            ("structured_mlp", model.structured_mlp),
            ("fusion", model.fusion),
            ("heads", model.heads),
        ):
            state = states.get(name)
            if not isinstance(state, Mapping):
                raise RuntimeError(
                    f"multimodal checkpoint is missing {name} state"
                )
            module.load_state_dict(dict(state), strict=True)
        return model

    def checkpoint_metadata(
        self,
        *,
        feature_encoder: str,
        partition_id: str,
        partition_manifest_sha256: str,
        taxonomy_version: str,
    ) -> dict[str, Any]:
        return {
            "model_format": MULTIMODAL_MODEL_FORMAT,
            "feature_version": MULTIMODAL_FEATURE_VERSION,
            "feature_encoder": str(feature_encoder),
            "partition_id": str(partition_id),
            "partition_manifest_sha256": str(partition_manifest_sha256),
            "taxonomy_version": str(taxonomy_version),
            "input_views": ["new", "old", "signed_difference"],
            "structured_features": list(
                self.feature_normalization.feature_names
            ),
            "structured_normalization": {
                "mean": list(self.feature_normalization.mean),
                "std": list(self.feature_normalization.std),
            },
            "missing_feature_policy": "zero_after_normalization_with_mask",
        }


def build_multimodal_checkpoint(
    model: SharedEncoderLateFusionClassifier,
    *,
    feature_encoder: str,
    partition_id: str,
    partition_manifest_sha256: str,
    taxonomy_version: str,
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        **model.checkpoint_metadata(
            feature_encoder=feature_encoder,
            partition_id=partition_id,
            partition_manifest_sha256=partition_manifest_sha256,
            taxonomy_version=taxonomy_version,
        ),
        "task_type": "classification",
        "fusion_config": {
            "structured_hidden_dim": model.structured_hidden_dim,
            "fusion_hidden_dim": model.fusion_hidden_dim,
            "dropout": model.dropout,
        },
        "temperatures": dict(model.temperatures),
        "trainable_states": {
            "structured_mlp": model.structured_mlp.state_dict(),
            "fusion": model.fusion.state_dict(),
            "heads": model.heads.state_dict(),
        },
        "metrics": dict(metrics or {}),
        "gold_test_used_for_selection": False,
    }


def synchronized_triplet_transform(
    views: np.ndarray,
    *,
    rotation_quadrants: int = 0,
    horizontal_flip: bool = False,
    vertical_flip: bool = False,
) -> np.ndarray:
    """Apply identical geometry to all views, preserving correspondence."""

    transformed = np.array(views, copy=True)
    if transformed.ndim < 3 or transformed.shape[0] != 3:
        raise ValueError("triplet views must have leading view dimension of 3")
    if horizontal_flip:
        transformed = np.flip(transformed, axis=-1)
    if vertical_flip:
        transformed = np.flip(transformed, axis=-2)
    quadrants = int(rotation_quadrants) % 4
    if quadrants:
        transformed = np.rot90(
            transformed,
            quadrants,
            axes=(-2, -1),
        )
    return np.ascontiguousarray(transformed)
