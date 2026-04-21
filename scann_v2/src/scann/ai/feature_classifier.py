"""Frozen feature encoders and lightweight heads for extreme long-tail training."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


_DINO_ENCODERS: dict[str, dict[str, Any]] = {
    "dinov2_vitb14_reg": {"hub_name": "dinov2_vitb14_reg", "feature_dim": 768, "input_size": 224},
    "dinov2_vitl14_reg": {"hub_name": "dinov2_vitl14_reg", "feature_dim": 1024, "input_size": 224},
}
_CLIP_ENCODERS: dict[str, dict[str, Any]] = {
    "clip_vitl14": {"feature_dim": 768, "input_size": 224, "model_name": "ViT-L/14"},
    "clip_vit_l_14": {"feature_dim": 768, "input_size": 224, "model_name": "ViT-L/14"},
}
_TEST_ENCODERS: dict[str, dict[str, Any]] = {
    "scann_test_identity": {"feature_dim": 6, "input_size": 224},
}


@dataclass(frozen=True)
class FeatureEncoderSpec:
    name: str
    feature_dim: int
    input_size: int = 224
    family: str = "dinov2"


def normalize_feature_encoder_name(name: str | None) -> str:
    normalized = str(name or "").strip().lower().replace("-", "_")
    if normalized in {"", "auto"}:
        return "auto"
    aliases = {
        "dinov2_vit_b_14_reg": "dinov2_vitb14_reg",
        "dinov2_vit_l_14_reg": "dinov2_vitl14_reg",
        "clip_vit_l14": "clip_vitl14",
        "clip_vit_l_14": "clip_vitl14",
    }
    return aliases.get(normalized, normalized)


def auto_feature_encoder_name(device: torch.device | None = None) -> str:
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(device)
            memory_gb = float(props.total_memory) / (1024.0 ** 3)
            if memory_gb >= 16.0:
                return "dinov2_vitl14_reg"
        except Exception:
            logger.debug("Could not inspect CUDA memory for feature encoder selection", exc_info=True)
    return "dinov2_vitb14_reg"


def feature_encoder_spec(name: str | None, *, device: torch.device | None = None) -> FeatureEncoderSpec:
    normalized = normalize_feature_encoder_name(name)
    if normalized == "auto":
        normalized = auto_feature_encoder_name(device)
    if normalized in _DINO_ENCODERS:
        meta = _DINO_ENCODERS[normalized]
        return FeatureEncoderSpec(
            name=normalized,
            feature_dim=int(meta["feature_dim"]),
            input_size=int(meta["input_size"]),
            family="dinov2",
        )
    if normalized in _CLIP_ENCODERS:
        meta = _CLIP_ENCODERS[normalized]
        return FeatureEncoderSpec(
            name=normalized,
            feature_dim=int(meta["feature_dim"]),
            input_size=int(meta["input_size"]),
            family="clip",
        )
    if normalized in _TEST_ENCODERS:
        meta = _TEST_ENCODERS[normalized]
        return FeatureEncoderSpec(
            name=normalized,
            feature_dim=int(meta["feature_dim"]),
            input_size=int(meta["input_size"]),
            family="test",
        )
    raise ValueError(f"Unsupported feature encoder: {name}")


class FeatureHeadClassifier(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        *,
        hidden_dim: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        feature_dim = int(feature_dim)
        num_classes = int(num_classes)
        hidden_dim = int(hidden_dim or 0)
        dropout = max(0.0, float(dropout or 0.0))

        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, num_classes),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class _TestIdentityFeatureEncoder(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        means = x.mean(dim=(-2, -1))
        stds = x.std(dim=(-2, -1), unbiased=False)
        return torch.cat([means, stds], dim=1)


def load_feature_encoder(name: str | None, *, device: torch.device) -> tuple[nn.Module, FeatureEncoderSpec]:
    spec = feature_encoder_spec(name, device=device)
    if spec.family == "dinov2":
        meta = _DINO_ENCODERS[spec.name]
        encoder = torch.hub.load("facebookresearch/dinov2", str(meta["hub_name"]), pretrained=True)
    elif spec.family == "clip":
        try:
            import clip  # type: ignore
        except Exception as exc:
            raise RuntimeError("CLIP feature encoder requires the optional 'clip' package") from exc
        model, _preprocess = clip.load(_CLIP_ENCODERS[spec.name]["model_name"], device=device)
        encoder = model.visual
    elif spec.family == "test":
        encoder = _TestIdentityFeatureEncoder()
    else:
        raise ValueError(f"Unsupported feature encoder family: {spec.family}")

    encoder = encoder.to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder, spec


def preprocess_feature_batch(x: torch.Tensor, *, input_size: int) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError("feature encoder input must be a BCHW tensor")
    x = x.float()
    if x.shape[-2:] != (int(input_size), int(input_size)):
        x = F.interpolate(x, size=(int(input_size), int(input_size)), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x.clamp(0.0, 1.0) - mean) / std


def forward_feature_encoder(encoder: nn.Module, x: torch.Tensor, *, family: str) -> torch.Tensor:
    if family == "dinov2" and hasattr(encoder, "forward_features"):
        output = encoder.forward_features(x)
    elif family == "clip" and hasattr(encoder, "encode_image"):
        output = encoder.encode_image(x)
    else:
        output = encoder(x)

    if isinstance(output, dict):
        for key in ("x_norm_clstoken", "cls_token", "pooled", "last_hidden_state"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                output = value[:, 0] if value.ndim == 3 else value
                break
    if isinstance(output, (list, tuple)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise RuntimeError("feature encoder did not return a tensor")
    if output.ndim > 2:
        output = output.flatten(1)
    return output.float()


def apply_prior_logit_correction(
    logits: torch.Tensor,
    class_log_prior: list[float] | tuple[float, ...] | torch.Tensor | None,
    *,
    tau: float = 1.0,
) -> torch.Tensor:
    if class_log_prior is None or float(tau) <= 0.0:
        return logits
    prior = torch.as_tensor(class_log_prior, dtype=logits.dtype, device=logits.device)
    if prior.ndim != 1 or prior.numel() != logits.shape[-1]:
        return logits
    return logits - float(tau) * prior.view(1, -1)


class FrozenFeaturePatchClassifier(nn.Module):
    """Inference-time wrapper: frozen encoder + trained feature-space head."""

    uses_internal_preprocessing = True

    def __init__(
        self,
        *,
        feature_encoder: str,
        head: FeatureHeadClassifier,
        device: torch.device,
        class_log_prior: list[float] | None = None,
        prior_correction_tau: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder, self.spec = load_feature_encoder(feature_encoder, device=device)
        self.head = head.to(device)
        self.head.eval()
        self.class_log_prior = class_log_prior
        self.prior_correction_tau = float(prior_correction_tau or 0.0)
        self.device = device

    @classmethod
    def from_checkpoint(cls, checkpoint: dict[str, Any], *, device: torch.device) -> "FrozenFeaturePatchClassifier":
        feature_encoder = str(checkpoint.get("feature_encoder") or "auto")
        feature_dim = int(checkpoint.get("feature_dim") or 0)
        class_names = checkpoint.get("class_names")
        num_classes = len(class_names) if isinstance(class_names, list) and class_names else int(checkpoint.get("num_classes") or 2)
        head_config = checkpoint.get("feature_head_config") if isinstance(checkpoint.get("feature_head_config"), dict) else {}
        if feature_dim <= 0:
            feature_dim = feature_encoder_spec(feature_encoder, device=device).feature_dim
        head = FeatureHeadClassifier(
            feature_dim=feature_dim,
            num_classes=num_classes,
            hidden_dim=int(head_config.get("hidden_dim") or 0),
            dropout=float(head_config.get("dropout") or 0.0),
        )
        state = checkpoint.get("head_state")
        if state is None:
            raw_state = checkpoint.get("state")
            state = raw_state.get("head") if isinstance(raw_state, dict) else None
        if not isinstance(state, dict):
            raise RuntimeError("frozen feature checkpoint is missing head_state")
        head.load_state_dict(state, strict=True)

        prior_config = checkpoint.get("prior_logit_correction") if isinstance(checkpoint.get("prior_logit_correction"), dict) else {}
        tau = float(checkpoint.get("prior_correction_tau", prior_config.get("tau", 0.0)) or 0.0)
        if prior_config and prior_config.get("enabled") is False:
            tau = 0.0
        return cls(
            feature_encoder=feature_encoder,
            head=head,
            device=device,
            class_log_prior=checkpoint.get("class_log_prior") if isinstance(checkpoint.get("class_log_prior"), list) else None,
            prior_correction_tau=tau,
        )

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        prepared = preprocess_feature_batch(x.to(self.device), input_size=self.spec.input_size)
        return forward_feature_encoder(self.encoder, prepared, family=self.spec.family)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        logits = self.head(features)
        return apply_prior_logit_correction(
            logits,
            self.class_log_prior,
            tau=self.prior_correction_tau,
        )
