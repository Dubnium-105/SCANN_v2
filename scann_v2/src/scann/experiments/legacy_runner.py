"""Training and evaluation entry points for legacy v1 triplet experiments."""

from __future__ import annotations

import copy
import csv
import io
import importlib.util
import json
import logging
import math
import random
import sys
import time
import types
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models
from torchvision.models import swin_transformer
from torchvision.models.vision_transformer import interpolate_embeddings

from scann.ai.device_utils import resolve_device
from scann.ai.trainer import FocalLoss

from .legacy_dataset import (
    LegacyTripletExperimentDataset,
    format_image_size_spec,
    image_size_specs_equal,
    normalize_image_size_spec,
)
from .legacy_manifest import build_legacy_triplet_manifest, load_legacy_manifest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_DIR = PROJECT_ROOT.parent / "dataset"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments"
DEFAULT_MANIFEST_PATH = DEFAULT_OUTPUT_ROOT / "manifests" / "legacy_v1_manifest.json"
SUMMARY_COLUMNS = [
    "experiment_name",
    "model_name",
    "input_mode",
    "image_size",
    "resize_mode",
    "normalize",
    "attention_compression_mode",
    "attention_layer_selector",
    "attention_layer_count",
    "enabled_layer_indices",
    "kv_bits",
    "group_size",
    "token_block_size",
    "preserve_cls_token",
    "quantize_k",
    "quantize_v",
    "streaming_enabled",
    "materialize_attention_matrix",
    "pretrained",
    "seed",
    "batch_size",
    "lr",
    "epochs_requested",
    "epochs_ran",
    "best_epoch",
    "best_threshold",
    "val_accuracy",
    "val_precision",
    "val_recall",
    "val_f1",
    "val_f2",
    "val_roc_auc",
    "val_pr_auc",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_f2",
    "test_roc_auc",
    "test_pr_auc",
    "test_tn",
    "test_fp",
    "test_fn",
    "test_tp",
    "avg_epoch_seconds",
    "peak_gpu_memory_mb",
    "peak_gpu_memory_attention_only_mb",
    "packed_kv_size_mb",
    "token_count",
    "num_patched_layers",
    "manifest_path",
    "checkpoint_path",
    "history_path",
    "plots_dir",
    "split_distribution_plot",
    "learning_curve_plot",
    "val_analysis_plot",
    "test_analysis_plot",
    "val_predictions_path",
    "test_predictions_path",
]
INPUT_FUSION_VARIANTS = [
    ("new_old_diff", "new+old+diff"),
    ("diff_only", "diff"),
    ("diff_new", "diff+new"),
    ("diff_old", "diff+old"),
]
INPUT_FUSION_COLUMNS = [
    "variant",
    "input_mode",
    "experiment_name",
    "result_source",
    "model_name",
    "pretrained",
    "seed",
    "image_size",
    "resize_mode",
    "batch_size",
    "epochs_requested",
    "epochs_ran",
    "best_epoch",
    "best_threshold",
    "val_accuracy",
    "val_recall",
    "val_f1",
    "val_roc_auc",
    "test_accuracy",
    "test_recall",
    "test_f1",
    "test_roc_auc",
    "summary_path",
    "checkpoint_path",
]
PREPROCESSING_VARIANTS = [
    {
        "variant": "native_keep_80",
        "description": "native 80x80",
        "image_size": 80,
        "resize_mode": "keep",
        "normalize": True,
    },
    {
        "variant": "resize_224",
        "description": "resize 224",
        "image_size": 224,
        "resize_mode": "resize",
        "normalize": True,
    },
    {
        "variant": "pad_resize_224",
        "description": "pad_resize 224",
        "image_size": 224,
        "resize_mode": "pad_resize",
        "normalize": True,
    },
    {
        "variant": "resize_224_no_norm",
        "description": "resize 224 no_norm",
        "image_size": 224,
        "resize_mode": "resize",
        "normalize": False,
    },
]
PREPROCESSING_COLUMNS = [
    "variant",
    "description",
    "experiment_name",
    "result_source",
    "model_name",
    "pretrained",
    "input_mode",
    "seed",
    "image_size",
    "resize_mode",
    "normalize",
    "batch_size",
    "epochs_requested",
    "epochs_ran",
    "best_epoch",
    "best_threshold",
    "val_accuracy",
    "val_recall",
    "val_f1",
    "val_roc_auc",
    "test_accuracy",
    "test_recall",
    "test_f1",
    "test_roc_auc",
    "avg_epoch_seconds",
    "peak_gpu_memory_mb",
    "summary_path",
    "checkpoint_path",
]
MODEL_SCALE_VARIANTS = [
    {
        "variant": "tiny",
        "description": "Swin-T",
        "model_name": "swin_t",
    },
    {
        "variant": "small",
        "description": "Swin-S",
        "model_name": "swin_s",
    },
    {
        "variant": "base",
        "description": "Swin-B",
        "model_name": "swin_b",
    },
]
MODEL_SCALE_COLUMNS = [
    "variant",
    "description",
    "experiment_name",
    "result_source",
    "model_name",
    "pretrained",
    "input_mode",
    "seed",
    "image_size",
    "resize_mode",
    "batch_size",
    "epochs_requested",
    "epochs_ran",
    "best_epoch",
    "best_threshold",
    "val_accuracy",
    "val_recall",
    "val_f1",
    "val_roc_auc",
    "test_accuracy",
    "test_recall",
    "test_f1",
    "test_roc_auc",
    "params",
    "avg_epoch_seconds",
    "peak_gpu_memory_mb",
    "summary_path",
    "checkpoint_path",
]
QUANTIZATION_COLUMNS = [
    "variant",
    "description",
    "result_source",
    "status",
    "runnable",
    "model_name",
    "experiment_name",
    "split",
    "threshold",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_roc_auc",
    "delta_accuracy_vs_fp32",
    "delta_f1_vs_fp32",
    "cpu_ms_per_image",
    "gpu_ms_per_image",
    "cpu_peak_memory_mb",
    "gpu_peak_memory_mb",
    "model_state_size_mb",
    "checkpoint_path",
    "notes",
]
VIT_ATTENTION_ABLATION_COLUMNS = [
    "variant",
    "description",
    "result_source",
    "status",
    "supported",
    "layer_scope",
    "kv_target",
    "cls_policy",
    "precision_mode",
    "streaming_enabled",
    "materialize_attention_matrix",
    "experiment_name",
    "model_name",
    "image_size",
    "attention_compression_mode",
    "attention_layer_selector",
    "attention_layer_count",
    "enabled_layer_indices",
    "kv_bits",
    "group_size",
    "token_block_size",
    "preserve_cls_token",
    "quantize_k",
    "quantize_v",
    "split",
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "ms_per_image",
    "peak_cpu_memory_mb",
    "peak_gpu_memory_mb",
    "peak_gpu_memory_attention_only_mb",
    "packed_kv_size_mb",
    "token_count",
    "num_patched_layers",
    "checkpoint_path",
    "notes",
]
VIT_ATTENTION_ABLATION_VARIANTS = [
    {
        "variant": "baseline_dense",
        "description": "dense bf16 baseline",
        "layer_scope": "all_layers",
        "kv_target": "none",
        "cls_policy": "patch_plus_cls",
        "precision_mode": "bf16 baseline",
        "streaming_enabled": False,
        "materialize_attention_matrix": True,
        "attention_compression_mode": "none",
        "quantize_k": False,
        "quantize_v": False,
        "preserve_cls_token": False,
    },
    {
        "variant": "all_layers_k_only_4bit",
        "description": "full-module K-only packed streaming",
        "layer_scope": "all_layers",
        "kv_target": "K only",
        "cls_policy": "patch_plus_cls",
        "precision_mode": "K 4-bit",
        "streaming_enabled": True,
        "materialize_attention_matrix": False,
        "attention_compression_mode": "vit_packed_kv",
        "quantize_k": True,
        "quantize_v": False,
        "preserve_cls_token": False,
    },
    {
        "variant": "all_layers_v_only_4bit",
        "description": "full-module V-only packed streaming",
        "layer_scope": "all_layers",
        "kv_target": "V only",
        "cls_policy": "patch_plus_cls",
        "precision_mode": "KV 4-bit",
        "streaming_enabled": True,
        "materialize_attention_matrix": False,
        "attention_compression_mode": "vit_packed_kv",
        "quantize_k": False,
        "quantize_v": True,
        "preserve_cls_token": False,
    },
    {
        "variant": "all_layers_kv_4bit",
        "description": "full-module K/V packed streaming",
        "layer_scope": "all_layers",
        "kv_target": "K/V both",
        "cls_policy": "patch_plus_cls",
        "precision_mode": "KV 4-bit",
        "streaming_enabled": True,
        "materialize_attention_matrix": False,
        "attention_compression_mode": "vit_packed_kv",
        "quantize_k": True,
        "quantize_v": True,
        "preserve_cls_token": False,
    },
    {
        "variant": "all_layers_kv_4bit_patch_only",
        "description": "patch-token packed K/V with dense cls token",
        "layer_scope": "all_layers",
        "kv_target": "K/V both",
        "cls_policy": "patch_only",
        "precision_mode": "KV 4-bit",
        "streaming_enabled": True,
        "materialize_attention_matrix": False,
        "attention_compression_mode": "vit_packed_kv",
        "quantize_k": True,
        "quantize_v": True,
        "preserve_cls_token": True,
    },
    {
        "variant": "all_layers_kv_4bit_cls_preserved",
        "description": "full-module K/V packed streaming with cls preserved",
        "layer_scope": "all_layers",
        "kv_target": "K/V both",
        "cls_policy": "cls_preserved",
        "precision_mode": "KV 4-bit",
        "streaming_enabled": True,
        "materialize_attention_matrix": False,
        "attention_compression_mode": "vit_packed_kv",
        "quantize_k": True,
        "quantize_v": True,
        "preserve_cls_token": True,
    },
    {
        "variant": "first_25pct_kv_4bit",
        "description": "first quarter of ViT blocks with K/V packed streaming",
        "layer_scope": "first_25pct",
        "kv_target": "K/V both",
        "cls_policy": "patch_plus_cls",
        "precision_mode": "KV 4-bit",
        "streaming_enabled": True,
        "materialize_attention_matrix": False,
        "attention_compression_mode": "vit_packed_kv",
        "quantize_k": True,
        "quantize_v": True,
        "preserve_cls_token": False,
    },
    {
        "variant": "middle_50pct_kv_4bit",
        "description": "middle half of ViT blocks with K/V packed streaming",
        "layer_scope": "middle_50pct",
        "kv_target": "K/V both",
        "cls_policy": "patch_plus_cls",
        "precision_mode": "KV 4-bit",
        "streaming_enabled": True,
        "materialize_attention_matrix": False,
        "attention_compression_mode": "vit_packed_kv",
        "quantize_k": True,
        "quantize_v": True,
        "preserve_cls_token": False,
    },
    {
        "variant": "last_25pct_kv_4bit",
        "description": "last quarter of ViT blocks with K/V packed streaming",
        "layer_scope": "last_25pct",
        "kv_target": "K/V both",
        "cls_policy": "patch_plus_cls",
        "precision_mode": "KV 4-bit",
        "streaming_enabled": True,
        "materialize_attention_matrix": False,
        "attention_compression_mode": "vit_packed_kv",
        "quantize_k": True,
        "quantize_v": True,
        "preserve_cls_token": False,
    },
    {
        "variant": "custom_indices_kv_4bit",
        "description": "custom ViT block indices with K/V packed streaming",
        "layer_scope": "custom_indices",
        "kv_target": "K/V both",
        "cls_policy": "patch_plus_cls",
        "precision_mode": "KV 4-bit",
        "streaming_enabled": True,
        "materialize_attention_matrix": False,
        "attention_compression_mode": "vit_packed_kv",
        "quantize_k": True,
        "quantize_v": True,
        "preserve_cls_token": False,
    },
]


@dataclass
class LegacyExperimentConfig:
    """Serializable config for the legacy experiment pipeline."""

    experiment_name: str = "legacy_v1_experiment"
    dataset_dir: str = str(DEFAULT_DATASET_DIR)
    manifest_path: str = str(DEFAULT_MANIFEST_PATH)
    output_root: str = str(DEFAULT_OUTPUT_ROOT)
    model_name: str = "resnet18"
    pretrained: bool = False
    input_mode: str = "new_old_diff"
    image_size: int | list[int] | str = 224
    resize_mode: str = "resize"
    normalize: bool = True
    attention_compression_mode: str = "none"
    attention_layer_selector: str = "all"
    attention_layer_count: int = 0
    enabled_layer_indices: list[int] = field(default_factory=list)
    kv_bits: int = 4
    group_size: int = 32
    token_block_size: int = 64
    preserve_cls_token: bool = False
    quantize_k: bool = True
    quantize_v: bool = True
    streaming_enabled: bool = True
    materialize_attention_matrix: bool = False
    batch_size: int = 32
    epochs: int = 30
    lr: float = 2e-4
    weight_decay: float = 1e-3
    optimizer: str = "AdamW"
    scheduler: str = "cosine"
    step_size: int = 10
    step_gamma: float = 0.5
    early_stopping_patience: int = 10
    selection_metric: str = "f1"
    threshold_metric: str = "f1"
    loss_name: str = "cross_entropy"
    focal_gamma: float = 2.0
    focal_alpha: list[float] = field(default_factory=lambda: [1.0, 1.5])
    use_weighted_sampler: bool = True
    augment: bool = True
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.5
    enable_rotate_90: bool = True
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    summary_csv_name: str = "experiment_results.csv"
    log_every_n_steps: int = 25


def _resolve_relative_path(value: str | None, base_dir: Path) -> str:
    if not value:
        return ""
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def load_experiment_config(config: str | Path | dict[str, Any]) -> LegacyExperimentConfig:
    """Load a config from json/yaml or build one from a mapping."""

    base_dir = Path.cwd()
    raw: dict[str, Any]
    if isinstance(config, dict):
        raw = dict(config)
    else:
        config_path = Path(config).resolve()
        base_dir = config_path.parent
        suffix = config_path.suffix.lower()
        if suffix == ".json":
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("YAML config requires PyYAML to be installed") from exc
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        else:
            raise ValueError(f"Unsupported config format: {config_path}")

    for key in ("dataset_dir", "manifest_path", "output_root"):
        if key in raw:
            raw[key] = _resolve_relative_path(raw.get(key), base_dir)
    if "image_size" in raw:
        raw["image_size"] = normalize_image_size_spec(raw["image_size"])
    if "enabled_layer_indices" in raw and raw["enabled_layer_indices"] is not None:
        raw["enabled_layer_indices"] = [int(value) for value in raw["enabled_layer_indices"]]

    return LegacyExperimentConfig(**raw)


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_model_name(model_name: str) -> str:
    normalized = str(model_name).strip().lower()
    aliases = {
        "cnn_baseline": "resnet18",
        "legacy_cnn": "resnet18",
        "legacy_resnet18": "resnet18",
        "vit_baseline": "vit_b_16",
        "vit_b16": "vit_b_16",
        "vit_h14": "vit_h_14",
        "vit_huge": "vit_h_14",
        "swin_tiny": "swin_t",
        "swin_small": "swin_s",
        "swin_base": "swin_b",
    }
    return aliases.get(normalized, normalized)


def _resolve_square_model_image_size(image_size: int | list[int] | str) -> int:
    normalized = normalize_image_size_spec(image_size)
    if normalized == "keep":
        raise ValueError(
            "image_size='keep' cannot be used for model creation; provide an explicit numeric size"
        )
    if isinstance(normalized, list):
        if int(normalized[0]) != int(normalized[1]):
            raise ValueError(
                f"Current model factory only supports square image_size for model creation, got {normalized}"
            )
        return int(normalized[0])
    return int(normalized)


def _weights_default_image_size(weights: Any, *, fallback: int) -> int:
    crop_size = getattr(weights, "meta", {}).get("min_size")
    if isinstance(crop_size, (list, tuple)) and crop_size:
        return int(crop_size[0])

    transforms_factory = getattr(weights, "transforms", None)
    if callable(transforms_factory):
        transforms = transforms_factory()
        crop_size = getattr(transforms, "crop_size", None)
        if isinstance(crop_size, (list, tuple)) and crop_size:
            return int(crop_size[0])
        if isinstance(crop_size, int):
            return int(crop_size)

    return int(fallback)


def _interpolate_vit_state_dict(
    *,
    weights: Any,
    image_size: int,
    patch_size: int,
) -> dict[str, Any]:
    state_dict = weights.get_state_dict(progress=True)
    return interpolate_embeddings(
        image_size=int(image_size),
        patch_size=int(patch_size),
        model_state=state_dict,
        reset_heads=True,
    )


def _create_vit_classifier(
    *,
    model_builder: Any,
    weights_enum: Any,
    image_size: int | list[int] | str,
    patch_size: int,
    pretrained: bool,
) -> nn.Module:
    resolved_image_size = _resolve_square_model_image_size(image_size)

    if pretrained:
        weights = weights_enum.DEFAULT
        default_image_size = _weights_default_image_size(weights, fallback=resolved_image_size)
        if resolved_image_size != default_image_size:
            model = model_builder(weights=None, image_size=resolved_image_size)
            state_dict = _interpolate_vit_state_dict(
                weights=weights,
                image_size=resolved_image_size,
                patch_size=patch_size,
            )
            model.load_state_dict(state_dict, strict=False)
        else:
            model = model_builder(weights=weights, image_size=resolved_image_size)
    else:
        model = model_builder(weights=None, image_size=resolved_image_size)

    model.heads.head = nn.Linear(model.heads.head.in_features, 2)
    return model


def create_experiment_model(
    model_name: str,
    *,
    pretrained: bool,
    image_size: int | list[int] | str = 224,
) -> nn.Module:
    """Create a torchvision classifier for the experiment."""

    normalized = _normalize_model_name(model_name)
    if normalized == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model
    if normalized == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model
    if normalized == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model
    if normalized == "vit_b_16":
        return _create_vit_classifier(
            model_builder=models.vit_b_16,
            weights_enum=models.ViT_B_16_Weights,
            image_size=image_size,
            patch_size=16,
            pretrained=pretrained,
        )
    if normalized == "vit_h_14":
        return _create_vit_classifier(
            model_builder=models.vit_h_14,
            weights_enum=models.ViT_H_14_Weights,
            image_size=image_size,
            patch_size=14,
            pretrained=pretrained,
        )
    if normalized == "swin_t":
        weights = models.Swin_T_Weights.DEFAULT if pretrained else None
        model = models.swin_t(weights=None)
        if weights is not None:
            model.load_state_dict(weights.get_state_dict(progress=True, check_hash=False))
        model.head = nn.Linear(model.head.in_features, 2)
        return model
    if normalized == "swin_s":
        weights = models.Swin_S_Weights.DEFAULT if pretrained else None
        model = models.swin_s(weights=None)
        if weights is not None:
            model.load_state_dict(weights.get_state_dict(progress=True, check_hash=False))
        model.head = nn.Linear(model.head.in_features, 2)
        return model
    if normalized == "swin_b":
        weights = models.Swin_B_Weights.DEFAULT if pretrained else None
        model = models.swin_b(weights=None)
        if weights is not None:
            model.load_state_dict(weights.get_state_dict(progress=True, check_hash=False))
        model.head = nn.Linear(model.head.in_features, 2)
        return model
    raise ValueError(f"Unsupported model name: {model_name}")


def create_vit_attention_compression_model(
    base_model: nn.Module,
    *,
    layer_selector: str = "all",
    count: int | None = None,
    explicit_indices: list[int] | tuple[int, ...] | None = None,
    attention_adapter: Any = None,
) -> nn.Module:
    from .vit_attention_compression import (
        ViTAttentionPatchConfig,
        create_vit_attention_compression_model as _create_vit_attention_compression_model,
    )

    patch_config = ViTAttentionPatchConfig(
        layer_selector=layer_selector,
        count=count,
        explicit_indices=tuple(explicit_indices or ()),
    )
    return _create_vit_attention_compression_model(
        base_model,
        config=patch_config,
        attention_adapter=attention_adapter,
    )


def create_vit_packed_kv_attention_model(
    base_model: nn.Module,
    *,
    layer_selector: str = "all",
    count: int | None = None,
    explicit_indices: list[int] | tuple[int, ...] | None = None,
    group_size: int = 32,
    block_size: int = 64,
    quantize_k: bool = True,
    quantize_v: bool = True,
    preserve_cls_token: bool = False,
    compute_dtype: torch.dtype = torch.float32,
) -> nn.Module:
    from .vit_attention_compression import (
        PackedKVAttentionConfig,
        ViTAttentionPatchConfig,
        create_vit_packed_kv_compression_model,
    )

    patch_config = ViTAttentionPatchConfig(
        layer_selector=layer_selector,
        count=count,
        explicit_indices=tuple(explicit_indices or ()),
    )
    packed_kv_config = PackedKVAttentionConfig(
        group_size=group_size,
        block_size=block_size,
        quantize_k=quantize_k,
        quantize_v=quantize_v,
        preserve_cls_token=preserve_cls_token,
        compute_dtype=compute_dtype,
    )
    return create_vit_packed_kv_compression_model(
        base_model,
        patch_config=patch_config,
        packed_kv_config=packed_kv_config,
    )


def _attention_compression_enabled(config: LegacyExperimentConfig) -> bool:
    return str(config.attention_compression_mode).strip().lower() not in {"", "none"}


def _vit_encoder_layer_count(model_name: str) -> int:
    normalized = _normalize_model_name(model_name)
    if normalized == "vit_b_16":
        return 12
    if normalized == "vit_h_14":
        return 32
    raise ValueError(f"ViT attention ablation currently only supports torchvision ViT models, got {model_name!r}")


def _resolve_ablation_layer_config(
    *,
    layer_scope: str,
    model_name: str,
    custom_indices: list[int] | None = None,
) -> tuple[str, int, list[int]]:
    normalized_scope = str(layer_scope).strip().lower()
    total_layers = _vit_encoder_layer_count(model_name)
    if normalized_scope == "all_layers":
        return "all", 0, []
    if normalized_scope == "first_25pct":
        return "first_n", max(1, math.ceil(total_layers * 0.25)), []
    if normalized_scope == "middle_50pct":
        return "middle", max(1, math.ceil(total_layers * 0.50)), []
    if normalized_scope == "last_25pct":
        return "last_n", max(1, math.ceil(total_layers * 0.25)), []
    if normalized_scope == "custom_indices":
        indices = [int(index) for index in (custom_indices or [0])]
        return "explicit_indices", 0, indices
    raise ValueError(f"Unsupported ViT attention ablation layer_scope: {layer_scope}")


def _build_vit_attention_ablation_variants(
    config: LegacyExperimentConfig,
    *,
    variants: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    resolved_variants = variants or VIT_ATTENTION_ABLATION_VARIANTS
    rows: list[dict[str, Any]] = []
    for variant_spec in resolved_variants:
        layer_selector, layer_count, explicit_indices = _resolve_ablation_layer_config(
            layer_scope=str(variant_spec["layer_scope"]),
            model_name=config.model_name,
            custom_indices=list(config.enabled_layer_indices),
        )
        notes = ""
        if str(variant_spec["layer_scope"]).strip().lower() == "custom_indices" and not config.enabled_layer_indices:
            notes = "enabled_layer_indices was empty; defaulted custom_indices to [0]"
        rows.append(
            {
                "variant": str(variant_spec["variant"]),
                "description": str(variant_spec["description"]),
                "layer_scope": str(variant_spec["layer_scope"]),
                "kv_target": str(variant_spec["kv_target"]),
                "cls_policy": str(variant_spec["cls_policy"]),
                "precision_mode": str(variant_spec["precision_mode"]),
                "streaming_enabled": bool(variant_spec["streaming_enabled"]),
                "materialize_attention_matrix": bool(variant_spec["materialize_attention_matrix"]),
                "attention_compression_mode": str(variant_spec["attention_compression_mode"]),
                "attention_layer_selector": layer_selector,
                "attention_layer_count": int(layer_count),
                "enabled_layer_indices": list(explicit_indices),
                "quantize_k": bool(variant_spec["quantize_k"]),
                "quantize_v": bool(variant_spec["quantize_v"]),
                "preserve_cls_token": bool(variant_spec["preserve_cls_token"]),
                "kv_bits": 4 if str(variant_spec["attention_compression_mode"]).strip().lower() != "none" else 0,
                "group_size": int(config.group_size),
                "token_block_size": int(config.token_block_size),
                "notes": notes,
            }
        )
    return rows


def _apply_attention_compression_from_config(
    model: nn.Module,
    config: LegacyExperimentConfig,
    *,
    for_training: bool,
) -> nn.Module:
    mode = str(config.attention_compression_mode).strip().lower()
    if mode in {"", "none"}:
        return model

    normalized_model_name = _normalize_model_name(config.model_name)
    if not normalized_model_name.startswith("vit_"):
        raise ValueError(
            f"attention_compression_mode={config.attention_compression_mode!r} currently only supports ViT models"
        )
    if for_training:
        raise ValueError(
            "attention_compression_mode is currently inference-only; training with packed KV attention is not yet supported"
        )
    if int(config.kv_bits) != 4:
        raise ValueError("Current packed KV attention path only supports kv_bits=4")
    if not bool(config.streaming_enabled):
        raise ValueError("Current packed KV attention path requires streaming_enabled=True")
    if bool(config.materialize_attention_matrix):
        raise ValueError("Current packed KV attention path requires materialize_attention_matrix=False")

    explicit_indices = list(config.enabled_layer_indices or [])
    layer_count = int(config.attention_layer_count) if int(config.attention_layer_count) > 0 else None
    if mode == "vit_packed_kv":
        return create_vit_packed_kv_attention_model(
            model,
            layer_selector=config.attention_layer_selector,
            count=layer_count,
            explicit_indices=explicit_indices,
            group_size=int(config.group_size),
            block_size=int(config.token_block_size),
            quantize_k=bool(config.quantize_k),
            quantize_v=bool(config.quantize_v),
            preserve_cls_token=bool(config.preserve_cls_token),
            compute_dtype=torch.float32,
        )
    raise ValueError(f"Unsupported attention_compression_mode: {config.attention_compression_mode}")


def _estimate_vit_token_count_from_input(model: nn.Module, x: torch.Tensor) -> int:
    patch_size = getattr(model, "patch_size", None)
    if patch_size is None:
        return 0
    if isinstance(patch_size, int):
        patch_h = patch_w = int(patch_size)
    else:
        patch_h = int(patch_size[0])
        patch_w = int(patch_size[1])
    height = int(x.shape[-2])
    width = int(x.shape[-1])
    return int((height // patch_h) * (width // patch_w) + 1)


def _collect_attention_runtime_stats(model: nn.Module) -> dict[str, float]:
    packed_kv_size_bytes = 0
    attention_memory_bytes = 0
    token_count = 0
    for module in model.modules():
        packed_kv_size_bytes += int(getattr(module, "_vit_last_packed_kv_size_bytes", 0) or 0)
        attention_memory_bytes = max(
            attention_memory_bytes,
            int(getattr(module, "_vit_last_attention_memory_bytes", 0) or 0),
        )
        token_count = max(token_count, int(getattr(module, "_vit_last_token_count", 0) or 0))
    return {
        "packed_kv_size_mb": float(packed_kv_size_bytes) / (1024.0 * 1024.0),
        "peak_gpu_memory_attention_only_mb": float(attention_memory_bytes) / (1024.0 * 1024.0),
        "token_count": int(token_count),
        "num_patched_layers": int(len(getattr(model, "_vit_attention_patched_indices", []))),
    }


def _criterion_from_config(config: LegacyExperimentConfig) -> nn.Module:
    loss_name = str(config.loss_name).strip().lower()
    if loss_name == "focal":
        return FocalLoss(gamma=float(config.focal_gamma), alpha=list(config.focal_alpha))
    return nn.CrossEntropyLoss()


def _optimizer_from_config(model: nn.Module, config: LegacyExperimentConfig) -> optim.Optimizer:
    optimizer_name = str(config.optimizer).strip().lower()
    if optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)
    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    return optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)


def _scheduler_from_config(
    optimizer: optim.Optimizer,
    config: LegacyExperimentConfig,
) -> optim.lr_scheduler._LRScheduler | None:
    scheduler_name = str(config.scheduler).strip().lower()
    if scheduler_name == "none":
        return None
    if scheduler_name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, int(config.step_size)),
            gamma=float(config.step_gamma),
        )
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(config.epochs)))


def _compute_binary_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int32)
    probs = np.asarray(probs, dtype=np.float64)
    preds = (probs >= threshold).astype(np.int32)

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in cm.ravel()]
    eps = 1e-12

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2.0 * precision * recall) / (precision + recall + eps)
    f2 = (5.0 * precision * recall) / ((4.0 * precision) + recall + eps)

    if len(np.unique(labels)) > 1:
        roc_auc = float(roc_auc_score(labels, probs))
        pr_auc = float(average_precision_score(labels, probs))
    else:
        roc_auc = 0.0
        pr_auc = 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f2": float(f2),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "threshold": float(threshold),
    }


def _choose_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
    *,
    metric_name: str,
) -> tuple[float, dict[str, Any]]:
    labels = np.asarray(labels, dtype=np.int32)
    probs = np.asarray(probs, dtype=np.float64)

    if probs.size == 0:
        raise ValueError("Cannot choose a threshold from empty predictions")

    thresholds = np.unique(np.concatenate([probs, np.array([0.5], dtype=np.float64)]))
    if thresholds.size > 257:
        quantiles = np.linspace(0.0, 1.0, 257)
        thresholds = np.unique(np.quantile(thresholds, quantiles))

    metric_key = str(metric_name).strip().lower()
    best_threshold = 0.5
    best_metrics = _compute_binary_metrics(labels, probs, threshold=best_threshold)
    best_score = float(best_metrics.get(metric_key, best_metrics["f1"]))

    for threshold in thresholds:
        metrics = _compute_binary_metrics(labels, probs, threshold=float(threshold))
        score = float(metrics.get(metric_key, metrics["f1"]))
        if score > best_score + 1e-9:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_score = score
        elif abs(score - best_score) <= 1e-9 and metrics["recall"] > best_metrics["recall"] + 1e-9:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_score = score

    return best_threshold, best_metrics


def _build_datasets(config: LegacyExperimentConfig) -> dict[str, LegacyTripletExperimentDataset]:
    return {
        "train": LegacyTripletExperimentDataset(
            config.manifest_path,
            split="train",
            dataset_root=config.dataset_dir,
            input_mode=config.input_mode,
            image_size=config.image_size,
            resize_mode=config.resize_mode,
            normalize=config.normalize,
            augment=config.augment,
            horizontal_flip_prob=config.horizontal_flip_prob,
            vertical_flip_prob=config.vertical_flip_prob,
            enable_rotate_90=config.enable_rotate_90,
        ),
        "val": LegacyTripletExperimentDataset(
            config.manifest_path,
            split="val",
            dataset_root=config.dataset_dir,
            input_mode=config.input_mode,
            image_size=config.image_size,
            resize_mode=config.resize_mode,
            normalize=config.normalize,
            augment=False,
        ),
        "test": LegacyTripletExperimentDataset(
            config.manifest_path,
            split="test",
            dataset_root=config.dataset_dir,
            input_mode=config.input_mode,
            image_size=config.image_size,
            resize_mode=config.resize_mode,
            normalize=config.normalize,
            augment=False,
        ),
    }


def _build_loaders(
    datasets: dict[str, LegacyTripletExperimentDataset],
    config: LegacyExperimentConfig,
) -> dict[str, DataLoader]:
    train_labels = datasets["train"].labels()
    sampler = None
    shuffle = False
    if config.use_weighted_sampler:
        counts = {
            0: max(train_labels.count(0), 1),
            1: max(train_labels.count(1), 1),
        }
        sample_weights = torch.tensor(
            [1.0 / counts[label] for label in train_labels],
            dtype=torch.double,
        )
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
    else:
        shuffle = True

    common_kwargs = {
        "batch_size": int(config.batch_size),
        "num_workers": int(config.num_workers),
        "pin_memory": torch.cuda.is_available(),
    }
    return {
        "train": DataLoader(datasets["train"], sampler=sampler, shuffle=shuffle, **common_kwargs),
        "val": DataLoader(datasets["val"], shuffle=False, **common_kwargs),
        "test": DataLoader(datasets["test"], shuffle=False, **common_kwargs),
    }


def _label_summary(labels: list[int]) -> str:
    negative = labels.count(0)
    positive = labels.count(1)
    total = len(labels)
    return f"total={total}, real={positive}, bogus={negative}"


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _torch_runtime_summary() -> str:
    cuda_build = getattr(torch.version, "cuda", None)
    return (
        f"python={sys.executable} | "
        f"torch={torch.__version__} | "
        f"torch_cuda_build={cuda_build or 'none'} | "
        f"cuda_available={torch.cuda.is_available()}"
    )


def _run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    *,
    epoch: int,
    total_epochs: int,
    log_every_n_steps: int,
) -> tuple[float, float]:
    model.train()
    loss_sum = 0.0
    seen = 0
    step_count = max(len(loader), 1)
    epoch_start = time.perf_counter()

    for step_index, (x, y) in enumerate(loader, start=1):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        loss_sum += float(loss.item()) * batch_size
        seen += batch_size
        if log_every_n_steps > 0 and (step_index == 1 or step_index % log_every_n_steps == 0 or step_index == step_count):
            running_loss = loss_sum / max(seen, 1)
            logger.info(
                "Epoch %02d/%02d | train step %03d/%03d | lr=%.3e | running_loss=%.5f",
                epoch,
                total_epochs,
                step_index,
                step_count,
                float(optimizer.param_groups[0]["lr"]),
                float(running_loss),
            )

    return loss_sum / max(seen, 1), time.perf_counter() - epoch_start


@torch.no_grad()
def _run_eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    loss_sum = 0.0
    seen = 0
    probs_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)
        probs = torch.softmax(logits, dim=1)[:, 1]

        batch_size = x.size(0)
        loss_sum += float(loss.item()) * batch_size
        seen += batch_size
        probs_list.append(probs.cpu().numpy())
        labels_list.append(y.cpu().numpy())

    if not probs_list:
        raise ValueError("Evaluation loader produced no batches")

    avg_loss = loss_sum / max(seen, 1)
    return avg_loss, np.concatenate(probs_list), np.concatenate(labels_list)


def _prepare_output_paths(config: LegacyExperimentConfig) -> dict[str, Path]:
    output_root = Path(config.output_root).resolve()
    manifest_dir = output_root / "manifests"
    results_dir = output_root / "results"
    checkpoint_dir = output_root / "checkpoints"
    plots_dir = output_root / "plots"
    prediction_dir = results_dir / "predictions"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    return {
        "output_root": output_root,
        "manifest_path": Path(config.manifest_path).resolve(),
        "checkpoint_path": checkpoint_dir / f"{config.experiment_name}_best.pt",
        "history_path": results_dir / f"{config.experiment_name}_history.csv",
        "summary_path": results_dir / f"{config.experiment_name}_summary.json",
        "config_snapshot_path": results_dir / f"{config.experiment_name}_config.json",
        "summary_csv_path": results_dir / config.summary_csv_name,
        "plots_dir": plots_dir,
        "split_distribution_plot": plots_dir / f"{config.experiment_name}_split_distribution.png",
        "learning_curve_plot": plots_dir / f"{config.experiment_name}_learning_curves.png",
        "val_analysis_plot": plots_dir / f"{config.experiment_name}_val_analysis.png",
        "test_analysis_plot": plots_dir / f"{config.experiment_name}_test_analysis.png",
        "prediction_dir": prediction_dir,
        "val_predictions_path": prediction_dir / f"{config.experiment_name}_val_predictions.csv",
        "test_predictions_path": prediction_dir / f"{config.experiment_name}_test_predictions.csv",
    }


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_rows_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fieldnames} for row in rows])


def _append_summary_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=SUMMARY_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})


def _save_split_distribution_plot(manifest_path: Path, output_path: Path) -> None:
    manifest = load_legacy_manifest(manifest_path)
    split_counts = manifest.get("summary", {}).get("split_counts", {})
    split_names = ["train", "val", "test"]
    real_counts = [int(split_counts.get(split_name, {}).get("real", 0)) for split_name in split_names]
    bogus_counts = [int(split_counts.get(split_name, {}).get("bogus", 0)) for split_name in split_names]

    x = np.arange(len(split_names), dtype=np.float64)
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.bar(x - width / 2.0, real_counts, width=width, label="Real", color="#2b8cbe")
    ax.bar(x + width / 2.0, bogus_counts, width=width, label="Bogus", color="#de2d26")
    ax.set_xticks(x)
    ax.set_xticklabels([split_name.upper() for split_name in split_names])
    ax.set_ylabel("Sample Count")
    ax.set_title("Legacy V1 Split Distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _save_learning_curve_plot(history_rows: list[dict[str, Any]], output_path: Path) -> None:
    if not history_rows:
        return

    epochs = [int(row["epoch"]) for row in history_rows]
    train_loss = [float(row["train_loss"]) for row in history_rows]
    val_loss = [float(row["val_loss"]) for row in history_rows]
    val_f1 = [float(row["val_f1"]) for row in history_rows]
    val_f2 = [float(row["val_f2"]) for row in history_rows]
    val_roc_auc = [float(row["val_roc_auc"]) for row in history_rows]
    val_pr_auc = [float(row["val_pr_auc"]) for row in history_rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=160)

    axes[0].plot(epochs, train_loss, label="Train Loss", color="#2b8cbe", linewidth=2)
    axes[0].plot(epochs, val_loss, label="Val Loss", color="#de2d26", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Learning Curves")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    axes[1].plot(epochs, val_f1, label="Val F1", color="#31a354", linewidth=2)
    axes[1].plot(epochs, val_f2, label="Val F2", color="#756bb1", linewidth=2)
    axes[1].plot(epochs, val_roc_auc, label="Val ROC-AUC", color="#636363", linewidth=1.8)
    axes[1].plot(epochs, val_pr_auc, label="Val PR-AUC", color="#fd8d3c", linewidth=1.8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title("Validation Metrics")
    axes[1].grid(alpha=0.2)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _save_analysis_figure(
    labels: np.ndarray,
    probs: np.ndarray,
    *,
    threshold: float,
    split_name: str,
    output_path: Path,
) -> None:
    metrics = _compute_binary_metrics(labels, probs, threshold=threshold)
    preds = (np.asarray(probs) >= float(threshold)).astype(np.int32)
    cm = confusion_matrix(np.asarray(labels, dtype=np.int32), preds, labels=[0, 1])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=160)

    ax_cm = axes[0, 0]
    im = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Bogus", "Real"])
    ax_cm.set_yticklabels(["Bogus", "Real"])
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    ax_cm.set_title(f"{split_name.upper()} Confusion Matrix")
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax_cm.text(col, row, int(cm[row, col]), ha="center", va="center", color="#111111")
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    ax_roc = axes[0, 1]
    if len(np.unique(labels)) > 1:
        fpr, tpr, _ = roc_curve(labels, probs)
        ax_roc.plot(fpr, tpr, color="#2b8cbe", linewidth=2, label=f"AUC={metrics['roc_auc']:.4f}")
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1)
    ax_roc.set_xlim(0.0, 1.0)
    ax_roc.set_ylim(0.0, 1.0)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"{split_name.upper()} ROC Curve")
    ax_roc.grid(alpha=0.2)
    if ax_roc.lines:
        ax_roc.legend(loc="lower right")

    ax_pr = axes[1, 0]
    if len(np.unique(labels)) > 1:
        precision, recall, _ = precision_recall_curve(labels, probs)
        ax_pr.plot(recall, precision, color="#31a354", linewidth=2, label=f"AP={metrics['pr_auc']:.4f}")
    ax_pr.set_xlim(0.0, 1.0)
    ax_pr.set_ylim(0.0, 1.0)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"{split_name.upper()} Precision-Recall")
    ax_pr.grid(alpha=0.2)
    if ax_pr.lines:
        ax_pr.legend(loc="lower left")

    ax_hist = axes[1, 1]
    labels_arr = np.asarray(labels, dtype=np.int32)
    probs_arr = np.asarray(probs, dtype=np.float64)
    bogus_probs = probs_arr[labels_arr == 0]
    real_probs = probs_arr[labels_arr == 1]
    bins = np.linspace(0.0, 1.0, 21)
    if bogus_probs.size:
        ax_hist.hist(bogus_probs, bins=bins, alpha=0.6, color="#de2d26", label="Bogus")
    if real_probs.size:
        ax_hist.hist(real_probs, bins=bins, alpha=0.6, color="#2b8cbe", label="Real")
    ax_hist.axvline(float(threshold), color="#111111", linestyle="--", linewidth=1.5, label=f"Threshold={threshold:.3f}")
    ax_hist.set_xlabel("P(Real)")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title(
        f"{split_name.upper()} Score Distribution\n"
        f"Acc={metrics['accuracy']:.3f}  Prec={metrics['precision']:.3f}  Rec={metrics['recall']:.3f}"
    )
    ax_hist.grid(alpha=0.2)
    ax_hist.legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _save_predictions_csv(
    output_path: Path,
    entries: list[dict[str, Any]],
    labels: np.ndarray,
    probs: np.ndarray,
    *,
    threshold: float,
    split_name: str,
) -> None:
    fieldnames = [
        "split",
        "relative_path",
        "group_key",
        "label",
        "label_name",
        "prob_real",
        "pred_label",
        "pred_label_name",
        "correct",
        "candidate_id",
        "is_manual",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for entry, label, prob in zip(entries, labels.tolist(), probs.tolist()):
            pred_label = 1 if float(prob) >= float(threshold) else 0
            writer.writerow(
                {
                    "split": split_name,
                    "relative_path": entry.get("relative_path", ""),
                    "group_key": entry.get("group_key", ""),
                    "label": int(label),
                    "label_name": "real" if int(label) == 1 else "bogus",
                    "prob_real": float(prob),
                    "pred_label": pred_label,
                    "pred_label_name": "real" if pred_label == 1 else "bogus",
                    "correct": int(pred_label == int(label)),
                    "candidate_id": entry.get("candidate_id", ""),
                    "is_manual": entry.get("is_manual", ""),
                }
            )


def _ensure_manifest(config: LegacyExperimentConfig) -> Path:
    manifest_path = Path(config.manifest_path).resolve()
    if manifest_path.is_file():
        return manifest_path

    logger.info("Manifest not found, building one at %s", manifest_path)
    build_legacy_triplet_manifest(
        config.dataset_dir,
        manifest_path,
        seed=config.seed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )
    return manifest_path


def train_legacy_classifier(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Run the legacy v1 classifier experiment end to end."""

    experiment_config = load_experiment_config(config)
    if _attention_compression_enabled(experiment_config):
        raise ValueError(
            "attention_compression_mode is configured for this experiment, but train_legacy_classifier currently "
            "supports it only for inference/benchmark paths. Train a baseline checkpoint first and then benchmark it."
        )
    _set_random_seed(experiment_config.seed)
    _ensure_manifest(experiment_config)
    paths = _prepare_output_paths(experiment_config)

    resolved_device = resolve_device(experiment_config.device)
    device = resolved_device.resolved
    logger.info("Starting experiment: %s", experiment_config.experiment_name)
    logger.info("Training device: %s", resolved_device.message)
    logger.info("Runtime: %s", _torch_runtime_summary())
    if device.type == "cpu" and getattr(torch.version, "cuda", None) is None:
        logger.warning("Current interpreter uses a CPU-only PyTorch build; GPU cannot be selected in this environment")
    logger.info("Dataset root: %s", experiment_config.dataset_dir)
    logger.info("Manifest: %s", paths["manifest_path"])
    logger.info(
        "Config: model=%s pretrained=%s input_mode=%s image_size=%s batch=%d epochs=%d optimizer=%s lr=%.3e scheduler=%s",
        _normalize_model_name(experiment_config.model_name),
        experiment_config.pretrained,
        experiment_config.input_mode,
        format_image_size_spec(experiment_config.image_size),
        int(experiment_config.batch_size),
        int(experiment_config.epochs),
        experiment_config.optimizer,
        float(experiment_config.lr),
        experiment_config.scheduler,
    )

    datasets = _build_datasets(experiment_config)
    loaders = _build_loaders(datasets, experiment_config)
    logger.info("Split train: %s", _label_summary(datasets["train"].labels()))
    logger.info("Split val: %s", _label_summary(datasets["val"].labels()))
    logger.info("Split test: %s", _label_summary(datasets["test"].labels()))

    model = create_experiment_model(
        experiment_config.model_name,
        pretrained=experiment_config.pretrained,
        image_size=experiment_config.image_size,
    ).to(device)
    parameter_count = int(_parameter_count(model))
    logger.info("Model parameters: %s", f"{parameter_count:,}")
    criterion = _criterion_from_config(experiment_config).to(device)
    optimizer = _optimizer_from_config(model, experiment_config)
    scheduler = _scheduler_from_config(optimizer, experiment_config)

    best_score = float("-inf")
    best_epoch = 0
    best_threshold = 0.5
    best_val_metrics: dict[str, Any] | None = None
    history_rows: list[dict[str, Any]] = []
    patience = 0
    epochs_ran = 0
    peak_gpu_memory_mb = 0.0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, int(experiment_config.epochs) + 1):
        eval_start = 0.0
        train_loss, train_seconds = _run_train_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            epoch=epoch,
            total_epochs=int(experiment_config.epochs),
            log_every_n_steps=max(0, int(experiment_config.log_every_n_steps)),
        )
        eval_start = time.perf_counter()
        val_loss, val_probs, val_labels = _run_eval_epoch(model, loaders["val"], criterion, device)
        eval_seconds = time.perf_counter() - eval_start
        threshold, val_metrics = _choose_threshold(
            val_labels,
            val_probs,
            metric_name=experiment_config.threshold_metric,
        )

        metric_key = str(experiment_config.selection_metric).strip().lower()
        score = float(val_metrics.get(metric_key, val_metrics["f1"]))
        current_lr = float(optimizer.param_groups[0]["lr"])
        history_rows.append(
            {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_seconds": float(train_seconds),
                "eval_seconds": float(eval_seconds),
                "epoch_seconds": float(train_seconds + eval_seconds),
                "val_threshold": float(threshold),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_precision": float(val_metrics["precision"]),
                "val_recall": float(val_metrics["recall"]),
                "val_f1": float(val_metrics["f1"]),
                "val_f2": float(val_metrics["f2"]),
                "val_roc_auc": float(val_metrics["roc_auc"]),
                "val_pr_auc": float(val_metrics["pr_auc"]),
            }
        )
        epochs_ran = epoch
        if device.type == "cuda":
            peak_gpu_memory_mb = max(
                peak_gpu_memory_mb,
                float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0),
            )
        logger.info(
            "Epoch %02d/%02d done | %.1fs train + %.1fs eval | train_loss=%.5f | val_loss=%.5f | val_acc=%.4f | val_f1=%.4f | val_f2=%.4f | val_roc_auc=%.4f | val_pr_auc=%.4f | threshold=%.4f",
            epoch,
            int(experiment_config.epochs),
            float(train_seconds),
            float(eval_seconds),
            float(train_loss),
            float(val_loss),
            float(val_metrics["accuracy"]),
            float(val_metrics["f1"]),
            float(val_metrics["f2"]),
            float(val_metrics["roc_auc"]),
            float(val_metrics["pr_auc"]),
            float(threshold),
        )

        if score > best_score + 1e-9:
            best_score = score
            best_epoch = epoch
            best_threshold = threshold
            best_val_metrics = dict(val_metrics)
            checkpoint = {
                "state_dict": model.state_dict(),
                "model_name": _normalize_model_name(experiment_config.model_name),
                "threshold": float(best_threshold),
                "best_epoch": int(best_epoch),
                "selection_metric": experiment_config.selection_metric,
                "config": asdict(experiment_config),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            torch.save(checkpoint, paths["checkpoint_path"])
            patience = 0
            logger.info(
                "New best checkpoint | epoch=%d | metric=%s | score=%.4f | saved=%s",
                epoch,
                metric_key,
                float(score),
                paths["checkpoint_path"],
            )
        else:
            patience += 1
            logger.info(
                "No improvement | epoch=%d | metric=%s | score=%.4f | patience=%d/%d",
                epoch,
                metric_key,
                float(score),
                int(patience),
                int(experiment_config.early_stopping_patience),
            )

        if scheduler is not None:
            scheduler.step()

        if patience >= int(experiment_config.early_stopping_patience):
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    if best_val_metrics is None:
        raise RuntimeError("Training finished without producing a valid checkpoint")

    val_result = evaluate_legacy_checkpoint(
        paths["checkpoint_path"],
        split="val",
        batch_size=experiment_config.batch_size,
        num_workers=experiment_config.num_workers,
        device=experiment_config.device,
        return_outputs=True,
    )
    test_result = evaluate_legacy_checkpoint(
        paths["checkpoint_path"],
        split="test",
        batch_size=experiment_config.batch_size,
        num_workers=experiment_config.num_workers,
        device=experiment_config.device,
        return_outputs=True,
    )
    _save_split_distribution_plot(paths["manifest_path"], paths["split_distribution_plot"])
    _save_learning_curve_plot(history_rows, paths["learning_curve_plot"])
    _save_analysis_figure(
        np.asarray(val_result["labels"]),
        np.asarray(val_result["probs"]),
        threshold=float(best_threshold),
        split_name="val",
        output_path=paths["val_analysis_plot"],
    )
    _save_analysis_figure(
        np.asarray(test_result["labels"]),
        np.asarray(test_result["probs"]),
        threshold=float(best_threshold),
        split_name="test",
        output_path=paths["test_analysis_plot"],
    )
    _save_predictions_csv(
        paths["val_predictions_path"],
        list(val_result["entries"]),
        np.asarray(val_result["labels"]),
        np.asarray(val_result["probs"]),
        threshold=float(best_threshold),
        split_name="val",
    )
    _save_predictions_csv(
        paths["test_predictions_path"],
        list(test_result["entries"]),
        np.asarray(test_result["labels"]),
        np.asarray(test_result["probs"]),
        threshold=float(best_threshold),
        split_name="test",
    )
    avg_epoch_seconds = float(
        np.mean([float(row["epoch_seconds"]) for row in history_rows]) if history_rows else 0.0
    )

    summary = {
        "experiment_name": experiment_config.experiment_name,
        "model_name": _normalize_model_name(experiment_config.model_name),
        "input_mode": experiment_config.input_mode,
        "image_size": normalize_image_size_spec(experiment_config.image_size),
        "resize_mode": experiment_config.resize_mode,
        "normalize": bool(experiment_config.normalize),
        "attention_compression_mode": experiment_config.attention_compression_mode,
        "attention_layer_selector": experiment_config.attention_layer_selector,
        "attention_layer_count": int(experiment_config.attention_layer_count),
        "enabled_layer_indices": list(experiment_config.enabled_layer_indices),
        "kv_bits": int(experiment_config.kv_bits),
        "group_size": int(experiment_config.group_size),
        "token_block_size": int(experiment_config.token_block_size),
        "preserve_cls_token": bool(experiment_config.preserve_cls_token),
        "quantize_k": bool(experiment_config.quantize_k),
        "quantize_v": bool(experiment_config.quantize_v),
        "streaming_enabled": bool(experiment_config.streaming_enabled),
        "materialize_attention_matrix": bool(experiment_config.materialize_attention_matrix),
        "pretrained": bool(experiment_config.pretrained),
        "seed": int(experiment_config.seed),
        "batch_size": int(experiment_config.batch_size),
        "lr": float(experiment_config.lr),
        "epochs_requested": int(experiment_config.epochs),
        "epochs_ran": int(epochs_ran),
        "best_epoch": int(best_epoch),
        "best_threshold": float(best_threshold),
        "val_accuracy": float(val_result["accuracy"]),
        "val_precision": float(val_result["precision"]),
        "val_recall": float(val_result["recall"]),
        "val_f1": float(val_result["f1"]),
        "val_f2": float(val_result["f2"]),
        "val_roc_auc": float(val_result["roc_auc"]),
        "val_pr_auc": float(val_result["pr_auc"]),
        "test_accuracy": float(test_result["accuracy"]),
        "test_precision": float(test_result["precision"]),
        "test_recall": float(test_result["recall"]),
        "test_f1": float(test_result["f1"]),
        "test_f2": float(test_result["f2"]),
        "test_roc_auc": float(test_result["roc_auc"]),
        "test_pr_auc": float(test_result["pr_auc"]),
        "test_tn": int(test_result["tn"]),
        "test_fp": int(test_result["fp"]),
        "test_fn": int(test_result["fn"]),
        "test_tp": int(test_result["tp"]),
        "params": parameter_count,
        "avg_epoch_seconds": avg_epoch_seconds,
        "peak_gpu_memory_mb": float(peak_gpu_memory_mb),
        "peak_gpu_memory_attention_only_mb": 0.0,
        "packed_kv_size_mb": 0.0,
        "token_count": 0,
        "num_patched_layers": 0,
        "manifest_path": str(paths["manifest_path"]),
        "checkpoint_path": str(paths["checkpoint_path"]),
        "history_path": str(paths["history_path"]),
        "plots_dir": str(paths["plots_dir"]),
        "split_distribution_plot": str(paths["split_distribution_plot"]),
        "learning_curve_plot": str(paths["learning_curve_plot"]),
        "val_analysis_plot": str(paths["val_analysis_plot"]),
        "test_analysis_plot": str(paths["test_analysis_plot"]),
        "val_predictions_path": str(paths["val_predictions_path"]),
        "test_predictions_path": str(paths["test_predictions_path"]),
    }

    logger.info(
        "Final summary | best_epoch=%d | val_f1=%.4f | val_f2=%.4f | test_f1=%.4f | test_f2=%.4f | test_recall=%.4f | avg_epoch_seconds=%.2f | peak_gpu_memory_mb=%.1f",
        int(best_epoch),
        float(val_result["f1"]),
        float(val_result["f2"]),
        float(test_result["f1"]),
        float(test_result["f2"]),
        float(test_result["recall"]),
        float(avg_epoch_seconds),
        float(peak_gpu_memory_mb),
    )
    logger.info("Plots directory: %s", paths["plots_dir"])
    logger.info("Summary json: %s", paths["summary_path"])

    paths["config_snapshot_path"].write_text(
        json.dumps(asdict(experiment_config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_history(paths["history_path"], history_rows)
    paths["summary_path"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _append_summary_csv(paths["summary_csv_path"], summary)
    return summary


def _input_fusion_experiment_name(base_experiment_name: str, *, base_input_mode: str, input_mode: str) -> str:
    normalized_base_mode = str(base_input_mode).strip().lower()
    normalized_input_mode = str(input_mode).strip().lower()
    if normalized_input_mode == normalized_base_mode:
        return str(base_experiment_name)
    return f"{base_experiment_name}_{normalized_input_mode}"


def _input_fusion_summary_path(output_root: Path, experiment_name: str) -> Path:
    return output_root / "results" / f"{experiment_name}_summary.json"


def _input_fusion_row(
    summary: dict[str, Any],
    *,
    variant: str,
    result_source: str,
    summary_path: Path,
) -> dict[str, Any]:
    return {
        "variant": variant,
        "input_mode": summary.get("input_mode", ""),
        "experiment_name": summary.get("experiment_name", ""),
        "result_source": result_source,
        "model_name": summary.get("model_name", ""),
        "pretrained": summary.get("pretrained", ""),
        "seed": summary.get("seed", ""),
        "image_size": summary.get("image_size", ""),
        "resize_mode": summary.get("resize_mode", ""),
        "batch_size": summary.get("batch_size", ""),
        "epochs_requested": summary.get("epochs_requested", ""),
        "epochs_ran": summary.get("epochs_ran", ""),
        "best_epoch": summary.get("best_epoch", ""),
        "best_threshold": summary.get("best_threshold", ""),
        "val_accuracy": summary.get("val_accuracy", ""),
        "val_recall": summary.get("val_recall", ""),
        "val_f1": summary.get("val_f1", ""),
        "val_roc_auc": summary.get("val_roc_auc", ""),
        "test_accuracy": summary.get("test_accuracy", ""),
        "test_recall": summary.get("test_recall", ""),
        "test_f1": summary.get("test_f1", ""),
        "test_roc_auc": summary.get("test_roc_auc", ""),
        "summary_path": str(summary_path),
        "checkpoint_path": summary.get("checkpoint_path", ""),
    }


def run_legacy_input_fusion_comparison(
    base_config: str | Path | dict[str, Any],
    *,
    comparison_csv_path: str | Path | None = None,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """Run Exp-4 by comparing multiple semantic input fusion modes."""

    experiment_config = load_experiment_config(base_config)
    output_root = Path(experiment_config.output_root).resolve()
    csv_path = (
        Path(comparison_csv_path).resolve()
        if comparison_csv_path is not None
        else output_root / "results" / "input_fusion_comparison.csv"
    )

    base_config_dict = asdict(experiment_config)
    comparison_rows: list[dict[str, Any]] = []

    for input_mode, variant in INPUT_FUSION_VARIANTS:
        experiment_name = _input_fusion_experiment_name(
            experiment_config.experiment_name,
            base_input_mode=experiment_config.input_mode,
            input_mode=input_mode,
        )
        summary_path = _input_fusion_summary_path(output_root, experiment_name)

        if skip_existing and summary_path.is_file():
            logger.info(
                "Reusing existing input fusion result | input_mode=%s | summary=%s",
                input_mode,
                summary_path,
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            result_source = "reused"
        else:
            logger.info(
                "Running input fusion experiment | input_mode=%s | experiment_name=%s",
                input_mode,
                experiment_name,
            )
            run_config = dict(base_config_dict)
            run_config["experiment_name"] = experiment_name
            run_config["input_mode"] = input_mode
            summary = train_legacy_classifier(run_config)
            result_source = "trained"
            summary_path = Path(summary.get("checkpoint_path", "")).resolve().parents[1] / "results" / f"{experiment_name}_summary.json"
            if not summary_path.is_file():
                summary_path = _input_fusion_summary_path(output_root, experiment_name)

        comparison_rows.append(
            _input_fusion_row(
                summary,
                variant=variant,
                result_source=result_source,
                summary_path=summary_path,
            )
        )

    _write_rows_csv(csv_path, INPUT_FUSION_COLUMNS, comparison_rows)
    best_row = max(
        comparison_rows,
        key=lambda row: (
            float(row.get("test_f1") or 0.0),
            float(row.get("test_recall") or 0.0),
            float(row.get("test_roc_auc") or 0.0),
        ),
    )
    logger.info(
        "Input fusion comparison complete | best_variant=%s | input_mode=%s | test_f1=%.4f | csv=%s",
        best_row["variant"],
        best_row["input_mode"],
        float(best_row.get("test_f1") or 0.0),
        csv_path,
    )
    return {
        "base_experiment_name": experiment_config.experiment_name,
        "model_name": _normalize_model_name(experiment_config.model_name),
        "pretrained": bool(experiment_config.pretrained),
        "comparison_csv_path": str(csv_path),
        "best_variant": best_row["variant"],
        "best_input_mode": best_row["input_mode"],
        "best_test_f1": float(best_row.get("test_f1") or 0.0),
        "results": comparison_rows,
    }


def _preprocessing_experiment_name(
    base_experiment_name: str,
    *,
    base_image_size: int | list[int] | str,
    base_resize_mode: str,
    base_normalize: bool,
    image_size: int | list[int] | str,
    resize_mode: str,
    normalize: bool,
    variant: str,
) -> str:
    if (
        image_size_specs_equal(image_size, base_image_size)
        and str(resize_mode).strip().lower() == str(base_resize_mode).strip().lower()
        and bool(normalize) == bool(base_normalize)
    ):
        return str(base_experiment_name)
    return f"{base_experiment_name}_{variant}"


def _preprocessing_summary_path(output_root: Path, experiment_name: str) -> Path:
    return output_root / "results" / f"{experiment_name}_summary.json"


def _preprocessing_row(
    summary: dict[str, Any],
    *,
    variant: str,
    description: str,
    result_source: str,
    summary_path: Path,
) -> dict[str, Any]:
    return {
        "variant": variant,
        "description": description,
        "experiment_name": summary.get("experiment_name", ""),
        "result_source": result_source,
        "model_name": summary.get("model_name", ""),
        "pretrained": summary.get("pretrained", ""),
        "input_mode": summary.get("input_mode", ""),
        "seed": summary.get("seed", ""),
        "image_size": summary.get("image_size", ""),
        "resize_mode": summary.get("resize_mode", ""),
        "normalize": summary.get("normalize", ""),
        "batch_size": summary.get("batch_size", ""),
        "epochs_requested": summary.get("epochs_requested", ""),
        "epochs_ran": summary.get("epochs_ran", ""),
        "best_epoch": summary.get("best_epoch", ""),
        "best_threshold": summary.get("best_threshold", ""),
        "val_accuracy": summary.get("val_accuracy", ""),
        "val_recall": summary.get("val_recall", ""),
        "val_f1": summary.get("val_f1", ""),
        "val_roc_auc": summary.get("val_roc_auc", ""),
        "test_accuracy": summary.get("test_accuracy", ""),
        "test_recall": summary.get("test_recall", ""),
        "test_f1": summary.get("test_f1", ""),
        "test_roc_auc": summary.get("test_roc_auc", ""),
        "avg_epoch_seconds": summary.get("avg_epoch_seconds", ""),
        "peak_gpu_memory_mb": summary.get("peak_gpu_memory_mb", ""),
        "summary_path": str(summary_path),
        "checkpoint_path": summary.get("checkpoint_path", ""),
    }


def run_legacy_preprocessing_comparison(
    base_config: str | Path | dict[str, Any],
    *,
    comparison_csv_path: str | Path | None = None,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """Run Exp-5 by comparing image-size and preprocessing strategies."""

    experiment_config = load_experiment_config(base_config)
    output_root = Path(experiment_config.output_root).resolve()
    csv_path = (
        Path(comparison_csv_path).resolve()
        if comparison_csv_path is not None
        else output_root / "results" / "preprocessing_comparison.csv"
    )

    base_config_dict = asdict(experiment_config)
    comparison_rows: list[dict[str, Any]] = []

    for variant_config in PREPROCESSING_VARIANTS:
        variant = str(variant_config["variant"])
        description = str(variant_config["description"])
        image_size = normalize_image_size_spec(variant_config["image_size"])
        resize_mode = str(variant_config["resize_mode"])
        normalize = bool(variant_config["normalize"])
        experiment_name = _preprocessing_experiment_name(
            experiment_config.experiment_name,
            base_image_size=experiment_config.image_size,
            base_resize_mode=experiment_config.resize_mode,
            base_normalize=experiment_config.normalize,
            image_size=image_size,
            resize_mode=resize_mode,
            normalize=normalize,
            variant=variant,
        )
        summary_path = _preprocessing_summary_path(output_root, experiment_name)

        if skip_existing and summary_path.is_file():
            logger.info(
                "Reusing existing preprocessing result | variant=%s | summary=%s",
                variant,
                summary_path,
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            result_source = "reused"
        else:
            logger.info(
                "Running preprocessing experiment | variant=%s | image_size=%s | resize_mode=%s | normalize=%s | experiment_name=%s",
                variant,
                image_size,
                resize_mode,
                normalize,
                experiment_name,
            )
            run_config = dict(base_config_dict)
            run_config["experiment_name"] = experiment_name
            run_config["image_size"] = image_size
            run_config["resize_mode"] = resize_mode
            run_config["normalize"] = normalize
            summary = train_legacy_classifier(run_config)
            result_source = "trained"
            summary_path = _preprocessing_summary_path(output_root, experiment_name)

        summary.setdefault("image_size", image_size)
        summary.setdefault("resize_mode", resize_mode)
        summary.setdefault("normalize", normalize)
        summary.setdefault("avg_epoch_seconds", 0.0)
        summary.setdefault("peak_gpu_memory_mb", 0.0)

        comparison_rows.append(
            _preprocessing_row(
                summary,
                variant=variant,
                description=description,
                result_source=result_source,
                summary_path=summary_path,
            )
        )

    _write_rows_csv(csv_path, PREPROCESSING_COLUMNS, comparison_rows)
    best_row = max(
        comparison_rows,
        key=lambda row: (
            float(row.get("test_f1") or 0.0),
            float(row.get("test_recall") or 0.0),
            float(row.get("test_roc_auc") or 0.0),
        ),
    )
    fastest_row = min(
        comparison_rows,
        key=lambda row: float(row.get("avg_epoch_seconds") or float("inf")),
    )
    lowest_memory_row = min(
        comparison_rows,
        key=lambda row: float(row.get("peak_gpu_memory_mb") or float("inf")),
    )
    logger.info(
        "Preprocessing comparison complete | best_variant=%s | test_f1=%.4f | fastest=%s | lowest_memory=%s | csv=%s",
        best_row["variant"],
        float(best_row.get("test_f1") or 0.0),
        fastest_row["variant"],
        lowest_memory_row["variant"],
        csv_path,
    )
    return {
        "base_experiment_name": experiment_config.experiment_name,
        "model_name": _normalize_model_name(experiment_config.model_name),
        "pretrained": bool(experiment_config.pretrained),
        "comparison_csv_path": str(csv_path),
        "best_variant": best_row["variant"],
        "best_test_f1": float(best_row.get("test_f1") or 0.0),
        "fastest_variant": fastest_row["variant"],
        "lowest_memory_variant": lowest_memory_row["variant"],
        "results": comparison_rows,
    }


def _model_scale_experiment_name(
    base_experiment_name: str,
    *,
    base_model_name: str,
    model_name: str,
    variant: str,
) -> str:
    normalized_base_model = _normalize_model_name(base_model_name)
    normalized_model = _normalize_model_name(model_name)
    if normalized_model == normalized_base_model:
        return str(base_experiment_name)
    return f"{base_experiment_name}_{variant}"


def _model_scale_summary_path(output_root: Path, experiment_name: str) -> Path:
    return output_root / "results" / f"{experiment_name}_summary.json"


def _model_scale_row(
    summary: dict[str, Any],
    *,
    variant: str,
    description: str,
    result_source: str,
    summary_path: Path,
) -> dict[str, Any]:
    return {
        "variant": variant,
        "description": description,
        "experiment_name": summary.get("experiment_name", ""),
        "result_source": result_source,
        "model_name": summary.get("model_name", ""),
        "pretrained": summary.get("pretrained", ""),
        "input_mode": summary.get("input_mode", ""),
        "seed": summary.get("seed", ""),
        "image_size": summary.get("image_size", ""),
        "resize_mode": summary.get("resize_mode", ""),
        "batch_size": summary.get("batch_size", ""),
        "epochs_requested": summary.get("epochs_requested", ""),
        "epochs_ran": summary.get("epochs_ran", ""),
        "best_epoch": summary.get("best_epoch", ""),
        "best_threshold": summary.get("best_threshold", ""),
        "val_accuracy": summary.get("val_accuracy", ""),
        "val_recall": summary.get("val_recall", ""),
        "val_f1": summary.get("val_f1", ""),
        "val_roc_auc": summary.get("val_roc_auc", ""),
        "test_accuracy": summary.get("test_accuracy", ""),
        "test_recall": summary.get("test_recall", ""),
        "test_f1": summary.get("test_f1", ""),
        "test_roc_auc": summary.get("test_roc_auc", ""),
        "params": summary.get("params", ""),
        "avg_epoch_seconds": summary.get("avg_epoch_seconds", ""),
        "peak_gpu_memory_mb": summary.get("peak_gpu_memory_mb", ""),
        "summary_path": str(summary_path),
        "checkpoint_path": summary.get("checkpoint_path", ""),
    }


def run_legacy_model_scale_comparison(
    base_config: str | Path | dict[str, Any],
    *,
    comparison_csv_path: str | Path | None = None,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """Run Exp-7 by comparing Swin Tiny/Small/Base model scales."""

    experiment_config = load_experiment_config(base_config)
    output_root = Path(experiment_config.output_root).resolve()
    csv_path = (
        Path(comparison_csv_path).resolve()
        if comparison_csv_path is not None
        else output_root / "results" / "model_scale_comparison.csv"
    )

    base_config_dict = asdict(experiment_config)
    comparison_rows: list[dict[str, Any]] = []

    for variant_config in MODEL_SCALE_VARIANTS:
        variant = str(variant_config["variant"])
        description = str(variant_config["description"])
        model_name = str(variant_config["model_name"])
        experiment_name = _model_scale_experiment_name(
            experiment_config.experiment_name,
            base_model_name=experiment_config.model_name,
            model_name=model_name,
            variant=variant,
        )
        summary_path = _model_scale_summary_path(output_root, experiment_name)

        if skip_existing and summary_path.is_file():
            logger.info(
                "Reusing existing model-scale result | variant=%s | model=%s | summary=%s",
                variant,
                model_name,
                summary_path,
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            result_source = "reused"
        else:
            logger.info(
                "Running model-scale experiment | variant=%s | model=%s | experiment_name=%s",
                variant,
                model_name,
                experiment_name,
            )
            run_config = dict(base_config_dict)
            run_config["experiment_name"] = experiment_name
            run_config["model_name"] = model_name
            summary = train_legacy_classifier(run_config)
            result_source = "trained"
            summary_path = _model_scale_summary_path(output_root, experiment_name)

        comparison_rows.append(
            _model_scale_row(
                summary,
                variant=variant,
                description=description,
                result_source=result_source,
                summary_path=summary_path,
            )
        )

    _write_rows_csv(csv_path, MODEL_SCALE_COLUMNS, comparison_rows)
    best_row = max(
        comparison_rows,
        key=lambda row: (
            float(row.get("test_f1") or 0.0),
            float(row.get("test_recall") or 0.0),
            float(row.get("test_roc_auc") or 0.0),
        ),
    )
    fastest_row = min(
        comparison_rows,
        key=lambda row: float(row.get("avg_epoch_seconds") or float("inf")),
    )
    lowest_memory_row = min(
        comparison_rows,
        key=lambda row: float(row.get("peak_gpu_memory_mb") or float("inf")),
    )
    logger.info(
        "Model-scale comparison complete | best_variant=%s | test_f1=%.4f | fastest=%s | lowest_memory=%s | csv=%s",
        best_row["variant"],
        float(best_row.get("test_f1") or 0.0),
        fastest_row["variant"],
        lowest_memory_row["variant"],
        csv_path,
    )
    return {
        "base_experiment_name": experiment_config.experiment_name,
        "base_model_name": _normalize_model_name(experiment_config.model_name),
        "pretrained": bool(experiment_config.pretrained),
        "comparison_csv_path": str(csv_path),
        "best_variant": best_row["variant"],
        "best_model_name": best_row["model_name"],
        "best_test_f1": float(best_row.get("test_f1") or 0.0),
        "fastest_variant": fastest_row["variant"],
        "lowest_memory_variant": lowest_memory_row["variant"],
        "results": comparison_rows,
    }


class Int4WeightOnlyLinear(nn.Module):
    """A lightweight pluggable INT4-style weight-only linear layer for smoke tests."""

    def __init__(self, qweight: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None) -> None:
        super().__init__()
        self.register_buffer("qweight", qweight.to(torch.int8))
        self.register_buffer("scale", scale.to(torch.float32))
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.to(torch.float32))

    @classmethod
    def from_float(cls, module: nn.Linear) -> "Int4WeightOnlyLinear":
        weight = module.weight.detach().to(torch.float32)
        scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 7.0
        qweight = torch.clamp(torch.round(weight / scale), min=-8, max=7).to(torch.int8)
        bias = None if module.bias is None else module.bias.detach().clone()
        return cls(qweight=qweight, scale=scale, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.qweight.to(dtype=x.dtype) * self.scale.to(device=x.device, dtype=x.dtype)
        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=x.dtype)
        return nn.functional.linear(x, weight, bias)

    @property
    def weight(self) -> torch.Tensor:
        return self.qweight.to(torch.float32) * self.scale


def _replace_modules(
    module: nn.Module,
    *,
    target_type: type[nn.Module],
    replace_fn: Any,
) -> nn.Module:
    for child_name, child_module in list(module.named_children()):
        if isinstance(child_module, target_type):
            setattr(module, child_name, replace_fn(child_module))
            continue
        _replace_modules(child_module, target_type=target_type, replace_fn=replace_fn)
    return module


def create_int4_weight_only_model(model: nn.Module) -> nn.Module:
    cloned = copy.deepcopy(model).cpu().eval()
    return _replace_modules(
        cloned,
        target_type=nn.Linear,
        replace_fn=Int4WeightOnlyLinear.from_float,
    )


def _load_legacy_checkpoint_bundle(
    checkpoint_path: str | Path,
    *,
    config_override: str | Path | dict[str, Any] | None = None,
) -> tuple[dict[str, Any], LegacyExperimentConfig, nn.Module]:
    checkpoint_file = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    if config_override is None:
        config = load_experiment_config(checkpoint.get("config") or {})
    elif isinstance(config_override, LegacyExperimentConfig):
        config = config_override
    else:
        config = load_experiment_config(config_override)
    model = create_experiment_model(
        checkpoint.get("model_name", config.model_name),
        pretrained=False,
        image_size=config.image_size,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model = _apply_attention_compression_from_config(model, config, for_training=False)
    model.eval()
    return checkpoint, config, model


def _build_eval_loader_for_split(
    config: LegacyExperimentConfig,
    *,
    split: str,
) -> tuple[LegacyTripletExperimentDataset, DataLoader]:
    dataset = LegacyTripletExperimentDataset(
        config.manifest_path,
        split=split,
        dataset_root=config.dataset_dir,
        input_mode=config.input_mode,
        image_size=config.image_size,
        resize_mode=config.resize_mode,
        normalize=config.normalize,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


@torch.no_grad()
def benchmark_legacy_model_inference(
    model: nn.Module,
    config: LegacyExperimentConfig,
    *,
    split: str = "test",
    device: str = "cpu",
    threshold: float = 0.5,
    input_dtype: torch.dtype | None = None,
    clone_model: bool = True,
) -> dict[str, Any]:
    try:
        import psutil
    except ImportError:
        psutil = None

    runtime_device = torch.device(device)
    dataset, loader = _build_eval_loader_for_split(config, split=split)
    if clone_model:
        model = copy.deepcopy(model)
    model = model.to(runtime_device).eval()

    process = psutil.Process() if psutil is not None else None
    peak_cpu_memory_mb = (
        float(process.memory_info().rss) / (1024.0 * 1024.0) if process is not None else 0.0
    )
    peak_gpu_memory_mb = 0.0
    peak_gpu_memory_attention_only_mb = 0.0
    packed_kv_size_mb = 0.0
    token_count = 0
    num_patched_layers = int(len(getattr(model, "_vit_attention_patched_indices", [])))
    probs_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    total_images = 0

    if runtime_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runtime_device)
        torch.cuda.synchronize(runtime_device)

    start_time = time.perf_counter()
    for x, y in loader:
        x = x.to(runtime_device)
        if input_dtype is not None:
            x = x.to(dtype=input_dtype)
        y = y.to(runtime_device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1]
        runtime_stats = _collect_attention_runtime_stats(model)
        peak_gpu_memory_attention_only_mb = max(
            peak_gpu_memory_attention_only_mb,
            float(runtime_stats.get("peak_gpu_memory_attention_only_mb") or 0.0),
        )
        packed_kv_size_mb = max(
            packed_kv_size_mb,
            float(runtime_stats.get("packed_kv_size_mb") or 0.0),
        )
        token_count = max(
            token_count,
            int(runtime_stats.get("token_count") or 0) or _estimate_vit_token_count_from_input(model, x),
        )
        num_patched_layers = max(
            num_patched_layers,
            int(runtime_stats.get("num_patched_layers") or 0),
        )
        probs_list.append(probs.detach().to(torch.float32).cpu().numpy())
        labels_list.append(y.detach().cpu().numpy())
        total_images += int(x.size(0))
        if process is not None:
            peak_cpu_memory_mb = max(
                peak_cpu_memory_mb,
                float(process.memory_info().rss) / (1024.0 * 1024.0),
            )
        if runtime_device.type == "cuda":
            torch.cuda.synchronize(runtime_device)
            peak_gpu_memory_mb = max(
                peak_gpu_memory_mb,
                float(torch.cuda.max_memory_allocated(runtime_device)) / (1024.0 * 1024.0),
            )
    total_seconds = time.perf_counter() - start_time

    if not probs_list:
        raise ValueError("Evaluation loader produced no batches")

    probs = np.concatenate(probs_list)
    labels = np.concatenate(labels_list)
    metrics = _compute_binary_metrics(labels, probs, threshold=threshold)
    metrics["split"] = str(split).strip().lower()
    metrics["device"] = runtime_device.type
    metrics["seconds_total"] = float(total_seconds)
    metrics["ms_per_image"] = float((total_seconds / max(total_images, 1)) * 1000.0)
    metrics["peak_cpu_memory_mb"] = float(peak_cpu_memory_mb)
    metrics["peak_gpu_memory_mb"] = float(peak_gpu_memory_mb)
    metrics["peak_gpu_memory_attention_only_mb"] = float(peak_gpu_memory_attention_only_mb)
    metrics["packed_kv_size_mb"] = float(packed_kv_size_mb)
    metrics["token_count"] = int(token_count)
    metrics["num_patched_layers"] = int(num_patched_layers)
    metrics["labels"] = labels
    metrics["probs"] = probs
    metrics["entries"] = [dataset.entry(index) for index in range(len(dataset))]
    return metrics


def benchmark_legacy_checkpoint(
    checkpoint_path: str | Path,
    *,
    config_override: str | Path | dict[str, Any] | None = None,
    split: str = "test",
    device: str = "cpu",
    threshold: float | None = None,
    input_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Load a checkpoint bundle and run the unified inference benchmark."""

    checkpoint, config, model = _load_legacy_checkpoint_bundle(
        checkpoint_path,
        config_override=config_override,
    )
    resolved_threshold = float(threshold) if threshold is not None else float(checkpoint.get("threshold", 0.5))
    metrics = benchmark_legacy_model_inference(
        model,
        config,
        split=split,
        device=device,
        threshold=resolved_threshold,
        input_dtype=input_dtype,
        clone_model=False,
    )
    metrics["checkpoint_path"] = str(Path(checkpoint_path).resolve())
    metrics["experiment_name"] = str(config.experiment_name)
    metrics["model_name"] = _normalize_model_name(checkpoint.get("model_name", config.model_name))
    metrics["attention_compression_mode"] = str(config.attention_compression_mode)
    metrics["threshold"] = float(resolved_threshold)
    metrics["streaming_enabled"] = bool(config.streaming_enabled)
    metrics["materialize_attention_matrix"] = bool(config.materialize_attention_matrix)
    return metrics


def _vit_attention_ablation_row(
    metrics: dict[str, Any],
    *,
    variant_config: dict[str, Any],
    config: LegacyExperimentConfig,
    status: str,
    supported: bool,
    result_source: str,
    notes: str = "",
) -> dict[str, Any]:
    merged_notes = [str(variant_config.get("notes", "")).strip(), str(notes).strip()]
    return {
        "variant": variant_config.get("variant", ""),
        "description": variant_config.get("description", ""),
        "result_source": result_source,
        "status": status,
        "supported": bool(supported),
        "layer_scope": variant_config.get("layer_scope", ""),
        "kv_target": variant_config.get("kv_target", ""),
        "cls_policy": variant_config.get("cls_policy", ""),
        "precision_mode": variant_config.get("precision_mode", ""),
        "streaming_enabled": bool(variant_config.get("streaming_enabled", False)),
        "materialize_attention_matrix": bool(variant_config.get("materialize_attention_matrix", False)),
        "experiment_name": str(config.experiment_name),
        "model_name": _normalize_model_name(config.model_name),
        "image_size": format_image_size_spec(config.image_size),
        "attention_compression_mode": str(config.attention_compression_mode),
        "attention_layer_selector": str(config.attention_layer_selector),
        "attention_layer_count": int(config.attention_layer_count),
        "enabled_layer_indices": json.dumps(list(config.enabled_layer_indices), ensure_ascii=False),
        "kv_bits": int(config.kv_bits),
        "group_size": int(config.group_size),
        "token_block_size": int(config.token_block_size),
        "preserve_cls_token": bool(config.preserve_cls_token),
        "quantize_k": bool(config.quantize_k),
        "quantize_v": bool(config.quantize_v),
        "split": metrics.get("split", ""),
        "threshold": metrics.get("threshold", ""),
        "accuracy": metrics.get("accuracy", ""),
        "precision": metrics.get("precision", ""),
        "recall": metrics.get("recall", ""),
        "f1": metrics.get("f1", ""),
        "roc_auc": metrics.get("roc_auc", ""),
        "ms_per_image": metrics.get("ms_per_image", ""),
        "peak_cpu_memory_mb": metrics.get("peak_cpu_memory_mb", ""),
        "peak_gpu_memory_mb": metrics.get("peak_gpu_memory_mb", ""),
        "peak_gpu_memory_attention_only_mb": metrics.get("peak_gpu_memory_attention_only_mb", ""),
        "packed_kv_size_mb": metrics.get("packed_kv_size_mb", ""),
        "token_count": metrics.get("token_count", ""),
        "num_patched_layers": metrics.get("num_patched_layers", ""),
        "checkpoint_path": metrics.get("checkpoint_path", ""),
        "notes": " | ".join(note for note in merged_notes if note),
    }


def run_vit_attention_ablation(
    checkpoint_path: str | Path,
    *,
    base_config: str | Path | dict[str, Any] | None = None,
    comparison_csv_path: str | Path | None = None,
    split: str = "test",
    device: str = "cpu",
    threshold: float | None = None,
    variants: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Benchmark a saved ViT checkpoint across the default full-module attention ablation matrix."""

    if base_config is None:
        checkpoint, experiment_config, _ = _load_legacy_checkpoint_bundle(checkpoint_path)
        resolved_threshold = float(threshold) if threshold is not None else float(checkpoint.get("threshold", 0.5))
    else:
        experiment_config = load_experiment_config(base_config)
        checkpoint = None
        resolved_threshold = float(threshold) if threshold is not None else 0.5

    normalized_model_name = _normalize_model_name(experiment_config.model_name)
    if not normalized_model_name.startswith("vit_"):
        raise ValueError("run_vit_attention_ablation currently only supports ViT checkpoints")

    output_root = Path(experiment_config.output_root).resolve()
    csv_path = (
        Path(comparison_csv_path).resolve()
        if comparison_csv_path is not None
        else output_root / "results" / "vit_attention_ablation.csv"
    )

    variant_configs = _build_vit_attention_ablation_variants(experiment_config, variants=variants)
    comparison_rows: list[dict[str, Any]] = []

    for variant in variant_configs:
        run_config = asdict(experiment_config)
        run_config["attention_compression_mode"] = variant["attention_compression_mode"]
        run_config["attention_layer_selector"] = variant["attention_layer_selector"]
        run_config["attention_layer_count"] = int(variant["attention_layer_count"])
        run_config["enabled_layer_indices"] = list(variant["enabled_layer_indices"])
        run_config["quantize_k"] = bool(variant["quantize_k"])
        run_config["quantize_v"] = bool(variant["quantize_v"])
        run_config["preserve_cls_token"] = bool(variant["preserve_cls_token"])
        run_config["kv_bits"] = int(variant["kv_bits"])
        run_config["streaming_enabled"] = bool(variant["streaming_enabled"])
        run_config["materialize_attention_matrix"] = bool(variant["materialize_attention_matrix"])
        resolved_config = load_experiment_config(run_config)

        try:
            metrics = benchmark_legacy_checkpoint(
                checkpoint_path,
                config_override=run_config,
                split=split,
                device=device,
                threshold=resolved_threshold,
            )
            metrics.setdefault(
                "threshold",
                float(checkpoint.get("threshold", resolved_threshold)) if checkpoint is not None else float(resolved_threshold),
            )
            comparison_rows.append(
                _vit_attention_ablation_row(
                    metrics,
                    variant_config=variant,
                    config=resolved_config,
                    status="completed",
                    supported=True,
                    result_source="benchmarked",
                )
            )
        except ValueError as exc:
            comparison_rows.append(
                _vit_attention_ablation_row(
                    {"checkpoint_path": str(Path(checkpoint_path).resolve())},
                    variant_config=variant,
                    config=resolved_config,
                    status="unsupported",
                    supported=False,
                    result_source="skipped",
                    notes=str(exc),
                )
            )

    _write_rows_csv(csv_path, VIT_ATTENTION_ABLATION_COLUMNS, comparison_rows)
    completed_rows = [row for row in comparison_rows if row["status"] == "completed"]
    if not completed_rows:
        raise RuntimeError("No ViT attention ablation variants completed successfully")
    best_row = max(
        completed_rows,
        key=lambda row: (
            float(row.get("f1") or 0.0),
            float(row.get("recall") or 0.0),
            float(row.get("roc_auc") or 0.0),
        ),
    )
    lowest_memory_row = min(
        completed_rows,
        key=lambda row: float(row.get("peak_gpu_memory_mb") or float("inf")),
    )
    logger.info(
        "ViT attention ablation complete | best_variant=%s | best_f1=%.4f | lowest_memory=%s | csv=%s",
        best_row["variant"],
        float(best_row.get("f1") or 0.0),
        lowest_memory_row["variant"],
        csv_path,
    )
    return {
        "base_experiment_name": experiment_config.experiment_name,
        "model_name": normalized_model_name,
        "comparison_csv_path": str(csv_path),
        "best_variant": best_row["variant"],
        "best_f1": float(best_row.get("f1") or 0.0),
        "lowest_memory_variant": lowest_memory_row["variant"],
        "results": comparison_rows,
    }


def _load_torchao_quantization_api() -> tuple[Any, Any]:
    try:
        import torchao.quantization as torchao_quantization
        import torchao.quantization.quant_api as torchao_quant_api

        return torchao_quantization, torchao_quant_api
    except Exception:
        spec = importlib.util.find_spec("torchao")
        if spec is None or not spec.submodule_search_locations:
            raise

        package_root = Path(next(iter(spec.submodule_search_locations))).resolve()
        root_init = package_root / "__init__.py"
        quant_init = package_root / "quantization" / "__init__.py"
        quant_api_path = package_root / "quantization" / "quant_api.py"
        if not quant_init.is_file() or not quant_api_path.is_file():
            raise ImportError(f"torchao quantization package not found under {package_root}")

        sys.modules.pop("torchao", None)
        sys.modules.pop("torchao.quantization", None)
        sys.modules.pop("torchao.quantization.quant_api", None)

        pkg = types.ModuleType("torchao")
        pkg.__path__ = [str(package_root)]
        pkg.__file__ = str(root_init)
        sys.modules["torchao"] = pkg

        quant_spec = importlib.util.spec_from_file_location(
            "torchao.quantization",
            quant_init,
            submodule_search_locations=[str(package_root / "quantization")],
        )
        if quant_spec is None or quant_spec.loader is None:
            raise ImportError(f"Unable to construct import spec for {quant_init}")
        quant_module = importlib.util.module_from_spec(quant_spec)
        sys.modules["torchao.quantization"] = quant_module
        pkg.quantization = quant_module
        quant_spec.loader.exec_module(quant_module)

        import torchao.quantization.quant_api as torchao_quant_api

        return quant_module, torchao_quant_api


def _estimate_model_state_size_mb(model: nn.Module) -> float:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return float(buffer.tell()) / (1024.0 * 1024.0)


def _quantization_summary_path(output_root: Path, experiment_name: str) -> Path:
    return output_root / "results" / f"{experiment_name}_quantization_smoke_summary.json"


@dataclass(frozen=True)
class TurboQuantVariantConfig:
    variant: str
    description: str
    mode: str
    bits: int
    base_bits: int | None = None
    qjl_dim: int = 0
    distribution: str = "beta"
    distribution_alpha: float = 2.0
    distribution_beta: float = 2.0
    sample_size: int = 200_000
    iterations: int = 48
    centroid_seed: int = 42
    rotation_seed: int = 1234
    projection_seed: int = 5678


TURBOQUANT_VARIANTS = [
    TurboQuantVariantConfig(
        variant="turboquant_mse_b3",
        description="TurboQuant MSE 3-bit",
        mode="mse",
        bits=3,
    ),
    TurboQuantVariantConfig(
        variant="turboquant_mse_b4",
        description="TurboQuant MSE 4-bit",
        mode="mse",
        bits=4,
    ),
    TurboQuantVariantConfig(
        variant="turboquant_prod_qjl_b4_m16",
        description="TurboQuant prod 4-bit + QJL residual (m=16)",
        mode="prod_qjl",
        bits=4,
        base_bits=3,
        qjl_dim=16,
    ),
    TurboQuantVariantConfig(
        variant="turboquant_prod_qjl_b4_m32",
        description="TurboQuant prod 4-bit + QJL residual (m=32)",
        mode="prod_qjl",
        bits=4,
        base_bits=3,
        qjl_dim=32,
    ),
]


def _turboquant_cache_dir(output_root: Path) -> Path:
    cache_dir = output_root / "results" / "turboquant_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _turboquant_centroid_cache_path(output_root: Path, config: TurboQuantVariantConfig, *, bits: int) -> Path:
    return (
        _turboquant_cache_dir(output_root)
        / (
            f"lloyd_max_{config.distribution}_bits{bits}_"
            f"a{config.distribution_alpha:g}_b{config.distribution_beta:g}_"
            f"seed{config.centroid_seed}.pt"
        )
    )


def _turboquant_rotation_cache_path(output_root: Path, config: TurboQuantVariantConfig, *, dim: int) -> Path:
    return _turboquant_cache_dir(output_root) / f"rotation_dim{dim}_seed{config.rotation_seed}.pt"


def _turboquant_projection_cache_path(
    output_root: Path,
    config: TurboQuantVariantConfig,
    *,
    dim: int,
    qjl_dim: int,
) -> Path:
    return (
        _turboquant_cache_dir(output_root)
        / f"qjl_projection_dim{dim}_m{qjl_dim}_seed{config.projection_seed}.pt"
    )


def _compute_lloyd_max_codebook(
    *,
    bits: int,
    distribution: str,
    alpha: float,
    beta_value: float,
    sample_size: int,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if bits <= 0:
        raise ValueError("bits must be positive")
    levels = 2**bits
    rng = np.random.default_rng(seed)
    if distribution != "beta":
        raise ValueError(f"Unsupported distribution for Lloyd-Max codebook: {distribution}")

    samples = rng.beta(alpha, beta_value, size=sample_size).astype(np.float64)
    samples = (samples * 2.0) - 1.0
    quantiles = np.linspace(0.0, 1.0, levels + 2, dtype=np.float64)[1:-1]
    centroids = np.quantile(samples, quantiles)

    for _ in range(max(1, int(iterations))):
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        assignments = np.searchsorted(boundaries, samples, side="left")
        updated = centroids.copy()
        for level in range(levels):
            mask = assignments == level
            if np.any(mask):
                updated[level] = float(np.mean(samples[mask]))
        updated = np.sort(updated)
        if np.allclose(updated, centroids, atol=1e-7, rtol=0.0):
            centroids = updated
            break
        centroids = updated

    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return centroids.astype(np.float32), boundaries.astype(np.float32)


def _load_or_create_lloyd_max_codebook(
    output_root: Path,
    config: TurboQuantVariantConfig,
    *,
    bits: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_path = _turboquant_centroid_cache_path(output_root, config, bits=bits)
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu")
        centroids = payload["centroids"]
        boundaries = payload["boundaries"]
    else:
        centroids_np, boundaries_np = _compute_lloyd_max_codebook(
            bits=bits,
            distribution=config.distribution,
            alpha=float(config.distribution_alpha),
            beta_value=float(config.distribution_beta),
            sample_size=int(config.sample_size),
            iterations=int(config.iterations),
            seed=int(config.centroid_seed),
        )
        centroids = torch.from_numpy(centroids_np)
        boundaries = torch.from_numpy(boundaries_np)
        torch.save(
            {
                "centroids": centroids,
                "boundaries": boundaries,
                "bits": bits,
                "distribution": config.distribution,
            },
            cache_path,
        )

    return centroids.to(device=device, dtype=torch.float32), boundaries.to(device=device, dtype=torch.float32)


def _load_or_create_random_rotation(
    output_root: Path,
    config: TurboQuantVariantConfig,
    *,
    dim: int,
    device: torch.device,
) -> torch.Tensor:
    cache_path = _turboquant_rotation_cache_path(output_root, config, dim=dim)
    if cache_path.is_file():
        rotation = torch.load(cache_path, map_location="cpu")
    else:
        generator = torch.Generator(device="cpu").manual_seed(int(config.rotation_seed) + int(dim))
        gaussian = torch.randn((dim, dim), generator=generator, dtype=torch.float32)
        rotation, r = torch.linalg.qr(gaussian, mode="reduced")
        signs = torch.sign(torch.diag(r))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        rotation = rotation * signs.unsqueeze(0)
        torch.save(rotation, cache_path)
    return rotation.to(device=device, dtype=torch.float32)


def _load_or_create_qjl_projection(
    output_root: Path,
    config: TurboQuantVariantConfig,
    *,
    dim: int,
    qjl_dim: int,
    device: torch.device,
) -> torch.Tensor:
    cache_path = _turboquant_projection_cache_path(output_root, config, dim=dim, qjl_dim=qjl_dim)
    if cache_path.is_file():
        projection = torch.load(cache_path, map_location="cpu")
    else:
        generator = torch.Generator(device="cpu").manual_seed(
            int(config.projection_seed) + int(dim) * 17 + int(qjl_dim)
        )
        gaussian = torch.randn((qjl_dim, dim), generator=generator, dtype=torch.float32)
        projection = F.normalize(gaussian, dim=-1)
        torch.save(projection, cache_path)
    return projection.to(device=device, dtype=torch.float32)


class TurboQuantRuntime:
    """TurboQuant-style scalar codebook compression with optional QJL residual correction."""

    def __init__(self, output_root: Path, config: TurboQuantVariantConfig) -> None:
        self.output_root = Path(output_root).resolve()
        self.config = config
        self._centroid_cache: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        self._rotation_cache: dict[tuple[int, str], torch.Tensor] = {}
        self._projection_cache: dict[tuple[int, int, str], torch.Tensor] = {}

    def _codebook(self, *, bits: int, dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = (bits, str(device))
        if key not in self._centroid_cache:
            self._centroid_cache[key] = _load_or_create_lloyd_max_codebook(
                self.output_root,
                self.config,
                bits=bits,
                device=device,
            )
        return self._centroid_cache[key]

    def _rotation(self, *, dim: int, device: torch.device) -> torch.Tensor:
        key = (dim, str(device))
        if key not in self._rotation_cache:
            self._rotation_cache[key] = _load_or_create_random_rotation(
                self.output_root,
                self.config,
                dim=dim,
                device=device,
            )
        return self._rotation_cache[key]

    def _projection(self, *, dim: int, qjl_dim: int, device: torch.device) -> torch.Tensor:
        key = (dim, qjl_dim, str(device))
        if key not in self._projection_cache:
            self._projection_cache[key] = _load_or_create_qjl_projection(
                self.output_root,
                self.config,
                dim=dim,
                qjl_dim=qjl_dim,
                device=device,
            )
        return self._projection_cache[key]

    def _turboquant_mse(self, x: torch.Tensor, *, bits: int) -> torch.Tensor:
        device = x.device
        dim = int(x.shape[-1])
        x_fp32 = x.to(torch.float32)
        centroids, boundaries = self._codebook(bits=bits, dim=dim, device=device)
        rotation = self._rotation(dim=dim, device=device)
        rotated = torch.matmul(x_fp32, rotation.T)
        indices = torch.bucketize(rotated.contiguous(), boundaries)
        quantized = centroids[indices]
        reconstructed = torch.matmul(quantized, rotation)
        return reconstructed

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        if self.config.mode == "mse":
            reconstructed = self._turboquant_mse(x, bits=int(self.config.bits))
            return reconstructed.to(dtype=original_dtype)

        if self.config.mode == "prod_qjl":
            base_bits = int(self.config.base_bits or max(int(self.config.bits) - 1, 1))
            qjl_dim = int(self.config.qjl_dim)
            x_fp32 = x.to(torch.float32)
            mse_reconstructed = self._turboquant_mse(x, bits=base_bits)
            residual = x_fp32 - mse_reconstructed
            residual_norm = torch.linalg.vector_norm(residual, dim=-1, keepdim=True).clamp_min(1e-6)
            projection = self._projection(dim=int(x.shape[-1]), qjl_dim=qjl_dim, device=x.device)
            signed = torch.sign(torch.matmul(residual, projection.T))
            signed = torch.where(signed == 0, torch.ones_like(signed), signed)
            direction = torch.matmul(signed, projection)
            direction_norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(1e-6)
            corrected = mse_reconstructed + (residual_norm * (direction / direction_norm))
            return corrected.to(dtype=original_dtype)

        raise ValueError(f"Unsupported TurboQuant mode: {self.config.mode}")


def _shifted_window_attention_turboquant(
    input_tensor: torch.Tensor,
    module: swin_transformer.ShiftedWindowAttention,
    runtime: TurboQuantRuntime,
) -> torch.Tensor:
    B, H, W, C = input_tensor.shape
    window_size = list(module.window_size)
    shift_size = list(module.shift_size)

    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    x = F.pad(input_tensor, (0, 0, 0, pad_r, 0, pad_b))
    _, pad_H, pad_W, _ = x.shape

    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0

    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
    x = x.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)

    qkv_bias = module.qkv.bias
    logit_scale = getattr(module, "logit_scale", None)
    if logit_scale is not None and qkv_bias is not None:
        qkv_bias = qkv_bias.clone()
        length = qkv_bias.numel() // 3
        qkv_bias[length : 2 * length].zero_()

    qkv = F.linear(x, module.qkv.weight, qkv_bias)
    qkv = qkv.reshape(x.size(0), x.size(1), 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    k = runtime.apply(k)
    v = runtime.apply(v)

    if logit_scale is not None:
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        attn = attn * torch.clamp(logit_scale, max=math.log(100.0)).exp()
    else:
        q = q * (C // module.num_heads) ** -0.5
        attn = q.matmul(k.transpose(-2, -1))

    attn = attn + module.get_relative_position_bias()

    if sum(shift_size) > 0:
        attn_mask = x.new_zeros((pad_H, pad_W))
        h_slices = ((0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None))
        w_slices = ((0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None))
        count = 0
        for h in h_slices:
            for w in w_slices:
                attn_mask[h[0] : h[1], w[0] : w[1]] = count
                count += 1
        attn_mask = attn_mask.view(pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1])
        attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, window_size[0] * window_size[1])
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        attn = attn.view(x.size(0) // num_windows, num_windows, module.num_heads, x.size(1), x.size(1))
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, module.num_heads, x.size(1), x.size(1))

    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=float(module.attention_dropout), training=module.training)

    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
    x = F.linear(x, module.proj.weight, module.proj.bias)
    x = F.dropout(x, p=float(module.dropout), training=module.training)

    x = x.view(B, pad_H // window_size[0], pad_W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

    return x[:, :H, :W, :].contiguous()


def create_turboquant_attention_model(
    base_model: nn.Module,
    *,
    output_root: Path,
    variant_config: TurboQuantVariantConfig,
) -> nn.Module:
    turbo_model = copy.deepcopy(base_model).cuda().eval()
    runtime = TurboQuantRuntime(output_root=output_root, config=variant_config)
    patched_count = 0

    for module in turbo_model.modules():
        if isinstance(module, swin_transformer.ShiftedWindowAttention):
            def _forward(self, x, _runtime=runtime):
                return _shifted_window_attention_turboquant(x, self, _runtime)

            module.forward = types.MethodType(_forward, module)
            patched_count += 1

    if patched_count == 0:
        raise RuntimeError("TurboQuant attention experiment currently expects a Swin model with ShiftedWindowAttention modules")

    setattr(turbo_model, "quant_variant", variant_config.variant)
    return turbo_model


def _run_turboquant_attention_variant(
    *,
    variant_config: TurboQuantVariantConfig,
    output_root: Path,
    checkpoint_file: Path,
    model_name: str,
    experiment_name: str,
    config: LegacyExperimentConfig,
    base_model: nn.Module,
    threshold: float,
    baseline_metrics: dict[str, Any],
) -> dict[str, Any]:
    turbo_model = create_turboquant_attention_model(
        base_model,
        output_root=output_root,
        variant_config=variant_config,
    )
    turbo_gpu = benchmark_legacy_model_inference(
        turbo_model,
        config,
        split="test",
        device="cuda",
        threshold=threshold,
        clone_model=False,
    )
    return _quantization_row(
        variant=variant_config.variant,
        description=variant_config.description,
        result_source="measured",
        status="ok",
        runnable=True,
        model_name=model_name,
        experiment_name=experiment_name,
        split="test",
        threshold=threshold,
        checkpoint_path=checkpoint_file,
        notes=(
            f"TurboQuant-style {variant_config.mode} attention compression with bits={variant_config.bits}, "
            f"base_bits={variant_config.base_bits}, qjl_dim={variant_config.qjl_dim}; "
            "compresses attention K/V activations at runtime, so checkpoint size remains unchanged."
        ),
        metrics_gpu=turbo_gpu,
        model_state_size_mb=_estimate_model_state_size_mb(turbo_model),
        baseline_metrics=baseline_metrics,
    )


def _quantization_row(
    *,
    variant: str,
    description: str,
    result_source: str,
    status: str,
    runnable: bool,
    model_name: str,
    experiment_name: str,
    split: str,
    threshold: float,
    checkpoint_path: Path,
    notes: str,
    metrics_cpu: dict[str, Any] | None = None,
    metrics_gpu: dict[str, Any] | None = None,
    model_state_size_mb: float | None = None,
    baseline_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cpu_metrics = metrics_cpu or {}
    gpu_metrics = metrics_gpu or {}
    primary_metrics = cpu_metrics or gpu_metrics
    baseline = baseline_metrics or {}
    test_accuracy = float(primary_metrics.get("accuracy") or 0.0) if runnable else ""
    test_f1 = float(primary_metrics.get("f1") or 0.0) if runnable else ""
    return {
        "variant": variant,
        "description": description,
        "result_source": result_source,
        "status": status,
        "runnable": runnable,
        "model_name": model_name,
        "experiment_name": experiment_name,
        "split": split,
        "threshold": float(threshold),
        "test_accuracy": test_accuracy,
        "test_precision": float(primary_metrics.get("precision") or 0.0) if runnable else "",
        "test_recall": float(primary_metrics.get("recall") or 0.0) if runnable else "",
        "test_f1": test_f1,
        "test_roc_auc": float(primary_metrics.get("roc_auc") or 0.0) if runnable else "",
        "delta_accuracy_vs_fp32": (
            float(test_accuracy) - float(baseline.get("accuracy") or 0.0)
            if runnable and baseline_metrics is not None
            else ""
        ),
        "delta_f1_vs_fp32": (
            float(test_f1) - float(baseline.get("f1") or 0.0)
            if runnable and baseline_metrics is not None
            else ""
        ),
        "cpu_ms_per_image": float(cpu_metrics.get("ms_per_image") or 0.0) if runnable else "",
        "gpu_ms_per_image": float(gpu_metrics.get("ms_per_image") or 0.0) if gpu_metrics else "",
        "cpu_peak_memory_mb": float(cpu_metrics.get("peak_cpu_memory_mb") or 0.0) if runnable else "",
        "gpu_peak_memory_mb": float(gpu_metrics.get("peak_gpu_memory_mb") or 0.0) if gpu_metrics else "",
        "model_state_size_mb": float(model_state_size_mb or 0.0) if model_state_size_mb is not None else "",
        "checkpoint_path": str(checkpoint_path),
        "notes": notes,
    }


def _run_torchao_gpu_quantization_variant(
    *,
    variant: str,
    description: str,
    checkpoint_file: Path,
    model_name: str,
    experiment_name: str,
    config: LegacyExperimentConfig,
    base_model: nn.Module,
    threshold: float,
    torchao_quantization: Any,
    quant_config: Any,
    baseline_metrics: dict[str, Any],
) -> dict[str, Any]:
    torchao_model = copy.deepcopy(base_model).cuda().to(torch.bfloat16).eval()
    torchao_quantization.quantize_(torchao_model, quant_config, device="cuda")
    torchao_gpu = benchmark_legacy_model_inference(
        torchao_model,
        config,
        split="test",
        device="cuda",
        threshold=threshold,
        input_dtype=torch.bfloat16,
        clone_model=False,
    )
    return _quantization_row(
        variant=variant,
        description=description,
        result_source="measured",
        status="ok",
        runnable=True,
        model_name=model_name,
        experiment_name=experiment_name,
        split="test",
        threshold=threshold,
        checkpoint_path=checkpoint_file,
        notes=f"Measured with torchao config {quant_config}.",
        metrics_gpu=torchao_gpu,
        model_state_size_mb=_estimate_model_state_size_mb(torchao_model),
        baseline_metrics=baseline_metrics,
    )


def run_legacy_quantization_smoke(
    base_config: str | Path | dict[str, Any],
    *,
    checkpoint_path: str | Path | None = None,
    comparison_csv_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run Exp-8 quantization smoke validation against an existing checkpoint."""

    experiment_config = load_experiment_config(base_config)
    output_root = Path(experiment_config.output_root).resolve()
    checkpoint_file = (
        Path(checkpoint_path).resolve()
        if checkpoint_path is not None
        else output_root / "checkpoints" / f"{experiment_config.experiment_name}_best.pt"
    )
    csv_path = (
        Path(comparison_csv_path).resolve()
        if comparison_csv_path is not None
        else output_root / "results" / "quantization_smoke_results.csv"
    )
    summary_path = _quantization_summary_path(output_root, experiment_config.experiment_name)

    checkpoint, config, base_model = _load_legacy_checkpoint_bundle(
        checkpoint_file,
        config_override=experiment_config,
    )
    threshold = float(checkpoint.get("threshold", 0.5))
    model_name = str(checkpoint.get("model_name", config.model_name))

    rows: list[dict[str, Any]] = []
    baseline_cpu = benchmark_legacy_model_inference(
        base_model,
        config,
        split="test",
        device="cpu",
        threshold=threshold,
    )
    baseline_gpu = None
    if torch.cuda.is_available():
        baseline_gpu = benchmark_legacy_model_inference(
            base_model,
            config,
            split="test",
            device="cuda",
            threshold=threshold,
        )
    baseline_size_mb = _estimate_model_state_size_mb(base_model)
    rows.append(
        _quantization_row(
            variant="fp32_baseline",
            description="FP32 baseline",
            result_source="measured",
            status="ok",
            runnable=True,
            model_name=model_name,
            experiment_name=config.experiment_name,
            split="test",
            threshold=threshold,
            checkpoint_path=checkpoint_file,
            notes="Original checkpoint reloaded without quantization.",
            metrics_cpu=baseline_cpu,
            metrics_gpu=baseline_gpu,
            model_state_size_mb=baseline_size_mb,
            baseline_metrics=baseline_cpu,
        )
    )

    try:
        quantized_int8 = torch.ao.quantization.quantize_dynamic(
            copy.deepcopy(base_model).cpu().eval(),
            {nn.Linear},
            dtype=torch.qint8,
        )
        int8_cpu = benchmark_legacy_model_inference(
            quantized_int8,
            config,
            split="test",
            device="cpu",
            threshold=threshold,
        )
        rows.append(
            _quantization_row(
                variant="dynamic_int8",
                description="Torch AO dynamic INT8 for Linear layers",
                result_source="measured",
                status="ok",
                runnable=True,
                model_name=model_name,
                experiment_name=config.experiment_name,
                split="test",
                threshold=threshold,
                checkpoint_path=checkpoint_file,
                notes="Official torch.ao dynamic quantization on CPU.",
                metrics_cpu=int8_cpu,
                model_state_size_mb=_estimate_model_state_size_mb(quantized_int8),
                baseline_metrics=baseline_cpu,
            )
        )
    except Exception as exc:
        rows.append(
            _quantization_row(
                variant="dynamic_int8",
                description="Torch AO dynamic INT8 for Linear layers",
                result_source="checked",
                status="unsupported_for_model",
                runnable=False,
                model_name=model_name,
                experiment_name=config.experiment_name,
                split="test",
                threshold=threshold,
                checkpoint_path=checkpoint_file,
                notes=f"Official dynamic INT8 conversion completed but inference failed on this model implementation: {type(exc).__name__}: {exc}",
                baseline_metrics=baseline_cpu,
            )
        )

    try:
        torchao_quantization, torchao_quant_api = _load_torchao_quantization_api()
        if not torch.cuda.is_available():
            raise RuntimeError("TorchAO GPU quantization paths currently require CUDA in this experiment")

        torchao_variants = [
            (
                "standard_int4_g32",
                "TorchAO standard INT4 group_size=32",
                torchao_quantization.Int4WeightOnlyConfig(
                    group_size=32,
                    int4_packing_format=torchao_quant_api.Int4PackingFormat.TILE_PACKED_TO_4D,
                ),
            ),
            (
                "standard_int4_g64",
                "TorchAO standard INT4 group_size=64",
                torchao_quantization.Int4WeightOnlyConfig(
                    group_size=64,
                    int4_packing_format=torchao_quant_api.Int4PackingFormat.TILE_PACKED_TO_4D,
                ),
            ),
            (
                "standard_int4_g128",
                "TorchAO standard INT4 group_size=128",
                torchao_quantization.Int4WeightOnlyConfig(
                    group_size=128,
                    int4_packing_format=torchao_quant_api.Int4PackingFormat.TILE_PACKED_TO_4D,
                ),
            ),
            (
                "int8_weight_only",
                "TorchAO INT8 weight-only",
                torchao_quantization.Int8WeightOnlyConfig(),
            ),
            (
                "int8_dynact_int8_weight",
                "TorchAO INT8 dynamic activation + INT8 weight",
                torchao_quantization.Int8DynamicActivationInt8WeightConfig(),
            ),
        ]

        for variant_name, description, quant_config in torchao_variants:
            try:
                rows.append(
                    _run_torchao_gpu_quantization_variant(
                        variant=variant_name,
                        description=description,
                        checkpoint_file=checkpoint_file,
                        model_name=model_name,
                        experiment_name=config.experiment_name,
                        config=config,
                        base_model=base_model,
                        threshold=threshold,
                        torchao_quantization=torchao_quantization,
                        quant_config=quant_config,
                        baseline_metrics=baseline_gpu or baseline_cpu,
                    )
                )
            except Exception as exc:
                rows.append(
                    _quantization_row(
                        variant=variant_name,
                        description=description,
                        result_source="checked",
                        status="unsupported_in_environment",
                        runnable=False,
                        model_name=model_name,
                        experiment_name=config.experiment_name,
                        split="test",
                        threshold=threshold,
                        checkpoint_path=checkpoint_file,
                        notes=f"TorchAO path could not be executed in the current environment: {type(exc).__name__}: {exc}",
                        baseline_metrics=baseline_gpu or baseline_cpu,
                    )
                )
    except Exception as exc:
        for variant_name, description in [
            ("standard_int4_g32", "TorchAO standard INT4 group_size=32"),
            ("standard_int4_g64", "TorchAO standard INT4 group_size=64"),
            ("standard_int4_g128", "TorchAO standard INT4 group_size=128"),
            ("int8_weight_only", "TorchAO INT8 weight-only"),
            ("int8_dynact_int8_weight", "TorchAO INT8 dynamic activation + INT8 weight"),
        ]:
            rows.append(
                _quantization_row(
                    variant=variant_name,
                    description=description,
                    result_source="checked",
                    status="unsupported_in_environment",
                    runnable=False,
                    model_name=model_name,
                    experiment_name=config.experiment_name,
                    split="test",
                    threshold=threshold,
                    checkpoint_path=checkpoint_file,
                    notes=f"TorchAO quantization API could not be loaded in the current environment: {type(exc).__name__}: {exc}",
                    baseline_metrics=baseline_gpu or baseline_cpu,
                )
            )

    for variant_config in TURBOQUANT_VARIANTS:
        try:
            rows.append(
                _run_turboquant_attention_variant(
                    variant_config=variant_config,
                    output_root=output_root,
                    checkpoint_file=checkpoint_file,
                    model_name=model_name,
                    experiment_name=config.experiment_name,
                    config=config,
                    base_model=base_model,
                    threshold=threshold,
                    baseline_metrics=baseline_gpu or baseline_cpu,
                )
            )
        except Exception as exc:
            rows.append(
                _quantization_row(
                    variant=variant_config.variant,
                    description=variant_config.description,
                    result_source="checked",
                    status="unsupported_in_environment",
                    runnable=False,
                    model_name=model_name,
                    experiment_name=config.experiment_name,
                    split="test",
                    threshold=threshold,
                    checkpoint_path=checkpoint_file,
                    notes=(
                        "TurboQuant-style attention compression path could not be executed in the current "
                        f"environment: {type(exc).__name__}: {exc}"
                    ),
                    baseline_metrics=baseline_gpu or baseline_cpu,
                )
            )

    custom_int4_model = create_int4_weight_only_model(base_model)
    custom_int4_cpu = benchmark_legacy_model_inference(
        custom_int4_model,
        config,
        split="test",
        device="cpu",
        threshold=threshold,
    )
    custom_int4_gpu = None
    if torch.cuda.is_available():
        custom_int4_gpu = benchmark_legacy_model_inference(
            custom_int4_model,
            config,
            split="test",
            device="cuda",
            threshold=threshold,
        )
    rows.append(
        _quantization_row(
            variant="custom_weight_only_int4",
            description="Pluggable custom INT4-style weight-only linear layer",
            result_source="measured",
            status="ok",
            runnable=True,
            model_name=model_name,
            experiment_name=config.experiment_name,
            split="test",
            threshold=threshold,
            checkpoint_path=checkpoint_file,
            notes="Custom adapter replaces nn.Linear recursively and dequantizes on the fly for smoke validation.",
            metrics_cpu=custom_int4_cpu,
            metrics_gpu=custom_int4_gpu,
            model_state_size_mb=_estimate_model_state_size_mb(custom_int4_model),
            baseline_metrics=baseline_cpu,
        )
    )

    _write_rows_csv(csv_path, QUANTIZATION_COLUMNS, rows)
    summary = {
        "experiment_name": config.experiment_name,
        "model_name": model_name,
        "checkpoint_path": str(checkpoint_file),
        "comparison_csv_path": str(csv_path),
        "baseline_variant": "fp32_baseline",
        "best_quantized_variant": max(
            [row for row in rows if row["runnable"] and row["variant"] != "fp32_baseline"],
            key=lambda row: (float(row.get("test_f1") or 0.0), float(row.get("test_accuracy") or 0.0)),
        )["variant"],
        "results": rows,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def evaluate_legacy_checkpoint(
    checkpoint_path: str | Path,
    *,
    split: str = "test",
    manifest_path: str | Path | None = None,
    dataset_dir: str | Path | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    device: str | None = None,
    return_outputs: bool = False,
) -> dict[str, Any]:
    """Load a saved checkpoint and evaluate one split."""

    checkpoint_file = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    config = load_experiment_config(checkpoint.get("config") or {})

    if manifest_path is not None:
        config.manifest_path = str(Path(manifest_path).resolve())
    if dataset_dir is not None:
        config.dataset_dir = str(Path(dataset_dir).resolve())
    if batch_size is not None:
        config.batch_size = int(batch_size)
    if num_workers is not None:
        config.num_workers = int(num_workers)
    if device is not None:
        config.device = str(device)

    resolved_device = resolve_device(config.device)
    runtime_device = resolved_device.resolved
    logger.info("Evaluation device: %s", resolved_device.message)

    dataset = LegacyTripletExperimentDataset(
        config.manifest_path,
        split=split,
        dataset_root=config.dataset_dir,
        input_mode=config.input_mode,
        image_size=config.image_size,
        resize_mode=config.resize_mode,
        normalize=config.normalize,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    model = create_experiment_model(
        checkpoint.get("model_name", config.model_name),
        pretrained=False,
        image_size=config.image_size,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model = _apply_attention_compression_from_config(model, config, for_training=False)
    model = model.to(runtime_device)

    criterion = _criterion_from_config(config).to(runtime_device)
    loss, probs, labels = _run_eval_epoch(model, loader, criterion, runtime_device)
    threshold = float(checkpoint.get("threshold", 0.5))
    metrics = _compute_binary_metrics(labels, probs, threshold=threshold)
    metrics["loss"] = float(loss)
    metrics["split"] = str(split).strip().lower()
    metrics["checkpoint_path"] = str(checkpoint_file)
    if return_outputs:
        metrics["probs"] = probs
        metrics["labels"] = labels
        metrics["entries"] = [dataset.entry(index) for index in range(len(dataset))]
    return metrics
