"""Experiment helpers for legacy and research workflows."""

from .legacy_dataset import LegacyTripletExperimentDataset
from .legacy_manifest import build_legacy_triplet_manifest, load_legacy_manifest
from .legacy_runner import (
    LegacyExperimentConfig,
    create_vit_attention_compression_model,
    create_vit_packed_kv_attention_model,
    evaluate_legacy_checkpoint,
    load_experiment_config,
    run_legacy_input_fusion_comparison,
    run_legacy_model_scale_comparison,
    run_legacy_preprocessing_comparison,
    run_legacy_quantization_smoke,
    train_legacy_classifier,
)

__all__ = [
    "LegacyExperimentConfig",
    "LegacyTripletExperimentDataset",
    "build_legacy_triplet_manifest",
    "create_vit_attention_compression_model",
    "create_vit_packed_kv_attention_model",
    "evaluate_legacy_checkpoint",
    "load_experiment_config",
    "load_legacy_manifest",
    "run_legacy_input_fusion_comparison",
    "run_legacy_model_scale_comparison",
    "run_legacy_preprocessing_comparison",
    "run_legacy_quantization_smoke",
    "train_legacy_classifier",
]
