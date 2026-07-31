from __future__ import annotations

import numpy as np
import torch

from scann.ai.hierarchical_classifier import (
    ACTION_CLASSES,
    DETAIL_CLASSES,
    FAMILY_CLASSES,
    FrozenFeatureHierarchicalClassifier,
    build_hierarchical_checkpoint,
    calibration_metrics,
    fit_temperature_scaling,
    hierarchical_loss,
    hierarchical_predictions,
    taxonomy_target_indices,
)
from scann.ai.inference import InferenceConfig, InferenceEngine


def test_taxonomy_targets_and_masked_loss():
    model = FrozenFeatureHierarchicalClassifier(
        feature_encoder="scann_test_identity",
        device=torch.device("cpu"),
        hidden_dim=8,
        dropout=0.0,
    )
    images = torch.rand(3, 3, 32, 32)
    outputs = model(images)
    supernova = taxonomy_target_indices("supernova")
    noise = taxonomy_target_indices("noise")
    targets = {
        "review_action": torch.tensor(
            [supernova["review_action"], noise["review_action"], -1]
        ),
        "phenomenon_family": torch.tensor(
            [
                supernova["phenomenon_family"],
                noise["phenomenon_family"],
                -1,
            ]
        ),
        "detail_type": torch.tensor(
            [supernova["detail_type"], noise["detail_type"], -1]
        ),
    }

    losses = hierarchical_loss(outputs, targets)
    losses["loss"].backward()

    assert torch.isfinite(losses["loss"])
    assert outputs["review_action_logits"].shape == (3, len(ACTION_CLASSES))
    assert outputs["phenomenon_family_logits"].shape == (
        3,
        len(FAMILY_CLASSES),
    )
    assert outputs["detail_type_logits"].shape == (3, len(DETAIL_CLASSES))


def test_hierarchical_predictions_preserve_compatibility_fields():
    outputs = {
        "review_action_logits": torch.tensor([[4.0, 0.0]]),
        "phenomenon_family_logits": torch.zeros(
            1,
            len(FAMILY_CLASSES),
        ),
        "detail_type_logits": torch.zeros(1, len(DETAIL_CLASSES)),
    }
    predictions = hierarchical_predictions(outputs)

    assert predictions[0]["review_action"] == "keep"
    assert predictions[0]["label"] == "real"
    assert predictions[0]["score"] > 0.9
    assert predictions[0]["detail_type"] in DETAIL_CLASSES


def test_hierarchical_checkpoint_loads_in_existing_inference_engine(tmp_path):
    model = FrozenFeatureHierarchicalClassifier(
        feature_encoder="scann_test_identity",
        device=torch.device("cpu"),
        hidden_dim=8,
        dropout=0.0,
    )
    checkpoint = build_hierarchical_checkpoint(
        model,
        partition_id="partition-test",
        partition_manifest_sha256="a" * 64,
        feature_version="candidate-structured-v1",
    )
    checkpoint["head_config"] = {
        "hidden_dim": 8,
        "dropout": 0.0,
    }
    path = tmp_path / "hierarchical.pth"
    torch.save(checkpoint, path)

    engine = InferenceEngine(
        str(path),
        config=InferenceConfig(
            device="cpu",
            use_amp=False,
            batch_size=2,
        ),
    )
    predictions = engine.classify_patches_detailed(
        [np.zeros((3, 32, 32), dtype=np.float32)]
    )

    assert engine.is_ready
    assert predictions[0]["review_action"] in ACTION_CLASSES
    assert predictions[0]["phenomenon_family"] in FAMILY_CLASSES
    assert predictions[0]["detail_type"] in DETAIL_CLASSES
    assert predictions[0]["label"] in {"real", "bogus"}


def test_calibration_metrics_reports_brier_and_ece():
    metrics = calibration_metrics(
        torch.tensor([[0.9, 0.1], [0.4, 0.6]]),
        torch.tensor([0, 1]),
        bin_count=5,
    )

    assert metrics["support"] == 2
    assert 0.0 <= metrics["brier_score"] <= 2.0
    assert 0.0 <= metrics["ece"] <= 1.0


def test_temperature_scaling_returns_positive_validation_parameter():
    temperature = fit_temperature_scaling(
        torch.tensor(
            [[4.0, 0.0], [0.0, 4.0], [2.0, 1.0]],
            dtype=torch.float32,
        ),
        torch.tensor([0, 1, 1], dtype=torch.long),
        maximum_steps=10,
    )

    assert 0.0 < temperature <= 100.0
