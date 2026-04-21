from __future__ import annotations

import torch

from scann.ai.feature_classifier import (
    FeatureHeadClassifier,
    apply_prior_logit_correction,
    feature_encoder_spec,
    preprocess_feature_batch,
)


def test_feature_head_outputs_class_logits() -> None:
    head = FeatureHeadClassifier(feature_dim=6, num_classes=11)
    logits = head(torch.randn(4, 6))

    assert logits.shape == (4, 11)


def test_prior_logit_correction_subtracts_log_frequency() -> None:
    logits = torch.zeros((1, 3), dtype=torch.float32)
    corrected = apply_prior_logit_correction(logits, [-0.1, -2.0, 0.0], tau=1.0)

    assert corrected[0, 1] > corrected[0, 0]
    assert corrected[0, 2] == 0.0


def test_feature_encoder_spec_supports_test_identity() -> None:
    spec = feature_encoder_spec("scann_test_identity")

    assert spec.feature_dim == 6
    assert spec.family == "test"


def test_preprocess_feature_batch_resizes_and_normalizes() -> None:
    batch = torch.full((2, 3, 16, 16), 0.5)
    prepared = preprocess_feature_batch(batch, input_size=32)

    assert prepared.shape == (2, 3, 32, 32)
    assert torch.isfinite(prepared).all()
