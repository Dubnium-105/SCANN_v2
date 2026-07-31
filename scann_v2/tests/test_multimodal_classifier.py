from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from scann.ai.inference import InferenceConfig, InferenceEngine
from scann.ai.hierarchical_classifier import (
    ACTION_CLASSES,
    DETAIL_CLASSES,
    FAMILY_CLASSES,
)
from scann.ai.multimodal_classifier import (
    FeatureNormalization,
    SharedEncoderLateFusionClassifier,
    build_multimodal_checkpoint,
    build_structured_feature_matrix,
    synchronized_triplet_transform,
)
from scann.ai.feature_classifier import load_feature_encoder


class _IdentityEncoder(nn.Module):
    def forward(self, images):
        means = images.mean(dim=(-2, -1))
        standard_deviations = images.std(
            dim=(-2, -1),
            unbiased=False,
        )
        return torch.cat([means, standard_deviations], dim=1)


def test_feature_normalization_uses_mask_for_missing_values():
    normalization = FeatureNormalization.fit(
        np.asarray([[1.0, 10.0], [3.0, 99.0]]),
        np.asarray([[True, True], [True, False]]),
        feature_names=("snr", "fwhm"),
    )
    transformed = normalization.transform(
        torch.tensor([[3.0, 500.0]]),
        torch.tensor([[True, False]]),
    )

    assert normalization.mean == (2.0, 10.0)
    assert transformed[0, 1].item() == 0.0


def test_multimodal_shared_encoder_and_heads():
    normalization = FeatureNormalization(
        mean=(0.0, 0.0),
        std=(1.0, 1.0),
        feature_names=("snr", "fwhm"),
    )
    model = SharedEncoderLateFusionClassifier(
        encoder=_IdentityEncoder(),
        encoder_family="test",
        image_feature_dim=6,
        input_size=16,
        structured_feature_dim=2,
        feature_normalization=normalization,
        structured_hidden_dim=4,
        fusion_hidden_dim=12,
        dropout=0.0,
    )
    outputs = model(
        torch.rand(2, 3, 3, 16, 16),
        torch.tensor([[5.0, 2.0], [3.0, 0.0]]),
        torch.tensor([[True, True], [True, False]]),
    )

    assert outputs["review_action_logits"].shape == (
        2,
        len(ACTION_CLASSES),
    )
    assert outputs["phenomenon_family_logits"].shape == (
        2,
        len(FAMILY_CLASSES),
    )
    assert outputs["detail_type_logits"].shape == (
        2,
        len(DETAIL_CLASSES),
    )


def test_structured_feature_matrix_and_checkpoint_round_trip(tmp_path):
    values, mask = build_structured_feature_matrix(
        [
            {
                "candidate_features": {
                    "snr": 5.0,
                    "fwhm": None,
                }
            }
        ],
        feature_names=("snr", "fwhm"),
    )
    encoder, spec = load_feature_encoder(
        "scann_test_identity",
        device=torch.device("cpu"),
    )
    model = SharedEncoderLateFusionClassifier(
        encoder=encoder,
        encoder_family=spec.family,
        image_feature_dim=spec.feature_dim,
        input_size=spec.input_size,
        structured_feature_dim=2,
        feature_normalization=FeatureNormalization.fit(
            values,
            mask,
            feature_names=("snr", "fwhm"),
        ),
        structured_hidden_dim=4,
        fusion_hidden_dim=12,
        dropout=0.0,
    )
    checkpoint = build_multimodal_checkpoint(
        model,
        feature_encoder="scann_test_identity",
        partition_id="partition-1",
        partition_manifest_sha256="a" * 64,
        taxonomy_version="taxonomy-v1",
    )
    loaded = SharedEncoderLateFusionClassifier.from_checkpoint(
        checkpoint,
        device=torch.device("cpu"),
    )
    checkpoint_path = tmp_path / "multimodal.pth"
    torch.save(checkpoint, checkpoint_path)
    engine = InferenceEngine(
        str(checkpoint_path),
        config=InferenceConfig(
            device="cpu",
            use_amp=False,
            batch_size=1,
        ),
    )
    predictions = engine.classify_patches_detailed(
        [np.zeros((3, 16, 16), dtype=np.float32)],
        structured_features=[{"snr": 4.0}],
    )

    assert values.tolist() == [[5.0, 0.0]]
    assert mask.tolist() == [[True, False]]
    assert checkpoint["gold_test_used_for_selection"] is False
    assert loaded.feature_normalization.feature_names == ("snr", "fwhm")
    assert predictions[0]["review_action"] in ACTION_CLASSES


def test_triplet_geometry_is_synchronized():
    views = np.stack(
        [
            np.arange(16).reshape(4, 4),
            np.arange(16).reshape(4, 4) + 100,
            np.arange(16).reshape(4, 4) + 200,
        ]
    )
    transformed = synchronized_triplet_transform(
        views,
        rotation_quadrants=1,
        horizontal_flip=True,
    )

    assert np.array_equal(
        transformed[1] - transformed[0],
        np.full((4, 4), 100),
    )
    assert np.array_equal(
        transformed[2] - transformed[1],
        np.full((4, 4), 100),
    )
