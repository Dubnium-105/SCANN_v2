from __future__ import annotations

import numpy as np
import torch

from scann.ai.feature_classifier import FeatureHeadClassifier
from scann.ai.inference import InferenceEngine


def test_inference_loads_frozen_feature_checkpoint(tmp_path) -> None:
    head = FeatureHeadClassifier(feature_dim=6, num_classes=11)
    checkpoint = {
        "model_format": "frozen_feature_classifier",
        "training_mode": "frozen_feature_classifier",
        "feature_encoder": "scann_test_identity",
        "feature_dim": 6,
        "head_state": head.state_dict(),
        "class_names": [
            "asteroid",
            "supernova",
            "variable_star",
            "satellite_trail",
            "noise",
            "diffraction_spike",
            "cmos_condensation",
            "corresponding",
            "disappeared_asteroid",
            "disappeared_star",
            "disappeared_galaxy",
        ],
        "class_log_prior": [-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0],
        "prior_correction_tau": 1.0,
        "prior_logit_correction": {"enabled": True, "tau": 1.0},
    }
    checkpoint_path = tmp_path / "frozen.pth"
    torch.save(checkpoint, checkpoint_path)

    engine = InferenceEngine(str(checkpoint_path))
    result = engine.classify_patches_detailed([np.full((3, 8, 8), 0.5, dtype=np.float32)])

    assert engine.model_format == "frozen_feature_classifier"
    assert result
    assert "predicted_class" in result[0]
    assert "score" in result[0]
