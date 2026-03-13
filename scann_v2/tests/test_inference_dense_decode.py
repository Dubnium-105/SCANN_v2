import numpy as np
import torch

from scann.ai.inference import InferenceEngine


class _StaticDenseModel:
    def __init__(self, dense_output: torch.Tensor):
        self._dense_output = dense_output

    def forward_dense(self, x: torch.Tensor) -> torch.Tensor:
        _ = x
        return self._dense_output.to(x.device)


def _build_engine_with_dense_output(dense_output: torch.Tensor) -> InferenceEngine:
    engine = InferenceEngine(model_path="")
    engine.model = _StaticDenseModel(dense_output)
    return engine


def test_detect_dense_full_image_decodes_dense_predictions_above_threshold():
    dense_output = torch.zeros((1, 5, 4, 4), dtype=torch.float32)
    dense_output[0, 4, 1, 2] = 3.0

    engine = _build_engine_with_dense_output(dense_output)
    image = np.random.rand(64, 64).astype(np.float32)

    detections = engine.detect_dense_full_image(image, score_threshold=0.9, top_k=10)

    assert len(detections) == 1
    det = detections[0]
    assert 0 <= det.x < 64
    assert 0 <= det.y < 64
    assert 1 <= det.width <= 64
    assert 1 <= det.height <= 64
    assert det.confidence > 0.9


def test_detect_dense_full_image_honors_top_k():
    dense_output = torch.zeros((1, 5, 4, 4), dtype=torch.float32)
    dense_output[0, 4, 0, 0] = 2.0
    dense_output[0, 4, 3, 3] = 4.0

    engine = _build_engine_with_dense_output(dense_output)
    image = np.random.rand(64, 64).astype(np.float32)

    detections = engine.detect_dense_full_image(image, score_threshold=0.5, top_k=1)

    assert len(detections) == 1
    assert detections[0].confidence > 0.95


def test_detect_dense_full_image_applies_nms_on_overlapping_boxes():
    dense_output = torch.zeros((1, 5, 4, 4), dtype=torch.float32)
    dense_output[0, 4, 0, 0] = 6.0
    dense_output[0, 4, 0, 1] = 5.5

    dense_output[0, 2, 0, 0] = 2.0
    dense_output[0, 3, 0, 0] = 2.0
    dense_output[0, 2, 0, 1] = 2.0
    dense_output[0, 3, 0, 1] = 2.0

    engine = _build_engine_with_dense_output(dense_output)
    image = np.random.rand(64, 64).astype(np.float32)

    detections = engine.detect_dense_full_image(
        image,
        score_threshold=0.5,
        top_k=10,
        iou_threshold=0.3,
    )

    assert len(detections) == 1
    assert detections[0].confidence > 0.99
