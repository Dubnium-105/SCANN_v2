import numpy as np
import torch

from scann.ai.inference import InferenceEngine


class _DenseModelStub:
    def __init__(self):
        self.last_input = None
        self.called = 0

    def forward_dense(self, x: torch.Tensor) -> torch.Tensor:
        self.called += 1
        self.last_input = x.detach().cpu()
        return torch.zeros((x.shape[0], 5, 2, 2), dtype=torch.float32, device=x.device)


class _InvalidDenseModelStub:
    def forward_dense(self, x: torch.Tensor) -> torch.Tensor:
        _ = x
        return torch.zeros((1, 4, 2, 2), dtype=torch.float32)


def test_detect_dense_full_image_returns_empty_when_model_missing():
    engine = InferenceEngine(model_path="")

    new_image = np.random.rand(32, 32).astype(np.float32)
    detections = engine.detect_dense_full_image(new_image)

    assert detections == []


def test_detect_dense_full_image_returns_empty_on_empty_input():
    engine = InferenceEngine(model_path="")
    model = _DenseModelStub()
    engine.model = model

    detections = engine.detect_dense_full_image(np.array([], dtype=np.float32))

    assert detections == []
    assert model.called == 0


def test_detect_dense_full_image_builds_diff_new_old_input_and_returns_list():
    engine = InferenceEngine(model_path="")
    model = _DenseModelStub()
    engine.model = model

    new_image = np.array([[10, 20], [30, 40]], dtype=np.float32)
    old_image = np.array([[1, 2], [3, 4]], dtype=np.float32)

    detections = engine.detect_dense_full_image(new_image, old_image)

    assert isinstance(detections, list)
    assert detections == []
    assert model.called == 1
    assert model.last_input is not None
    assert tuple(model.last_input.shape) == (1, 3, 2, 2)

    diff_channel = model.last_input[0, 0].numpy()
    new_channel = model.last_input[0, 1].numpy()
    old_channel = model.last_input[0, 2].numpy()

    np.testing.assert_allclose(diff_channel, np.abs(new_channel - old_channel), atol=1e-6)


def test_detect_dense_full_image_returns_empty_on_invalid_dense_output_shape():
    engine = InferenceEngine(model_path="")
    engine.model = _InvalidDenseModelStub()

    new_image = np.random.rand(16, 16).astype(np.float32)
    detections = engine.detect_dense_full_image(new_image)

    assert detections == []
