"""AI 模型定义单元测试"""

import pytest


class TestSCANNClassifier:
    """测试分类器模型架构 (向后兼容 v1)"""

    def test_model_instantiates(self):
        torch = pytest.importorskip("torch")
        from scann.ai.model import SCANNClassifier

        model = SCANNClassifier(pretrained=False)
        assert model is not None

    def test_forward_shape(self):
        torch = pytest.importorskip("torch")
        from scann.ai.model import SCANNClassifier

        model = SCANNClassifier(pretrained=False)
        model.eval()
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 2)

    def test_load_checkpoint_nonexistent_raises(self):
        torch = pytest.importorskip("torch")
        from scann.ai.model import SCANNClassifier

        with pytest.raises((FileNotFoundError, OSError, RuntimeError)):
            SCANNClassifier.load_from_checkpoint("nonexistent.pth")


class TestSCANNDetector:
    """测试检测器模型架构 (MobileNetV3, ≤8GB VRAM)"""

    def test_detector_instantiates(self):
        torch = pytest.importorskip("torch")
        from scann.ai.model import SCANNDetector

        model = SCANNDetector(pretrained=False)
        assert model is not None

    def test_detector_forward(self):
        torch = pytest.importorskip("torch")
        from scann.ai.model import SCANNDetector

        model = SCANNDetector(in_channels=1, pretrained=False)
        model.eval()
        x = torch.randn(1, 1, 512, 512)
        with torch.no_grad():
            out = model(x)
        assert out is not None
