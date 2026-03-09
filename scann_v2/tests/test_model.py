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

    def test_model_instantiates_with_vit_backbone(self):
        torch = pytest.importorskip("torch")
        from scann.ai.model import SCANNClassifier

        model = SCANNClassifier(pretrained=False, backbone_name="ViT_B_16")
        assert model is not None

    def test_forward_shape_with_vit_backbone(self):
        torch = pytest.importorskip("torch")
        from scann.ai.model import SCANNClassifier

        model = SCANNClassifier(pretrained=False, backbone_name="ViT_B_16")
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 2)

    def test_detect_backbone_from_state_dict_vit(self):
        torch = pytest.importorskip("torch")
        from scann.ai.model import SCANNClassifier

        state_dict = {
            "backbone.conv_proj.weight": torch.zeros(1),
            "backbone.encoder.layers.encoder_layer_0.ln_1.weight": torch.zeros(1),
        }
        assert SCANNClassifier._detect_backbone_from_state_dict(state_dict) == "ViT_B_16"


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
