"""ModelService 单元测试。"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scann.core.models import AppConfig
from scann.services.model_service import ModelService


def test_load_model_uses_app_config_and_stores_engine():
    app_config = AppConfig(
        batch_size=128,
        compute_device="cpu",
        model_format="auto",
        model_backbone="ViT_B_16",
        ai_confidence=0.23,
    )
    mock_engine = Mock()
    mock_engine.threshold = 0.5
    mock_engine.model_format = "v2_classifier"
    mock_engine.model_backbone = "ViT_B_16"
    mock_engine.channel_order = (2, 1, 0)
    engine_factory = Mock(return_value=mock_engine)

    service = ModelService(engine_factory=engine_factory)
    result = service.load_model("/tmp/model.pth", app_config)

    engine_factory.assert_called_once()
    call_kwargs = engine_factory.call_args.kwargs
    assert call_kwargs["model_path"] == "/tmp/model.pth"
    assert call_kwargs["config"].batch_size == 128
    assert call_kwargs["config"].device == "cpu"
    assert call_kwargs["config"].model_backbone == "ViT_B_16"
    assert service.inference_engine is mock_engine
    assert result.effective_threshold == pytest.approx(0.23)
    assert app_config.ai_confidence == pytest.approx(0.23)
    assert app_config.model_path == "/tmp/model.pth"


def test_load_model_clamps_threshold_via_engine_property():
    app_config = AppConfig(ai_confidence=1.7)

    class FakeEngine:
        def __init__(self):
            self._threshold = 0.42
            self.config = SimpleNamespace(batch_size=64)
            self.model_format = "auto"
            self.model_backbone = "ResNet18"
            self.channel_order = (0, 1, 2)

        @property
        def threshold(self):
            return self._threshold

        @threshold.setter
        def threshold(self, value):
            self._threshold = max(0.0, min(1.0, float(value)))

    service = ModelService(engine_factory=Mock(return_value=FakeEngine()))
    result = service.load_model("/tmp/model.pth", app_config)

    assert result.effective_threshold == pytest.approx(1.0)
    assert app_config.ai_confidence == pytest.approx(1.0)


def test_apply_runtime_config_updates_ready_engine():
    app_config = AppConfig(ai_confidence=0.17, batch_size=256)
    engine = Mock()
    engine.is_ready = True
    engine.threshold = 0.5
    engine.config = SimpleNamespace(batch_size=64)

    service = ModelService()
    service.set_inference_engine(engine)

    applied = service.apply_runtime_config(app_config)

    assert applied is True
    assert engine.threshold == pytest.approx(0.17)
    assert engine.config.batch_size == 256


def test_apply_runtime_config_skips_unready_engine():
    app_config = AppConfig(ai_confidence=0.17, batch_size=256)
    engine = Mock()
    engine.is_ready = False
    engine.config = SimpleNamespace(batch_size=64)

    service = ModelService()
    service.set_inference_engine(engine)

    applied = service.apply_runtime_config(app_config)

    assert applied is False
    assert engine.config.batch_size == 64


def test_get_model_info_returns_summary_for_ready_engine():
    engine = Mock()
    engine.is_ready = True
    engine.threshold = 0.61
    engine.model_format = "v2_classifier"
    engine.model_backbone = "ViT_B_16"
    engine.channel_order = (0, 2, 1)
    engine.device = "cpu"

    model = Mock()
    model.__class__.__name__ = "FakeNet"
    model.parameters.return_value = [Mock(numel=Mock(return_value=100)), Mock(numel=Mock(return_value=23))]
    engine.model = model

    service = ModelService()
    service.set_inference_engine(engine)
    info = service.get_model_info()

    assert info is not None
    assert info.architecture == "FakeNet"
    assert info.total_params == 123
    assert info.threshold == pytest.approx(0.61)
    assert info.backbone_name == "ViT_B_16"