"""模型生命周期服务。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scann.ai.inference import InferenceConfig, InferenceEngine
from scann.core.models import AppConfig


@dataclass(frozen=True)
class ModelLoadResult:
    """模型加载结果摘要。"""

    model_path: str
    model_threshold: float
    effective_threshold: float
    format_name: str
    backbone_name: str
    channel_order: tuple[int, ...]


@dataclass(frozen=True)
class ModelInfo:
    """模型信息快照。"""

    architecture: str
    total_params: int
    threshold: float
    format_name: str
    backbone_name: str
    channel_order: tuple[int, ...]
    device: str


class ModelService:
    """集中管理推理引擎生命周期与运行时参数同步。"""

    def __init__(
        self,
        engine_factory=InferenceEngine,
        config_factory=InferenceConfig,
    ) -> None:
        self._engine_factory = engine_factory
        self._config_factory = config_factory
        self._inference_engine = None

    @property
    def inference_engine(self):
        """当前活动的推理引擎。"""
        return self._inference_engine

    def set_inference_engine(self, inference_engine) -> None:
        """设置当前活动的推理引擎。"""
        self._inference_engine = inference_engine

    def clear_inference_engine(self) -> None:
        """清空当前活动的推理引擎。"""
        self._inference_engine = None

    def build_inference_config(self, app_config: AppConfig) -> InferenceConfig:
        """从应用配置构造推理配置。"""
        return self._config_factory(
            batch_size=app_config.batch_size,
            device=app_config.compute_device,
            model_format=app_config.model_format,
            model_backbone=getattr(app_config, "model_backbone", "auto"),
        )

    def load_model(self, model_path: str, app_config: AppConfig) -> ModelLoadResult:
        """加载模型并同步 GUI 运行时配置。"""
        config = self.build_inference_config(app_config)
        inference_engine = self._engine_factory(model_path=model_path, config=config)

        model_threshold = float(inference_engine.threshold)
        gui_threshold = float(app_config.ai_confidence)
        inference_engine.threshold = gui_threshold

        app_config.ai_confidence = float(inference_engine.threshold)
        app_config.model_path = model_path

        self._inference_engine = inference_engine
        return ModelLoadResult(
            model_path=model_path,
            model_threshold=model_threshold,
            effective_threshold=float(inference_engine.threshold),
            format_name=self._format_name(getattr(inference_engine, "model_format", None)),
            backbone_name=str(getattr(inference_engine, "model_backbone", "auto")),
            channel_order=self._channel_order_of(inference_engine),
        )

    def apply_runtime_config(self, app_config: AppConfig) -> bool:
        """把最新 GUI 参数应用到当前推理引擎。"""
        inference_engine = self._inference_engine
        if inference_engine is None or not getattr(inference_engine, "is_ready", False):
            return False

        inference_engine.threshold = app_config.ai_confidence
        inference_engine.config.batch_size = app_config.batch_size
        return True

    def get_model_info(self) -> ModelInfo | None:
        """读取当前模型摘要。"""
        inference_engine = self._inference_engine
        if inference_engine is None or not getattr(inference_engine, "is_ready", False):
            return None

        model = inference_engine.model
        total_params = sum(parameter.numel() for parameter in model.parameters())
        return ModelInfo(
            architecture=model.__class__.__name__,
            total_params=total_params,
            threshold=float(inference_engine.threshold),
            format_name=self._format_name(getattr(inference_engine, "model_format", None)),
            backbone_name=str(getattr(inference_engine, "model_backbone", "auto")),
            channel_order=self._channel_order_of(inference_engine),
            device=str(getattr(inference_engine, "device", "unknown")),
        )

    @staticmethod
    def _format_name(model_format: Any) -> str:
        if model_format is None:
            return "unknown"
        return str(getattr(model_format, "value", model_format))

    @staticmethod
    def _channel_order_of(inference_engine) -> tuple[int, ...]:
        channel_order = getattr(inference_engine, "channel_order", None)
        if channel_order is None:
            channel_order = getattr(inference_engine, "_channel_order", (0, 1, 2))
        try:
            return tuple(channel_order)
        except TypeError:
            fallback = getattr(inference_engine, "_channel_order", (0, 1, 2))
            try:
                return tuple(fallback)
            except TypeError:
                return (0, 1, 2)