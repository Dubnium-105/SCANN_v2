"""模型管理控制器。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt5.QtWidgets import QFileDialog, QMessageBox

from scann.services.model_service import ModelService

if TYPE_CHECKING:
    from scann.gui.main_window import MainWindow


class ModelController:
    """集中主窗口中的模型加载、信息展示与运行时同步入口。"""

    def __init__(self, window: MainWindow, model_service: ModelService) -> None:
        self._window = window
        self._model_service = model_service

    @property
    def model_service(self) -> ModelService:
        """暴露模型服务，供其他流程复用。"""
        return self._model_service

    def load_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self._window,
            "加载模型",
            "",
            "PyTorch 模型 (*.pth *.pt)",
        )
        if not path:
            return

        try:
            result = self._model_service.load_model(path, self._window._config)
            inference_engine = self._model_service.inference_engine
            self._window._logger.info(
                "模型已加载: %s (格式=%s, backbone=%s, 模型阈值=%.4f, GUI阈值=%.4f, 生效阈值=%.4f, 通道=%s)",
                path,
                result.format_name,
                result.backbone_name,
                result.model_threshold,
                float(self._window._config.ai_confidence),
                result.effective_threshold,
                result.channel_order,
            )
            self._window._show_message(
                f"模型已加载: {path} (格式={result.format_name}, backbone={result.backbone_name}, 阈值={result.effective_threshold:.2f})",
                5000,
            )
            if inference_engine is not None:
                self._window._config.ai_confidence = float(inference_engine.threshold)
        except Exception as exc:
            self._model_service.clear_inference_engine()
            self._window._show_message(f"模型加载失败: {exc}", 5000, level="ERROR")

    def show_model_info(self) -> None:
        info = self._model_service.get_model_info()
        if info is None:
            self._window._show_message("尚未加载模型")
            return

        QMessageBox.information(
            self._window,
            "模型信息",
            f"<h3>AI 模型信息</h3>"
            f"<p>架构: {info.architecture}</p>"
            f"<p>模型格式: {info.format_name}</p>"
            f"<p>模型主干: {info.backbone_name}</p>"
            f"<p>参数量: {info.total_params:,}</p>"
            f"<p>检测阈值: {info.threshold:.4f}</p>"
            f"<p>通道顺序: {info.channel_order}</p>"
            f"<p>设备: {info.device}</p>",
        )

    def apply_runtime_config(self) -> None:
        if not self._model_service.apply_runtime_config(self._window._config):
            return

        inference_engine = self._model_service.inference_engine
        self._window._logger.info(
            "已应用GUI推理参数: threshold=%.4f, batch_size=%d",
            float(inference_engine.threshold),
            int(inference_engine.config.batch_size),
        )