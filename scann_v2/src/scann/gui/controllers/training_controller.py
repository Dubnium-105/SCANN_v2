"""训练流程控制器。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scann.ai.training_worker import TrainingWorker
from scann.gui.dialogs.training_dialog import TrainingDialog

if TYPE_CHECKING:
    from scann.gui.main_window import MainWindow


class TrainingController:
    """集中主窗口中的训练对话框和训练线程协调逻辑。"""

    def __init__(self, window: MainWindow) -> None:
        self._window = window
        self._training_dialog = None
        self._training_worker = None
        self._training_params: dict = {}

    @property
    def training_dialog(self):
        return self._training_dialog

    @property
    def training_worker(self):
        return self._training_worker

    @training_worker.setter
    def training_worker(self, worker) -> None:
        self._training_worker = worker

    @property
    def training_params(self) -> dict:
        return self._training_params

    @training_params.setter
    def training_params(self, params: dict) -> None:
        self._training_params = params

    def open_training(self) -> None:
        dialog = TrainingDialog(self._window)
        dialog.training_started.connect(self.training_started)
        dialog.training_stopped.connect(self.training_stopped)
        self._training_dialog = dialog
        self._training_worker = None
        dialog.exec_()

    def training_started(self, params: dict) -> None:
        self._window._show_message(
            f"训练已开始: epochs={params.get('epochs', '?')}, "
            f"lr={params.get('lr', '?')}, backbone={params.get('backbone', '?')}, "
            f"device={params.get('device', 'auto')}",
            5000,
        )
        self._training_params = dict(params)

        worker = TrainingWorker(params, parent=self._window)
        worker.progress.connect(self.training_progress)
        worker.finished.connect(self.training_finished)
        worker.error.connect(self.training_error)
        self._training_worker = worker
        worker.start()

    def training_progress(self, epoch: int, total: int, loss: float, val_loss: float) -> None:
        if self._training_dialog:
            self._training_dialog.update_progress(epoch, total, loss, val_loss)

    def training_finished(self, model_path: str, metrics: dict) -> None:
        if self._training_dialog:
            self._training_dialog.training_finished(model_path)
        self._training_worker = None
        best_f2 = metrics.get("best_f2", 0)
        best_threshold = metrics.get("best_threshold", 0.5)
        self._window._show_message(
            f"训练完成! 最佳 F2={best_f2:.4f}, 阈值={best_threshold:.3f}",
            5000,
        )

    def training_error(self, message: str) -> None:
        if self._training_dialog:
            self._training_dialog.log_text.appendPlainText(f"❌ 错误: {message}")
        self._training_worker = None
        self._window._show_message(f"训练失败: {message}", 5000, level="ERROR")

    def training_stopped(self) -> None:
        if self._training_worker:
            self._training_worker.stop()
        self._training_worker = None
        self._window._show_message("训练已停止")