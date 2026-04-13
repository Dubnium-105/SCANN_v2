"""GPU worker control dialog for long-running training/prelabel loops."""

from __future__ import annotations

import logging
import os
import socket
import threading
from pathlib import Path
from typing import Iterable

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from scann.core.candidate_detector import DetectionParams
from scann.core.config import load_config
from scann.native_annotation.prelabel_worker import (
    PrelabelWorkerConfig,
    PrelabelWorkerRunner,
    WorkerDetectionConfig,
)
from scann.native_annotation.training_job_worker import (
    RemoteTrainingWorkerConfig,
    TrainingExecutionConfig,
    TrainingJobWorkerRunner,
)


logger = logging.getLogger(__name__)


class _SignalLogHandler(logging.Handler):
    def __init__(self, emitter) -> None:
        super().__init__()
        self._emitter = emitter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._emitter(self.format(record))
        except Exception:
            return


class _RunnerLoopThread(QThread):
    status_changed = pyqtSignal(str)
    processed_changed = pyqtSignal(int)
    log_message = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(
        self,
        *,
        runner,
        idle_poll_seconds: float,
        logger_names: Iterable[str],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._runner = runner
        self._idle_poll_seconds = max(0.5, float(idle_poll_seconds))
        self._logger_names = list(logger_names)
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        processed = 0
        handler = _SignalLogHandler(self.log_message.emit)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        attached_loggers: list[logging.Logger] = []
        previous_levels: dict[logging.Logger, int] = {}
        try:
            for name in self._logger_names:
                target = logging.getLogger(name)
                previous_levels[target] = target.level
                target.addHandler(handler)
                if target.level == logging.NOTSET or target.level > logging.INFO:
                    target.setLevel(logging.INFO)
                attached_loggers.append(target)

            self.status_changed.emit("运行中")
            while not self._stop_event.is_set():
                handled = bool(self._runner.run_once())
                if handled:
                    processed += 1
                    self.processed_changed.emit(processed)
                    continue
                self._stop_event.wait(self._idle_poll_seconds)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            for target in attached_loggers:
                target.removeHandler(handler)
                target.setLevel(previous_levels.get(target, logging.NOTSET))
            self.status_changed.emit("已停止")


class WorkerConsoleDialog(QDialog):
    def __init__(self, parent=None, runner_thread_cls=_RunnerLoopThread):
        super().__init__(parent)
        self.setWindowTitle("长期运行 GPU Worker 控制台")
        self.resize(840, 720)
        self.setMinimumSize(760, 640)

        self._runner_thread_cls = runner_thread_cls
        self._prelabel_thread = None
        self._training_thread = None

        self._init_ui()
        self._load_defaults()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        description = QLabel(
            "在有 GPU 的桌面端上直接启动预标注 worker 和训练 worker。"
            "服务器继续承担控制面，当前窗口只负责把本机 GPU 作为执行器接入。"
        )
        description.setWordWrap(True)
        description.setStyleSheet("color: #666;")
        layout.addWidget(description)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_prelabel_tab(), "预标注 Worker")
        self.tabs.addTab(self._build_training_tab(), "训练 Worker")
        layout.addWidget(self.tabs)

    def _build_prelabel_tab(self) -> QWidget:
        tab = QWidget()
        outer = QVBoxLayout(tab)

        form_group = QGroupBox("运行配置")
        form = QFormLayout(form_group)

        self.edit_prelabel_server_url = QLineEdit()
        self.edit_prelabel_token = QLineEdit()
        self.edit_prelabel_token.setEchoMode(QLineEdit.Password)
        self.edit_prelabel_worker_id = QLineEdit()
        self.edit_prelabel_display_name = QLineEdit()
        self.edit_prelabel_device_label = QLineEdit()
        self.edit_prelabel_dataset_root = QLineEdit()
        self.edit_prelabel_config_path = QLineEdit()
        self.edit_prelabel_model_path = QLineEdit()
        self.edit_prelabel_model_version = QLineEdit()
        self.edit_prelabel_model_id = QLineEdit()
        self.edit_prelabel_model_backbone = QLineEdit()
        self.edit_prelabel_compute_device = QLineEdit()
        self.edit_prelabel_idle_seconds = QLineEdit("5")
        self.edit_prelabel_heartbeat_seconds = QLineEdit("30")
        self.edit_prelabel_timeout_seconds = QLineEdit("60")

        form.addRow("服务器 URL", self.edit_prelabel_server_url)
        form.addRow("Worker Token", self.edit_prelabel_token)
        form.addRow("Worker ID", self.edit_prelabel_worker_id)
        form.addRow("显示名称", self.edit_prelabel_display_name)
        form.addRow("设备标签", self.edit_prelabel_device_label)
        form.addRow("数据集根目录", self._path_row(self.edit_prelabel_dataset_root, directory=True))
        form.addRow("本地配置文件", self._path_row(self.edit_prelabel_config_path, open_file=True))
        form.addRow("模型路径", self._path_row(self.edit_prelabel_model_path, open_file=True))
        form.addRow("模型版本", self.edit_prelabel_model_version)
        form.addRow("模型 ID", self.edit_prelabel_model_id)
        form.addRow("模型骨干", self.edit_prelabel_model_backbone)
        form.addRow("推理设备", self.edit_prelabel_compute_device)
        form.addRow("空闲轮询秒数", self.edit_prelabel_idle_seconds)
        form.addRow("心跳秒数", self.edit_prelabel_heartbeat_seconds)
        form.addRow("请求超时秒数", self.edit_prelabel_timeout_seconds)
        outer.addWidget(form_group)

        status_row = QHBoxLayout()
        self.lbl_prelabel_status = QLabel("状态：未启动")
        self.lbl_prelabel_processed = QLabel("已处理：0")
        status_row.addWidget(self.lbl_prelabel_status)
        status_row.addStretch()
        status_row.addWidget(self.lbl_prelabel_processed)
        outer.addLayout(status_row)

        self.log_prelabel = QPlainTextEdit()
        self.log_prelabel.setReadOnly(True)
        self.log_prelabel.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        outer.addWidget(self.log_prelabel, 1)

        button_row = QHBoxLayout()
        self.btn_prelabel_start = QPushButton("启动预标注 Worker")
        self.btn_prelabel_stop = QPushButton("停止")
        self.btn_prelabel_stop.setEnabled(False)
        self.btn_prelabel_start.clicked.connect(self._start_prelabel_worker)
        self.btn_prelabel_stop.clicked.connect(self._stop_prelabel_worker)
        button_row.addWidget(self.btn_prelabel_start)
        button_row.addWidget(self.btn_prelabel_stop)
        button_row.addStretch()
        outer.addLayout(button_row)

        return tab

    def _build_training_tab(self) -> QWidget:
        tab = QWidget()
        outer = QVBoxLayout(tab)

        form_group = QGroupBox("运行配置")
        form = QFormLayout(form_group)

        self.edit_training_server_url = QLineEdit()
        self.edit_training_token = QLineEdit()
        self.edit_training_token.setEchoMode(QLineEdit.Password)
        self.edit_training_worker_id = QLineEdit()
        self.edit_training_display_name = QLineEdit()
        self.edit_training_device_label = QLineEdit()
        self.edit_training_dataset_root = QLineEdit()
        self.edit_training_output_root = QLineEdit()
        self.edit_training_task_types = QLineEdit("classification")
        self.edit_training_model_backbones = QLineEdit("ViT_B_16,ResNet18")
        self.edit_training_device = QLineEdit("auto")
        self.edit_training_idle_seconds = QLineEdit("10")
        self.edit_training_heartbeat_seconds = QLineEdit("60")
        self.edit_training_timeout_seconds = QLineEdit("120")

        form.addRow("服务器 URL", self.edit_training_server_url)
        form.addRow("Worker Token", self.edit_training_token)
        form.addRow("Worker ID", self.edit_training_worker_id)
        form.addRow("显示名称", self.edit_training_display_name)
        form.addRow("设备标签", self.edit_training_device_label)
        form.addRow("数据集根目录", self._path_row(self.edit_training_dataset_root, directory=True))
        form.addRow("输出目录", self._path_row(self.edit_training_output_root, directory=True))
        form.addRow("任务类型", self.edit_training_task_types)
        form.addRow("支持骨干", self.edit_training_model_backbones)
        form.addRow("训练设备", self.edit_training_device)
        form.addRow("空闲轮询秒数", self.edit_training_idle_seconds)
        form.addRow("心跳秒数", self.edit_training_heartbeat_seconds)
        form.addRow("请求超时秒数", self.edit_training_timeout_seconds)
        outer.addWidget(form_group)

        status_row = QHBoxLayout()
        self.lbl_training_status = QLabel("状态：未启动")
        self.lbl_training_processed = QLabel("已处理：0")
        status_row.addWidget(self.lbl_training_status)
        status_row.addStretch()
        status_row.addWidget(self.lbl_training_processed)
        outer.addLayout(status_row)

        self.log_training = QPlainTextEdit()
        self.log_training.setReadOnly(True)
        self.log_training.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        outer.addWidget(self.log_training, 1)

        button_row = QHBoxLayout()
        self.btn_training_start = QPushButton("启动训练 Worker")
        self.btn_training_stop = QPushButton("停止")
        self.btn_training_stop.setEnabled(False)
        self.btn_training_start.clicked.connect(self._start_training_worker)
        self.btn_training_stop.clicked.connect(self._stop_training_worker)
        button_row.addWidget(self.btn_training_start)
        button_row.addWidget(self.btn_training_stop)
        button_row.addStretch()
        outer.addLayout(button_row)

        return tab

    def _path_row(self, target: QLineEdit, *, directory: bool = False, open_file: bool = False) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(target, 1)
        button = QPushButton("浏览...")
        if directory:
            button.clicked.connect(lambda: self._browse_dir(target))
        elif open_file:
            button.clicked.connect(lambda: self._browse_file(target))
        layout.addWidget(button)
        return row

    def _load_defaults(self) -> None:
        host_name = socket.gethostname()
        parent = self.parentWidget()
        dataset_root = str(getattr(parent, "_dataset_root", "") or os.getenv("SCANN_PRELABEL_WORKER_DATASET_ROOT", "")).strip()
        if not dataset_root:
            dataset_root = str(os.getenv("SCANN_TRAINING_WORKER_DATASET_ROOT", "")).strip()

        app_config = None
        try:
            app_config = load_config(None)
        except Exception:
            app_config = None

        model_path = str(getattr(app_config, "model_path", "") or os.getenv("SCANN_PRELABEL_WORKER_MODEL_PATH", "")).strip()
        model_backbone = str(
            os.getenv("SCANN_PRELABEL_WORKER_MODEL_BACKBONE", "") or getattr(app_config, "model_backbone", "auto") or "auto"
        ).strip()
        compute_device = str(
            os.getenv("SCANN_PRELABEL_WORKER_COMPUTE_DEVICE", "") or getattr(app_config, "compute_device", "auto") or "auto"
        ).strip()
        output_root = str(
            os.getenv("SCANN_TRAINING_WORKER_OUTPUT_ROOT", "") or (str(Path(dataset_root) / ".scann_worker_output") if dataset_root else "")
        ).strip()

        self.edit_prelabel_server_url.setText(os.getenv("SCANN_PRELABEL_SERVER_URL", "").strip())
        self.edit_prelabel_token.setText(os.getenv("SCANN_PRELABEL_WORKER_TOKEN", "").strip())
        self.edit_prelabel_worker_id.setText(os.getenv("SCANN_PRELABEL_WORKER_ID", f"{host_name}-prelabel").strip())
        self.edit_prelabel_display_name.setText(os.getenv("SCANN_PRELABEL_WORKER_NAME", f"{host_name}-prelabel").strip())
        self.edit_prelabel_device_label.setText(os.getenv("SCANN_PRELABEL_WORKER_DEVICE_LABEL", "").strip())
        self.edit_prelabel_dataset_root.setText(dataset_root)
        self.edit_prelabel_config_path.setText(os.getenv("SCANN_PRELABEL_WORKER_CONFIG_PATH", "").strip())
        self.edit_prelabel_model_path.setText(model_path)
        self.edit_prelabel_model_version.setText(os.getenv("SCANN_PRELABEL_WORKER_MODEL_VERSION", Path(model_path).stem if model_path else "").strip())
        self.edit_prelabel_model_id.setText(os.getenv("SCANN_PRELABEL_WORKER_MODEL_ID", Path(model_path).stem if model_path else "").strip())
        self.edit_prelabel_model_backbone.setText(model_backbone or "auto")
        self.edit_prelabel_compute_device.setText(compute_device or "auto")

        self.edit_training_server_url.setText(os.getenv("SCANN_TRAINING_SERVER_URL", "").strip())
        self.edit_training_token.setText(os.getenv("SCANN_TRAINING_WORKER_TOKEN", "").strip())
        self.edit_training_worker_id.setText(os.getenv("SCANN_TRAINING_WORKER_ID", f"{host_name}-trainer").strip())
        self.edit_training_display_name.setText(os.getenv("SCANN_TRAINING_WORKER_NAME", f"{host_name}-trainer").strip())
        self.edit_training_device_label.setText(os.getenv("SCANN_TRAINING_WORKER_DEVICE_LABEL", "").strip())
        self.edit_training_dataset_root.setText(str(os.getenv("SCANN_TRAINING_WORKER_DATASET_ROOT", dataset_root)).strip())
        self.edit_training_output_root.setText(output_root)
        self.edit_training_task_types.setText(os.getenv("SCANN_TRAINING_WORKER_TASK_TYPES", "classification").strip())
        self.edit_training_model_backbones.setText(
            os.getenv("SCANN_TRAINING_WORKER_MODEL_BACKBONES", "ViT_B_16,ResNet18").strip()
        )
        self.edit_training_device.setText(os.getenv("SCANN_TRAINING_WORKER_DEVICE", "auto").strip())

    def _browse_dir(self, line_edit: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择目录")
        if path:
            line_edit.setText(path)

    def _browse_file(self, line_edit: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择文件")
        if path:
            line_edit.setText(path)

    @staticmethod
    def _parse_csv(raw: str) -> list[str]:
        return [item.strip() for item in str(raw or "").split(",") if item and item.strip()]

    def _load_prelabel_app_config(self):
        config_path = self.edit_prelabel_config_path.text().strip()
        return load_config(config_path or None)

    def _build_prelabel_config(self) -> PrelabelWorkerConfig:
        app_config = self._load_prelabel_app_config()
        model_path = self.edit_prelabel_model_path.text().strip() or str(getattr(app_config, "model_path", "") or "")
        if not model_path:
            raise ValueError("预标注模型路径不能为空")

        detection_params = DetectionParams(
            thresh=int(getattr(app_config, "thresh", 80)),
            min_area=int(getattr(app_config, "min_area", 6)),
            max_area=int(getattr(app_config, "max_area", 600)),
            sharpness_min=float(getattr(app_config, "sharpness", 1.2)),
            sharpness_max=float(getattr(app_config, "max_sharpness", 5.0)),
            contrast_min=int(getattr(app_config, "contrast", 15)),
            edge_margin=int(getattr(app_config, "edge_margin", 10)),
            dynamic_thresh=bool(getattr(app_config, "dynamic_thresh", False)),
            kill_flat=bool(getattr(app_config, "kill_flat", True)),
            kill_dipole=bool(getattr(app_config, "kill_dipole", True)),
            aspect_ratio_max=float(getattr(app_config, "aspect_ratio_max", 3.0)),
            extent_max=float(getattr(app_config, "extent_max", 0.9)),
            topk=int(getattr(app_config, "topk", 20)),
        )
        detection = WorkerDetectionConfig(
            model_path=model_path,
            model_version=self.edit_prelabel_model_version.text().strip() or Path(model_path).stem,
            model_id=self.edit_prelabel_model_id.text().strip() or Path(model_path).stem,
            model_format=str(getattr(app_config, "model_format", "auto") or "auto"),
            model_backbone=self.edit_prelabel_model_backbone.text().strip() or str(getattr(app_config, "model_backbone", "auto") or "auto"),
            compute_device=self.edit_prelabel_compute_device.text().strip() or str(getattr(app_config, "compute_device", "auto") or "auto"),
            batch_size=int(getattr(app_config, "batch_size", 64)),
            patch_size=int(getattr(app_config, "slice_size", 80)),
            detection_mode=str(getattr(app_config, "detection_mode", "patch") or "patch"),
            hybrid_primary_mode=str(getattr(app_config, "hybrid_primary_mode", "full_image") or "full_image"),
            hybrid_low_confidence=float(getattr(app_config, "hybrid_low_confidence", 0.5)),
            detection_params=detection_params,
        )

        dataset_root_text = self.edit_prelabel_dataset_root.text().strip()
        return PrelabelWorkerConfig(
            server_url=self.edit_prelabel_server_url.text().strip().rstrip("/"),
            worker_token=self.edit_prelabel_token.text().strip(),
            worker_id=self.edit_prelabel_worker_id.text().strip(),
            display_name=self.edit_prelabel_display_name.text().strip() or self.edit_prelabel_worker_id.text().strip(),
            host_name=socket.gethostname(),
            device_label=self.edit_prelabel_device_label.text().strip() or None,
            dataset_root=Path(dataset_root_text).resolve() if dataset_root_text else None,
            detection=detection,
            idle_poll_seconds=float(self.edit_prelabel_idle_seconds.text().strip() or 5),
            heartbeat_interval_seconds=float(self.edit_prelabel_heartbeat_seconds.text().strip() or 30),
            request_timeout_seconds=float(self.edit_prelabel_timeout_seconds.text().strip() or 60),
        )

    def _build_training_config(self) -> RemoteTrainingWorkerConfig:
        dataset_root_text = self.edit_training_dataset_root.text().strip()
        output_root_text = self.edit_training_output_root.text().strip()
        if not dataset_root_text:
            raise ValueError("训练数据集根目录不能为空")
        if not output_root_text:
            raise ValueError("训练输出目录不能为空")

        execution = TrainingExecutionConfig(
            dataset_root=Path(dataset_root_text).resolve(),
            output_root=Path(output_root_text).resolve(),
            task_types=self._parse_csv(self.edit_training_task_types.text()),
            model_backbones=self._parse_csv(self.edit_training_model_backbones.text()),
            device=self.edit_training_device.text().strip() or "auto",
        )
        return RemoteTrainingWorkerConfig(
            server_url=self.edit_training_server_url.text().strip().rstrip("/"),
            worker_token=self.edit_training_token.text().strip(),
            worker_id=self.edit_training_worker_id.text().strip(),
            display_name=self.edit_training_display_name.text().strip() or self.edit_training_worker_id.text().strip(),
            host_name=socket.gethostname(),
            device_label=self.edit_training_device_label.text().strip() or None,
            execution=execution,
            idle_poll_seconds=float(self.edit_training_idle_seconds.text().strip() or 10),
            heartbeat_interval_seconds=float(self.edit_training_heartbeat_seconds.text().strip() or 60),
            request_timeout_seconds=float(self.edit_training_timeout_seconds.text().strip() or 120),
        )

    def _append_log(self, widget: QPlainTextEdit, text: str) -> None:
        widget.appendPlainText(text)

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)

    def _wire_thread(self, *, thread, status_label: QLabel, processed_label: QLabel, log_widget: QPlainTextEdit) -> None:
        thread.status_changed.connect(lambda text: status_label.setText(f"状态：{text}"))
        thread.processed_changed.connect(lambda count: processed_label.setText(f"已处理：{count}"))
        thread.log_message.connect(lambda text: self._append_log(log_widget, text))
        thread.failed.connect(lambda text: self._append_log(log_widget, f"ERROR: {text}"))

    def _start_prelabel_worker(self) -> None:
        if self._prelabel_thread is not None and self._prelabel_thread.isRunning():
            return
        try:
            config = self._build_prelabel_config()
        except Exception as exc:
            self._show_error("预标注 Worker", str(exc))
            return
        runner = PrelabelWorkerRunner(config)
        thread = self._runner_thread_cls(
            runner=runner,
            idle_poll_seconds=config.idle_poll_seconds,
            logger_names=[
                "scann.native_annotation.prelabel_worker",
                "scann.services.detection_pipeline",
            ],
            parent=self,
        )
        self._wire_thread(
            thread=thread,
            status_label=self.lbl_prelabel_status,
            processed_label=self.lbl_prelabel_processed,
            log_widget=self.log_prelabel,
        )
        thread.finished.connect(self._on_prelabel_thread_finished)
        self._prelabel_thread = thread
        self.btn_prelabel_start.setEnabled(False)
        self.btn_prelabel_stop.setEnabled(True)
        self._append_log(self.log_prelabel, f"INFO: 启动预标注 worker {config.worker_id}")
        thread.start()

    def _stop_prelabel_worker(self) -> None:
        if self._prelabel_thread is None:
            return
        self._prelabel_thread.stop()
        self._append_log(self.log_prelabel, "INFO: 请求停止预标注 worker")

    def _on_prelabel_thread_finished(self) -> None:
        self._prelabel_thread = None
        self.btn_prelabel_start.setEnabled(True)
        self.btn_prelabel_stop.setEnabled(False)

    def _start_training_worker(self) -> None:
        if self._training_thread is not None and self._training_thread.isRunning():
            return
        try:
            config = self._build_training_config()
        except Exception as exc:
            self._show_error("训练 Worker", str(exc))
            return
        runner = TrainingJobWorkerRunner(config)
        thread = self._runner_thread_cls(
            runner=runner,
            idle_poll_seconds=config.idle_poll_seconds,
            logger_names=[
                "scann.native_annotation.training_job_worker",
                "scann.ai.training_worker",
            ],
            parent=self,
        )
        self._wire_thread(
            thread=thread,
            status_label=self.lbl_training_status,
            processed_label=self.lbl_training_processed,
            log_widget=self.log_training,
        )
        thread.finished.connect(self._on_training_thread_finished)
        self._training_thread = thread
        self.btn_training_start.setEnabled(False)
        self.btn_training_stop.setEnabled(True)
        self._append_log(self.log_training, f"INFO: 启动训练 worker {config.worker_id}")
        thread.start()

    def _stop_training_worker(self) -> None:
        if self._training_thread is None:
            return
        self._training_thread.stop()
        self._append_log(self.log_training, "INFO: 请求停止训练 worker")

    def _on_training_thread_finished(self) -> None:
        self._training_thread = None
        self.btn_training_start.setEnabled(True)
        self.btn_training_stop.setEnabled(False)

    def closeEvent(self, event) -> None:
        for thread in [self._prelabel_thread, self._training_thread]:
            if thread is not None and thread.isRunning():
                thread.stop()
                thread.wait(2000)
        super().closeEvent(event)
