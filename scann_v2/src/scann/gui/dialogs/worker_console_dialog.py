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

    def _build_prelabel_tab_legacy(self) -> QWidget:
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

    def _build_training_tab_legacy(self) -> QWidget:
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

    def _build_prelabel_tab(self) -> QWidget:
        tab = QWidget()
        outer = QVBoxLayout(tab)

        form_group = QGroupBox("\u8fd0\u884c\u914d\u7f6e")
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

        self._add_help_row(
            form,
            "\u670d\u52a1\u5668 URL",
            self.edit_prelabel_server_url,
            "\u586b\u5199 worker \u53ef\u4ee5\u8bbf\u95ee\u5230\u7684 API \u5165\u53e3\u5730\u5740\uff0c\u4f8b\u5982 http://192.168.1.10:8000\u3002\u5982\u679c\u53ea\u6709\u524d\u7aef\u5bb9\u5668\u5bf9\u516c\u7f51\u66b4\u9732\uff0c\u8fd9\u91cc\u5c31\u586b\u524d\u7aef\u516c\u7f51\u5730\u5740\uff0cNGINX \u4f1a\u4ee3\u7406 /api/* \u5230\u540e\u7aef\u3002",
            placeholder="http://127.0.0.1:8000",
        )
        self._add_help_row(
            form,
            "Worker Token",
            self.edit_prelabel_token,
            "\u4f7f\u7528\u540e\u7aef\u4e3a\u9884\u6807\u6ce8 worker \u7b7e\u53d1\u7684\u4e13\u7528 token\uff0c\u4e0d\u8981\u586b\u5199\u4eba\u5de5\u6807\u6ce8\u8d26\u53f7\u7684\u767b\u5f55 token\u3002",
        )
        self._add_help_row(
            form,
            "Worker ID",
            self.edit_prelabel_worker_id,
            "\u4fdd\u6301\u957f\u671f\u7a33\u5b9a\u7684\u552f\u4e00\u6807\u8bc6\uff0c\u7528\u4e8e\u961f\u5217\u9886\u53d6\u3001\u65e5\u5fd7\u8ffd\u8e2a\u548c\u65ad\u7ebf\u6062\u590d\u3002",
            placeholder="gpu-prelabel-1",
        )
        self._add_help_row(
            form,
            "\u663e\u793a\u540d\u79f0",
            self.edit_prelabel_display_name,
            "\u7528\u4e8e Web \u7ba1\u7406\u7aef\u548c\u65e5\u5fd7\u4e2d\u5c55\u793a\u7684\u53ef\u8bfb\u540d\u79f0\uff0c\u53ef\u4ee5\u586b\u5199 GPU \u673a\u5668\u540d\u6216\u90e8\u7f72\u4f4d\u7f6e\u3002",
            placeholder="Office RTX 4090",
        )
        self._add_help_row(
            form,
            "\u8bbe\u5907\u6807\u7b7e",
            self.edit_prelabel_device_label,
            "\u586b\u5199\u7ed9\u8fd9\u53f0 worker \u7684\u786c\u4ef6\u8bf4\u660e\uff0c\u4f8b\u5982 RTX-4090 / CUDA12\uff0c\u65b9\u4fbf\u8fd0\u7ef4\u6392\u67e5\u3002",
            placeholder="RTX-4090",
        )
        self._add_help_row(
            form,
            "\u6570\u636e\u96c6\u6839\u76ee\u5f55",
            self._path_row(self.edit_prelabel_dataset_root, directory=True),
            "\u586b\u5199\u672c\u5730 worker \u80fd\u8bfb\u5230\u7684\u6570\u636e\u96c6\u6839\u76ee\u5f55\u3002\u5982\u679c\u4f7f\u7528\u5171\u4eab\u76d8\uff0c\u8fd9\u91cc\u8981\u586b\u6302\u8f7d\u540e\u7684\u672c\u5730\u8def\u5f84\u3002",
            placeholder="G:/datasets/scann",
        )
        self._add_help_row(
            form,
            "\u672c\u5730\u914d\u7f6e\u6587\u4ef6",
            self._path_row(self.edit_prelabel_config_path, open_file=True),
            "\u53ef\u9009\u3002\u7528\u6765\u52a0\u8f7d\u9608\u503c\u3001patch \u5927\u5c0f\u3001\u6a21\u578b\u683c\u5f0f\u7b49\u63a8\u7406\u53c2\u6570\uff0c\u4e0d\u586b\u65f6\u4f7f\u7528\u7a0b\u5e8f\u9ed8\u8ba4\u503c\u3002",
            placeholder="config.yaml",
        )
        self._add_help_row(
            form,
            "\u6a21\u578b\u8def\u5f84",
            self._path_row(self.edit_prelabel_model_path, open_file=True),
            "\u586b\u5199\u672c\u5730 GPU \u673a\u5668\u4e0a\u7684 checkpoint \u6216\u5bfc\u51fa\u6a21\u578b\u6587\u4ef6\u8def\u5f84\u3002worker \u5b9e\u9645\u4ece\u8fd9\u91cc\u52a0\u8f7d\u9884\u6807\u6ce8\u6a21\u578b\u3002",
            placeholder="model-best.pth",
        )
        self._add_help_row(
            form,
            "\u6a21\u578b\u7248\u672c",
            self.edit_prelabel_model_version,
            "\u8868\u793a\u9884\u6807\u6ce8\u7528\u7684\u903b\u8f91\u7248\u672c\uff0c\u7528\u4e8e\u5de5\u5355\u5339\u914d\u3001\u56de\u6eaf\u548c\u91cd\u751f\u6210\u3002",
            placeholder="detector-v3",
        )
        self._add_help_row(
            form,
            "\u6a21\u578b ID",
            self.edit_prelabel_model_id,
            "\u586b\u5199\u6a21\u578b\u6ce8\u518c\u8868\u6216\u8bad\u7ec3 run \u7684\u552f\u4e00 ID\uff0c\u7528\u4e8e\u7cbe\u786e\u533a\u5206\u540c\u7248\u672c\u4e0b\u7684\u4e0d\u540c checkpoint\u3002",
            placeholder="detector-v3-run-001",
        )
        self._add_help_row(
            form,
            "\u6a21\u578b\u9aa8\u5e72",
            self.edit_prelabel_model_backbone,
            "\u586b\u5199\u6a21\u578b\u7684 backbone \u540d\u79f0\uff0c\u4f8b\u5982 ViT_B_16\u3001ResNet18\uff0c\u7528\u4e8e\u80fd\u529b\u5339\u914d\u548c\u4e0a\u7ebf\u8bb0\u5f55\u3002",
            placeholder="ViT_B_16",
        )
        self._add_help_row(
            form,
            "\u63a8\u7406\u8bbe\u5907",
            self.edit_prelabel_compute_device,
            "\u586b\u5199 auto\u3001cuda\u3001cuda:0 \u6216 cpu\u3002\u591a GPU \u673a\u5668\u53ef\u4ee5\u6307\u5b9a\u5230\u5355\u5361\u3002",
            placeholder="cuda:0",
        )
        self._add_help_row(
            form,
            "\u7a7a\u95f2\u8f6e\u8be2\u79d2\u6570",
            self.edit_prelabel_idle_seconds,
            "\u961f\u5217\u6682\u65f6\u6ca1\u6709\u4efb\u52a1\u65f6\uff0cworker \u6bcf\u9694\u591a\u4e45\u518d\u53bb claim \u4e00\u6b21\u3002",
            placeholder="5",
        )
        self._add_help_row(
            form,
            "\u5fc3\u8df3\u79d2\u6570",
            self.edit_prelabel_heartbeat_seconds,
            "\u8fd0\u884c\u4efb\u52a1\u65f6\u5411\u670d\u52a1\u5668\u62a5\u6d3b\u7684\u95f4\u9694\u3002\u4e00\u822c\u5e94\u5c0f\u4e8e\u540e\u7aef\u7684 stale \u8d85\u65f6\u9608\u503c\u3002",
            placeholder="30",
        )
        self._add_help_row(
            form,
            "\u8bf7\u6c42\u8d85\u65f6\u79d2\u6570",
            self.edit_prelabel_timeout_seconds,
            "\u5355\u6b21 HTTP \u8bf7\u6c42\u7684\u8d85\u65f6\u65f6\u95f4\u3002\u670d\u52a1\u5668\u6216\u5171\u4eab\u5b58\u50a8\u8f83\u6162\u65f6\u53ef\u4ee5\u9002\u5f53\u8c03\u5927\u3002",
            placeholder="60",
        )
        outer.addWidget(form_group)

        status_row = QHBoxLayout()
        self.lbl_prelabel_status = QLabel("\u72b6\u6001\uff1a\u672a\u542f\u52a8")
        self.lbl_prelabel_processed = QLabel("\u5df2\u5904\u7406\uff1a0")
        status_row.addWidget(self.lbl_prelabel_status)
        status_row.addStretch()
        status_row.addWidget(self.lbl_prelabel_processed)
        outer.addLayout(status_row)

        self.log_prelabel = QPlainTextEdit()
        self.log_prelabel.setReadOnly(True)
        self.log_prelabel.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        outer.addWidget(self.log_prelabel, 1)

        button_row = QHBoxLayout()
        self.btn_prelabel_start = QPushButton("\u542f\u52a8\u9884\u6807\u6ce8 Worker")
        self.btn_prelabel_stop = QPushButton("\u505c\u6b62")
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

        form_group = QGroupBox("\u8fd0\u884c\u914d\u7f6e")
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

        self._add_help_row(
            form,
            "\u670d\u52a1\u5668 URL",
            self.edit_training_server_url,
            "\u586b\u5199 worker \u53ef\u4ee5\u8bbf\u95ee\u5230\u7684 API \u5165\u53e3\u5730\u5740\uff0c\u901a\u5e38\u548c\u9884\u6807\u6ce8 worker \u4f7f\u7528\u540c\u4e00\u4e2a API \u670d\u52a1\u3002\u5982\u679c\u53ea\u66b4\u9732\u4e86\u524d\u7aef\u5bb9\u5668\uff0c\u8bf7\u76f4\u63a5\u586b\u524d\u7aef\u516c\u7f51 URL\u3002",
            placeholder="http://127.0.0.1:8000",
        )
        self._add_help_row(
            form,
            "Worker Token",
            self.edit_training_token,
            "\u4f7f\u7528\u540e\u7aef\u4e3a\u8bad\u7ec3 worker \u7b7e\u53d1\u7684 token\uff0c\u4fbf\u4e8e\u5355\u72ec\u63a7\u5236\u8bad\u7ec3\u6743\u9650\u548c\u8ba1\u8d39\u98ce\u9669\u3002",
        )
        self._add_help_row(
            form,
            "Worker ID",
            self.edit_training_worker_id,
            "\u4fdd\u6301\u7a33\u5b9a\u7684\u8bad\u7ec3\u8282\u70b9 ID\uff0c\u7528\u4e8e\u8bad\u7ec3 job claim\u3001\u5fc3\u8df3\u548c\u8fd0\u884c\u8bb0\u5f55\u5173\u8054\u3002",
            placeholder="gpu-trainer-1",
        )
        self._add_help_row(
            form,
            "\u663e\u793a\u540d\u79f0",
            self.edit_training_display_name,
            "\u7528\u4e8e\u7ba1\u7406\u7aef\u548c\u65e5\u5fd7\u4e2d\u663e\u793a\u7684\u53cb\u597d\u540d\u79f0\uff0c\u4f8b\u5982\u673a\u5668\u540d\u6216\u90e8\u7f72\u6240\u5728\u5730\u3002",
            placeholder="Lab Trainer A6000",
        )
        self._add_help_row(
            form,
            "\u8bbe\u5907\u6807\u7b7e",
            self.edit_training_device_label,
            "\u5199\u660e\u8fd9\u53f0\u8bad\u7ec3\u8282\u70b9\u7684\u4e3b\u8981\u786c\u4ef6\uff0c\u4f8b\u5982 RTX-A6000 / 48GB\uff0c\u65b9\u4fbf\u8c03\u5ea6\u548c\u6392\u969c\u3002",
            placeholder="RTX-A6000",
        )
        self._add_help_row(
            form,
            "\u6570\u636e\u96c6\u6839\u76ee\u5f55",
            self._path_row(self.edit_training_dataset_root, directory=True),
            "\u8bad\u7ec3 worker \u5728\u672c\u5730\u770b\u5230\u7684\u6570\u636e\u96c6\u6839\u76ee\u5f55\u3002\u5feb\u7167\u5bfc\u51fa\u7684\u6807\u6ce8\u6587\u6863\u4e5f\u4f1a\u76f8\u5bf9\u5230\u8fd9\u4e2a\u76ee\u5f55\u89e3\u6790\u3002",
            placeholder="G:/datasets/scann",
        )
        self._add_help_row(
            form,
            "\u8f93\u51fa\u76ee\u5f55",
            self._path_row(self.edit_training_output_root, directory=True),
            "\u7528\u6765\u5b58\u653e checkpoint\u3001\u65e5\u5fd7\u3001\u4e2d\u95f4\u5feb\u7167\u548c\u5bfc\u51fa\u7ed3\u679c\u3002\u5efa\u8bae\u4f7f\u7528\u5269\u4f59\u7a7a\u95f4\u5145\u8db3\u7684\u76d8\u7b26\u3002",
            placeholder="G:/.scann_worker_output",
        )
        self._add_help_row(
            form,
            "\u4efb\u52a1\u7c7b\u578b",
            self.edit_training_task_types,
            "\u7528\u82f1\u6587\u9017\u53f7\u5206\u9694\u586b\u5199 worker \u652f\u6301\u7684\u8bad\u7ec3\u4efb\u52a1\uff0c\u4f8b\u5982 classification,detection\u3002",
            placeholder="classification,detection",
        )
        self._add_help_row(
            form,
            "\u652f\u6301\u9aa8\u5e72",
            self.edit_training_model_backbones,
            "\u7528\u82f1\u6587\u9017\u53f7\u586b\u5199\u53ef\u8bad\u7ec3\u7684 backbone \u5217\u8868\uff0c\u670d\u52a1\u5668\u4f1a\u636e\u6b64\u5339\u914d\u5408\u9002\u7684\u8bad\u7ec3 job\u3002",
            placeholder="ViT_B_16,ResNet18",
        )
        self._add_help_row(
            form,
            "\u8bad\u7ec3\u8bbe\u5907",
            self.edit_training_device,
            "\u586b\u5199 auto\u3001cuda\u3001cuda:0 \u6216 cpu\u3002\u5982\u679c\u4f60\u60f3\u8ba9\u8bad\u7ec3\u548c\u9884\u6807\u6ce8\u5360\u7528\u4e0d\u540c GPU\uff0c\u53ef\u5728\u8fd9\u91cc\u5355\u72ec\u6307\u5b9a\u3002",
            placeholder="cuda:0",
        )
        self._add_help_row(
            form,
            "\u7a7a\u95f2\u8f6e\u8be2\u79d2\u6570",
            self.edit_training_idle_seconds,
            "\u5f53\u6ca1\u6709\u8bad\u7ec3 job \u65f6\uff0cworker \u591a\u4e45\u5411\u670d\u52a1\u5668\u8bf7\u6c42\u4e00\u6b21\u65b0\u4efb\u52a1\u3002",
            placeholder="10",
        )
        self._add_help_row(
            form,
            "\u5fc3\u8df3\u79d2\u6570",
            self.edit_training_heartbeat_seconds,
            "\u8bad\u7ec3\u8fc7\u7a0b\u4e2d\u4e0a\u62a5\u5b58\u6d3b\u72b6\u6001\u7684\u95f4\u9694\u3002\u957f\u8bad\u7ec3\u4efb\u52a1\u5efa\u8bae\u7565\u5c0f\u4e8e\u670d\u52a1\u5668\u56de\u6536\u8d85\u65f6\u3002",
            placeholder="60",
        )
        self._add_help_row(
            form,
            "\u8bf7\u6c42\u8d85\u65f6\u79d2\u6570",
            self.edit_training_timeout_seconds,
            "\u8bad\u7ec3 worker \u4e0e\u540e\u7aef\u901a\u4fe1\u7684 HTTP \u8d85\u65f6\u3002\u5982\u679c\u5feb\u7167\u6216\u6a21\u578b\u4e0a\u4f20\u8f83\u5927\uff0c\u53ef\u4ee5\u76f8\u5e94\u589e\u5927\u3002",
            placeholder="120",
        )
        outer.addWidget(form_group)

        status_row = QHBoxLayout()
        self.lbl_training_status = QLabel("\u72b6\u6001\uff1a\u672a\u542f\u52a8")
        self.lbl_training_processed = QLabel("\u5df2\u5904\u7406\uff1a0")
        status_row.addWidget(self.lbl_training_status)
        status_row.addStretch()
        status_row.addWidget(self.lbl_training_processed)
        outer.addLayout(status_row)

        self.log_training = QPlainTextEdit()
        self.log_training.setReadOnly(True)
        self.log_training.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        outer.addWidget(self.log_training, 1)

        button_row = QHBoxLayout()
        self.btn_training_start = QPushButton("\u542f\u52a8\u8bad\u7ec3 Worker")
        self.btn_training_stop = QPushButton("\u505c\u6b62")
        self.btn_training_stop.setEnabled(False)
        self.btn_training_start.clicked.connect(self._start_training_worker)
        self.btn_training_stop.clicked.connect(self._stop_training_worker)
        button_row.addWidget(self.btn_training_start)
        button_row.addWidget(self.btn_training_stop)
        button_row.addStretch()
        outer.addLayout(button_row)

        return tab

    def _add_help_row(
        self,
        form: QFormLayout,
        label: str,
        control: QWidget,
        help_text: str,
        *,
        placeholder: str | None = None,
    ) -> None:
        if placeholder:
            for line_edit in [control, *control.findChildren(QLineEdit)]:
                if isinstance(line_edit, QLineEdit) and not line_edit.placeholderText():
                    line_edit.setPlaceholderText(placeholder)

        self._apply_help_text(control, help_text)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(control)

        hint = QLabel(help_text)
        hint.setObjectName("fieldHelpLabel")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666; font-size: 11px;")
        self._apply_help_text(hint, help_text)
        layout.addWidget(hint)

        form.addRow(label, container)

    def _apply_help_text(self, control: QWidget, help_text: str) -> None:
        control.setToolTip(help_text)
        for child in control.findChildren(QWidget):
            child.setToolTip(help_text)

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
            default_detail_type=str(
                getattr(app_config, "prelabel_default_detail_type", "asteroid") or "asteroid"
            ),
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
