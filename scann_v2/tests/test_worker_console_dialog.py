from __future__ import annotations

from pathlib import Path

import pytest
from PyQt5.QtWidgets import QLabel

from scann.gui.dialogs.worker_console_dialog import WorkerConsoleDialog


@pytest.fixture
def dialog(qapp):
    return WorkerConsoleDialog()


def test_worker_console_dialog_has_two_tabs(dialog) -> None:
    assert dialog.tabs.count() == 2
    assert dialog.tabs.tabText(0) == "预标注 Worker"
    assert dialog.tabs.tabText(1) == "训练 Worker"


def test_worker_console_dialog_adds_help_for_all_editable_rows(dialog) -> None:
    fields = [
        dialog.edit_prelabel_server_url,
        dialog.edit_prelabel_token,
        dialog.edit_prelabel_worker_id,
        dialog.edit_prelabel_display_name,
        dialog.edit_prelabel_device_label,
        dialog.edit_prelabel_dataset_root,
        dialog.edit_prelabel_config_path,
        dialog.edit_prelabel_model_path,
        dialog.edit_prelabel_model_version,
        dialog.edit_prelabel_model_id,
        dialog.edit_prelabel_model_backbone,
        dialog.edit_prelabel_compute_device,
        dialog.edit_prelabel_idle_seconds,
        dialog.edit_prelabel_heartbeat_seconds,
        dialog.edit_prelabel_timeout_seconds,
        dialog.edit_training_server_url,
        dialog.edit_training_token,
        dialog.edit_training_worker_id,
        dialog.edit_training_display_name,
        dialog.edit_training_device_label,
        dialog.edit_training_dataset_root,
        dialog.edit_training_output_root,
        dialog.edit_training_task_types,
        dialog.edit_training_model_backbones,
        dialog.edit_training_device,
        dialog.edit_training_idle_seconds,
        dialog.edit_training_heartbeat_seconds,
        dialog.edit_training_timeout_seconds,
    ]

    for field in fields:
        assert field.toolTip()

    help_labels = [label for label in dialog.findChildren(QLabel) if label.objectName() == "fieldHelpLabel"]
    assert len(help_labels) == len(fields)
    assert any("/api/*" in label.text() for label in help_labels)
    assert any("\u53ef\u8bad\u7ec3\u7684 backbone" in label.text() for label in help_labels)


def test_build_prelabel_config_uses_ui_fields_and_config_defaults(dialog, config_file: Path) -> None:
    model_path = config_file.parent / "model-best.pth"
    model_path.write_bytes(b"checkpoint")

    dialog.edit_prelabel_server_url.setText("http://127.0.0.1:8000")
    dialog.edit_prelabel_token.setText("worker-secret")
    dialog.edit_prelabel_worker_id.setText("gpu-prelabel-1")
    dialog.edit_prelabel_display_name.setText("GPU Prelabel")
    dialog.edit_prelabel_device_label.setText("RTX-4090")
    dialog.edit_prelabel_dataset_root.setText(str(config_file.parent))
    dialog.edit_prelabel_config_path.setText(str(config_file))
    dialog.edit_prelabel_model_path.setText(str(model_path))
    dialog.edit_prelabel_model_version.setText("detector-v3")
    dialog.edit_prelabel_model_id.setText("detector-v3-run-001")
    dialog.edit_prelabel_model_backbone.setText("ViT_B_16")
    dialog.edit_prelabel_compute_device.setText("cuda")

    config = dialog._build_prelabel_config()

    assert config.server_url == "http://127.0.0.1:8000"
    assert config.worker_token == "worker-secret"
    assert config.worker_id == "gpu-prelabel-1"
    assert config.display_name == "GPU Prelabel"
    assert config.device_label == "RTX-4090"
    assert config.dataset_root == config_file.parent.resolve()
    assert config.detection.model_path == str(model_path)
    assert config.detection.model_version == "detector-v3"
    assert config.detection.model_id == "detector-v3-run-001"
    assert config.detection.model_backbone == "ViT_B_16"
    assert config.detection.compute_device == "cuda"
    assert config.detection.detection_params.thresh == 80
    assert config.detection.patch_size == 80


def test_build_training_config_parses_csv_fields(dialog, tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "output"
    dataset_root.mkdir()
    output_root.mkdir()

    dialog.edit_training_server_url.setText("http://127.0.0.1:8000")
    dialog.edit_training_token.setText("training-secret")
    dialog.edit_training_worker_id.setText("gpu-trainer-1")
    dialog.edit_training_display_name.setText("GPU Trainer")
    dialog.edit_training_device_label.setText("RTX-A6000")
    dialog.edit_training_dataset_root.setText(str(dataset_root))
    dialog.edit_training_output_root.setText(str(output_root))
    dialog.edit_training_task_types.setText("classification, detection")
    dialog.edit_training_model_backbones.setText("ViT_B_16, ResNet18")
    dialog.edit_training_device.setText("cuda")

    config = dialog._build_training_config()

    assert config.server_url == "http://127.0.0.1:8000"
    assert config.worker_token == "training-secret"
    assert config.worker_id == "gpu-trainer-1"
    assert config.display_name == "GPU Trainer"
    assert config.device_label == "RTX-A6000"
    assert config.execution.dataset_root == dataset_root.resolve()
    assert config.execution.output_root == output_root.resolve()
    assert config.execution.task_types == ["classification", "detection"]
    assert config.execution.model_backbones == ["ViT_B_16", "ResNet18"]
    assert config.execution.device == "cuda"
