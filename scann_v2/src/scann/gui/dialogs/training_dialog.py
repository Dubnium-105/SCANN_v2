"""AI 模型训练对话框

功能:
- 数据集路径配置
- 训练超参数设置
- 训练进度条
- 实时 loss 曲线显示 (简易文本模式)
- 提前停止
"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)


class TrainingDialog(QDialog):
    """AI 训练对话框

    信号:
        training_started: 训练开始
        training_stopped: 训练手动停止
    """

    training_started = pyqtSignal(dict)   # 超参数字典
    training_stopped = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI 模型训练")
        self.setMinimumSize(550, 600)

        self._is_training = False
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── 数据集 ──
        grp_data = QGroupBox("数据集")
        data_form = QFormLayout(grp_data)

        pos_layout = QHBoxLayout()
        self.edit_pos_dir = QLineEdit()
        self.edit_pos_dir.setPlaceholderText("正样本文件夹 (positive)")
        btn_pos = QPushButton("浏览...")
        btn_pos.clicked.connect(lambda: self._browse_dir(self.edit_pos_dir))
        pos_layout.addWidget(self.edit_pos_dir)
        pos_layout.addWidget(btn_pos)
        data_form.addRow("正样本:", pos_layout)

        neg_layout = QHBoxLayout()
        self.edit_neg_dir = QLineEdit()
        self.edit_neg_dir.setPlaceholderText("负样本文件夹 (negative)")
        btn_neg = QPushButton("浏览...")
        btn_neg.clicked.connect(lambda: self._browse_dir(self.edit_neg_dir))
        neg_layout.addWidget(self.edit_neg_dir)
        neg_layout.addWidget(btn_neg)
        data_form.addRow("负样本:", neg_layout)

        self.spin_val_split = QDoubleSpinBox()
        self.spin_val_split.setRange(0.05, 0.5)
        self.spin_val_split.setDecimals(2)
        self.spin_val_split.setSingleStep(0.05)
        self.spin_val_split.setValue(0.2)
        data_form.addRow("验证集比例:", self.spin_val_split)

        layout.addWidget(grp_data)

        # ── 超参数 ──
        grp_hyper = QGroupBox("超参数")
        hyper_form = QFormLayout(grp_hyper)

        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 1000)
        self.spin_epochs.setValue(50)
        hyper_form.addRow("Epochs:", self.spin_epochs)

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 512)
        self.spin_batch.setValue(32)
        hyper_form.addRow("Batch Size:", self.spin_batch)

        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setRange(0.00001, 1.0)
        self.spin_lr.setDecimals(5)
        self.spin_lr.setSingleStep(0.0001)
        self.spin_lr.setValue(0.001)
        hyper_form.addRow("Learning Rate:", self.spin_lr)

        self.combo_optimizer = QComboBox()
        self.combo_optimizer.addItems(["Adam", "AdamW", "SGD"])
        hyper_form.addRow("优化器:", self.combo_optimizer)

        self.combo_backbone = QComboBox()
        self.combo_backbone.addItems(["ResNet18", "ResNet34", "ResNet50"])
        hyper_form.addRow("骨干网络:", self.combo_backbone)

        self.chk_augment = QCheckBox("数据增强")
        self.chk_augment.setChecked(True)
        hyper_form.addRow(self.chk_augment)

        self.chk_early_stop = QCheckBox("提前停止 (patience)")
        self.chk_early_stop.setChecked(True)
        hyper_form.addRow(self.chk_early_stop)

        self.spin_patience = QSpinBox()
        self.spin_patience.setRange(1, 50)
        self.spin_patience.setValue(10)
        hyper_form.addRow("Patience:", self.spin_patience)

        layout.addWidget(grp_hyper)

        # ── 进度 ──
        grp_progress = QGroupBox("训练进度")
        prog_layout = QVBoxLayout(grp_progress)

        self.progress_bar = QProgressBar()
        prog_layout.addWidget(self.progress_bar)

        self.lbl_epoch_info = QLabel("Epoch: --/--  Loss: --  Val Loss: --")
        prog_layout.addWidget(self.lbl_epoch_info)

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        prog_layout.addWidget(self.log_text)

        layout.addWidget(grp_progress)

        # ── 按钮 ──
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("🚀 开始训练")
        self.btn_start.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;"
        )
        self.btn_start.clicked.connect(self._on_start)

        self.btn_stop = QPushButton("⏹ 停止")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)

        self.btn_close = QPushButton("关闭")
        self.btn_close.clicked.connect(self.close)

        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_close)
        layout.addLayout(btn_layout)

    # ── 公共 API ──

    def update_progress(self, epoch: int, total: int, loss: float, val_loss: float) -> None:
        """更新训练进度"""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(epoch)
        self.lbl_epoch_info.setText(
            f"Epoch: {epoch}/{total}  Loss: {loss:.4f}  Val Loss: {val_loss:.4f}"
        )
        self.log_text.appendPlainText(
            f"[Epoch {epoch:3d}] loss={loss:.4f}  val_loss={val_loss:.4f}"
        )

    def training_finished(self, model_path: str) -> None:
        """训练完成"""
        self._is_training = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.log_text.appendPlainText(f"\n✅ 训练完成! 模型已保存: {model_path}")

    # ── 事件 ──

    def _on_start(self) -> None:
        if not self.edit_pos_dir.text() or not self.edit_neg_dir.text():
            self.log_text.appendPlainText("⚠ 请先设置正负样本文件夹")
            return

        self._is_training = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.log_text.clear()

        params = {
            "pos_dir": self.edit_pos_dir.text(),
            "neg_dir": self.edit_neg_dir.text(),
            "val_split": self.spin_val_split.value(),
            "epochs": self.spin_epochs.value(),
            "batch_size": self.spin_batch.value(),
            "lr": self.spin_lr.value(),
            "optimizer": self.combo_optimizer.currentText(),
            "backbone": self.combo_backbone.currentText(),
            "augment": self.chk_augment.isChecked(),
            "early_stop": self.chk_early_stop.isChecked(),
            "patience": self.spin_patience.value(),
        }
        self.training_started.emit(params)

    def _on_stop(self) -> None:
        self._is_training = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.log_text.appendPlainText("⏹ 训练已手动停止")
        self.training_stopped.emit()

    def _browse_dir(self, line_edit: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if path:
            line_edit.setText(path)
