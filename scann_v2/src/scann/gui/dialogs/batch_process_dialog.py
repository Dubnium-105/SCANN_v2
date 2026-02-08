"""批量图像处理对话框

功能:
- 批量降噪 (中值滤波, 高斯滤波)
- 伪平场校正
- 输出文件夹选择
- 输出位深选择 (16-bit / 32-bit)
- 进度条
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)


class BatchProcessDialog(QDialog):
    """批量处理对话框

    信号:
        process_started: 处理开始, 传递参数字典
    """

    process_started = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("批量图像处理")
        self.setMinimumSize(500, 400)

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── 输入 ──
        grp_input = QGroupBox("输入文件夹")
        input_form = QFormLayout(grp_input)

        input_layout = QHBoxLayout()
        self.edit_input_dir = QLineEdit()
        self.edit_input_dir.setPlaceholderText("选择包含 FITS 文件的文件夹")
        btn_input = QPushButton("浏览...")
        btn_input.clicked.connect(self._browse_input)
        input_layout.addWidget(self.edit_input_dir)
        input_layout.addWidget(btn_input)
        input_form.addRow("输入:", input_layout)

        layout.addWidget(grp_input)

        # ── 降噪 ──
        grp_denoise = QGroupBox("降噪")
        grp_denoise.setCheckable(True)
        grp_denoise.setChecked(True)
        denoise_form = QFormLayout(grp_denoise)

        self.combo_denoise_method = QComboBox()
        self.combo_denoise_method.addItems(["中值滤波", "高斯滤波", "双边滤波"])
        denoise_form.addRow("方法:", self.combo_denoise_method)

        self.spin_kernel_size = QSpinBox()
        self.spin_kernel_size.setRange(3, 15)
        self.spin_kernel_size.setSingleStep(2)
        self.spin_kernel_size.setValue(3)
        denoise_form.addRow("核大小:", self.spin_kernel_size)

        layout.addWidget(grp_denoise)
        self.grp_denoise = grp_denoise

        # ── 伪平场 ──
        grp_flat = QGroupBox("伪平场校正")
        grp_flat.setCheckable(True)
        grp_flat.setChecked(False)
        flat_form = QFormLayout(grp_flat)

        self.spin_flat_sigma = QDoubleSpinBox()
        self.spin_flat_sigma.setRange(10.0, 500.0)
        self.spin_flat_sigma.setDecimals(1)
        self.spin_flat_sigma.setValue(100.0)
        flat_form.addRow("高斯 σ:", self.spin_flat_sigma)

        layout.addWidget(grp_flat)
        self.grp_flat = grp_flat

        # ── 输出 ──
        grp_output = QGroupBox("输出设置")
        output_form = QFormLayout(grp_output)

        output_layout = QHBoxLayout()
        self.edit_output_dir = QLineEdit()
        self.edit_output_dir.setPlaceholderText("默认: 输入文件夹/processed/")
        btn_output = QPushButton("浏览...")
        btn_output.clicked.connect(self._browse_output)
        output_layout.addWidget(self.edit_output_dir)
        output_layout.addWidget(btn_output)
        output_form.addRow("输出:", output_layout)

        self.combo_bit_depth = QComboBox()
        self.combo_bit_depth.addItems(["16-bit (保持原样)", "32-bit float"])
        output_form.addRow("位深:", self.combo_bit_depth)

        self.chk_overwrite = QCheckBox("覆盖已有文件")
        output_form.addRow(self.chk_overwrite)

        layout.addWidget(grp_output)

        # ── 进度 ──
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.lbl_status = QLabel("")
        layout.addWidget(self.lbl_status)

        # ── 按钮 ──
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("▶ 开始处理")
        self.btn_start.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold; padding: 8px;"
        )
        self.btn_start.clicked.connect(self._on_start)

        self.btn_close = QPushButton("关闭")
        self.btn_close.clicked.connect(self.close)

        btn_layout.addWidget(self.btn_start)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_close)
        layout.addLayout(btn_layout)

    # ── 公共 API ──

    def update_progress(self, current: int, total: int, filename: str) -> None:
        """更新处理进度"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.lbl_status.setText(f"处理中: {filename} ({current}/{total})")

    def processing_finished(self) -> None:
        """处理完成"""
        self.btn_start.setEnabled(True)
        self.lbl_status.setText("✅ 处理完成!")

    # ── 事件 ──

    def _on_start(self) -> None:
        if not self.edit_input_dir.text():
            self.lbl_status.setText("⚠ 请选择输入文件夹")
            return

        self.btn_start.setEnabled(False)
        params = {
            "input_dir": self.edit_input_dir.text(),
            "output_dir": self.edit_output_dir.text(),
            "denoise": self.grp_denoise.isChecked(),
            "denoise_method": self.combo_denoise_method.currentText(),
            "kernel_size": self.spin_kernel_size.value(),
            "flat_field": self.grp_flat.isChecked(),
            "flat_sigma": self.spin_flat_sigma.value(),
            "bit_depth": self.combo_bit_depth.currentText(),
            "overwrite": self.chk_overwrite.isChecked(),
        }
        self.process_started.emit(params)

    def _browse_input(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输入文件夹")
        if path:
            self.edit_input_dir.setText(path)

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if path:
            self.edit_output_dir.setText(path)
