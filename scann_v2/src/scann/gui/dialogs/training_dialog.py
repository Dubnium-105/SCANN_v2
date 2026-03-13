"""AI 模型训练对话框

功能:
- 数据集目录配置
- 训练超参数设置
- 训练进度条
- 实时 loss 曲线显示 (简易文本模式)
- 提前停止
"""

from __future__ import annotations

import subprocess
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
        self._refresh_cuda_status()  # 初始化时检测CUDA状态

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── 数据集 ──
        grp_data = QGroupBox("数据集")
        data_form = QFormLayout(grp_data)

        self.combo_dataset_format = QComboBox()
        self.combo_dataset_format.addItem("v1 三联图分类", "v1")
        self.combo_dataset_format.addItem("v2 FITS 配对", "v2")
        self.combo_dataset_format.setToolTip(
            "选择训练数据集格式\n"
            "v1: 数据集目录下应包含 positive/negative\n"
            "v2: 数据集目录下应包含 new/old 与 annotations.json"
        )
        self.combo_dataset_format.currentIndexChanged.connect(self._update_dataset_dir_hint)
        data_form.addRow("数据集类型:", self.combo_dataset_format)

        dataset_layout = QHBoxLayout()
        self.edit_dataset_dir = QLineEdit()
        btn_dataset = QPushButton("浏览...")
        btn_dataset.clicked.connect(lambda: self._browse_dir(self.edit_dataset_dir))
        dataset_layout.addWidget(self.edit_dataset_dir)
        dataset_layout.addWidget(btn_dataset)
        data_form.addRow("数据集目录:", dataset_layout)

        self.lbl_dataset_hint = QLabel()
        self.lbl_dataset_hint.setWordWrap(True)
        self.lbl_dataset_hint.setStyleSheet("font-size: 11px; color: #666;")
        data_form.addRow("目录要求:", self.lbl_dataset_hint)
        self._update_dataset_dir_hint()

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
        self.combo_backbone.addItems(["ResNet18", "ResNet34", "ResNet50", "ViT_B_16"])
        hyper_form.addRow("骨干网络:", self.combo_backbone)

        self.combo_task_type = QComboBox()
        self.combo_task_type.addItem("classification", "classification")
        self.combo_task_type.addItem("detection (dense)", "detection")
        self.combo_task_type.setToolTip("选择训练任务类型：分类或全图 dense 检测")
        hyper_form.addRow("任务类型:", self.combo_task_type)

        # 设备选择
        device_layout = QHBoxLayout()
        self.combo_device = QComboBox()
        self.combo_device.addItem("Auto (优先CUDA)", "auto")
        self.combo_device.addItem("CUDA", "cuda:0")
        self.combo_device.addItem("CPU", "cpu")
        self.combo_device.setToolTip(
            "选择训练设备\n"
            "Auto: 有CUDA则使用cuda:0，否则cpu\n"
            "CUDA: 强制使用cuda:0（不可用则回退cpu）\n"
            "CPU: 强制使用cpu"
        )
        device_layout.addWidget(self.combo_device)
        
        # CUDA状态标签
        self.lbl_cuda_status = QLabel()
        self.lbl_cuda_status.setStyleSheet("font-size: 11px; color: #666;")
        self.lbl_cuda_status.setWordWrap(True)
        self.lbl_cuda_status.setMaximumWidth(200)
        device_layout.addWidget(self.lbl_cuda_status)
        
        # 刷新按钮
        self.btn_refresh_cuda = QPushButton("刷新")
        self.btn_refresh_cuda.setMaximumWidth(50)
        self.btn_refresh_cuda.clicked.connect(self._refresh_cuda_status)
        device_layout.addWidget(self.btn_refresh_cuda)
        
        hyper_form.addRow("训练设备:", device_layout)

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

        # 模型保存格式
        self.combo_save_format = QComboBox()
        self.combo_save_format.addItems([
            "v2_classifier (SCANN v2 推荐)",
            "v1_classifier (SCANN v1 兼容)",
        ])
        self.combo_save_format.setToolTip(
            "选择训练后模型的保存格式\n"
            "v2_classifier: 新版格式，带格式元数据\n"
            "v1_classifier: 兼容旧版的格式"
        )
        hyper_form.addRow("保存格式:", self.combo_save_format)

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
        if not self.edit_dataset_dir.text().strip():
            self.log_text.appendPlainText("⚠ 请先设置数据集目录")
            return

        self._is_training = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.log_text.clear()

        params = {
            "dataset_dir": self.edit_dataset_dir.text().strip(),
            "dataset_format": self.combo_dataset_format.currentData() or "v1",
            "val_split": self.spin_val_split.value(),
            "epochs": self.spin_epochs.value(),
            "batch_size": self.spin_batch.value(),
            "lr": self.spin_lr.value(),
            "optimizer": self.combo_optimizer.currentText(),
            "backbone": self.combo_backbone.currentText(),
            "task_type": self.combo_task_type.currentData() or "classification",
            "device": (self.combo_device.currentData() or "auto"),
            "augment": self.chk_augment.isChecked(),
            "early_stop": self.chk_early_stop.isChecked(),
            "patience": self.spin_patience.value(),
            "save_format": ["v2_classifier", "v1_classifier"][self.combo_save_format.currentIndex()],
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

    def _update_dataset_dir_hint(self) -> None:
        dataset_format = self.combo_dataset_format.currentData() or "v1"
        if dataset_format == "v2":
            self.edit_dataset_dir.setPlaceholderText("数据集目录下包含 new、old 和 annotations.json")
            self.lbl_dataset_hint.setText("v2: 自动读取 new/old 子目录，并结合 annotations.json 提取训练样本")
            return

        self.edit_dataset_dir.setPlaceholderText("数据集目录下包含 positive 和 negative")
        self.lbl_dataset_hint.setText("v1: 自动读取 positive/negative 子目录中的已分类样本")

    def _refresh_cuda_status(self) -> None:
        """刷新CUDA状态显示"""
        self.lbl_cuda_status.setText("检测中...")
        self.lbl_cuda_status.setStyleSheet("font-size: 11px; color: #666;")
        self.lbl_cuda_status.repaint()
        
        # 刷新UI
        from PyQt5.QtWidgets import QApplication
        QApplication.processEvents()
        
        # 直接调用检查
        self._check_cuda_availability()

    def _check_cuda_availability(self) -> None:
        """检查CUDA可用性并更新状态标签"""
        try:
            import torch
        except ImportError:
            self.lbl_cuda_status.setText("❌ PyTorch未安装")
            self.lbl_cuda_status.setStyleSheet("font-size: 11px; color: #f44336;")
            return
        
        if torch.cuda.is_available():
            # CUDA可用
            count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            cuda_version = torch.version.cuda
            
            msg = f"✅ CUDA可用\n"
            msg += f"  • 版本: {cuda_version}\n"
            msg += f"  • 设备数: {count}\n"
            msg += f"  • 当前: {device_name}"
            
            self.lbl_cuda_status.setText(msg)
            self.lbl_cuda_status.setStyleSheet("font-size: 11px; color: #4CAF50;")
            
            # 确保CUDA选项可用
            if self.combo_device.findData("cuda:0") == -1:
                self.combo_device.insertItem(1, "CUDA", "cuda:0")
        else:
            # CUDA不可用，显示可能的原因
            msg = "❌ CUDA不可用\n"
            msg += "可能原因:\n"
            
            # 检查是否是CPU版本PyTorch
            if not hasattr(torch, 'cuda'):
                msg += "  • PyTorch CPU版未支持CUDA"
            elif not torch.cuda.is_available():
                # 检查CUDA是否安装
                try:
                    import subprocess
                    result = subprocess.run(
                        ["nvidia-smi"], 
                        capture_output=True, 
                        text=True,
                        timeout=2,
                        shell=True
                    )
                    if result.returncode != 0:
                        msg += "  • NVIDIA驱动未安装或nvidia-smi不可用\n"
                        msg += "  • 需要安装NVIDIA显卡驱动"
                    else:
                        msg += "  • 驱动已安装\n"
                        msg += "  • 可能是PyTorch CPU版\n"
                        msg += "  • 检查命令: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
                except Exception:
                    msg += "  • 无法检测NVIDIA驱动\n"
                    msg += "  • 请确保安装了NVIDIA显卡和驱动"
            
            self.lbl_cuda_status.setText(msg)
            self.lbl_cuda_status.setStyleSheet("font-size: 11px; color: #f44336;")
