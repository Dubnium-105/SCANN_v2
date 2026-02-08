"""首选项设置对话框

多标签页:
1. 望远镜/天文台参数
2. 检测参数
3. AI 模型参数
4. 保存/路径
5. 高级选项
"""

from __future__ import annotations

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
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from scann.core.models import AppConfig as Config


class SettingsDialog(QDialog):
    """首选项设置对话框

    信号:
        settings_changed: 设置已更新
    """

    settings_changed = pyqtSignal()

    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("首选项设置")
        self.setMinimumSize(600, 500)
        self.config = config

        self._init_ui()
        self._load_from_config()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # ── Tab 1: 望远镜/天文台 ──
        self._init_observatory_tab()

        # ── Tab 2: 检测参数 ──
        self._init_detection_tab()

        # ── Tab 3: AI 模型 ──
        self._init_ai_tab()

        # ── Tab 4: 保存/路径 ──
        self._init_paths_tab()

        # ── Tab 5: 高级 ──
        self._init_advanced_tab()

        # ── 按钮 ──
        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )
        btn_box.accepted.connect(self._on_ok)
        btn_box.rejected.connect(self.reject)
        apply_btn = btn_box.button(QDialogButtonBox.Apply)
        apply_btn.clicked.connect(self._on_apply)
        layout.addWidget(btn_box)

    # ── 望远镜/天文台 ──

    def _init_observatory_tab(self) -> None:
        tab = QWidget()
        form = QFormLayout(tab)

        self.edit_obs_code = QLineEdit()
        self.edit_obs_code.setPlaceholderText("例: C42")
        form.addRow("天文台编号 (MPC):", self.edit_obs_code)

        self.edit_obs_name = QLineEdit()
        form.addRow("天文台名称:", self.edit_obs_name)

        self.spin_lat = QDoubleSpinBox()
        self.spin_lat.setRange(-90, 90)
        self.spin_lat.setDecimals(6)
        self.spin_lat.setSuffix(" °")
        form.addRow("纬度:", self.spin_lat)

        self.spin_lon = QDoubleSpinBox()
        self.spin_lon.setRange(-180, 180)
        self.spin_lon.setDecimals(6)
        self.spin_lon.setSuffix(" °")
        form.addRow("经度:", self.spin_lon)

        self.spin_alt = QDoubleSpinBox()
        self.spin_alt.setRange(0, 10000)
        self.spin_alt.setDecimals(1)
        self.spin_alt.setSuffix(" m")
        form.addRow("海拔:", self.spin_alt)

        self.edit_telescope = QLineEdit()
        self.edit_telescope.setPlaceholderText("例: 0.6m f/3.5 reflector")
        form.addRow("望远镜:", self.edit_telescope)

        self.spin_pixel_scale = QDoubleSpinBox()
        self.spin_pixel_scale.setRange(0.01, 100.0)
        self.spin_pixel_scale.setDecimals(3)
        self.spin_pixel_scale.setSuffix(' "/px')
        form.addRow("像素尺度:", self.spin_pixel_scale)

        self.tabs.addTab(tab, "🔭 望远镜/天文台")

    # ── 检测参数 ──

    def _init_detection_tab(self) -> None:
        tab = QWidget()
        form = QFormLayout(tab)

        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(1.0, 20.0)
        self.spin_sigma.setDecimals(1)
        form.addRow("检测 σ 阈值:", self.spin_sigma)

        self.spin_min_area = QSpinBox()
        self.spin_min_area.setRange(1, 1000)
        form.addRow("最小面积 (px):", self.spin_min_area)

        self.spin_max_area = QSpinBox()
        self.spin_max_area.setRange(1, 100000)
        form.addRow("最大面积 (px):", self.spin_max_area)

        self.spin_nms_radius = QDoubleSpinBox()
        self.spin_nms_radius.setRange(0, 100)
        self.spin_nms_radius.setDecimals(1)
        self.spin_nms_radius.setSuffix(" px")
        form.addRow("NMS 半径:", self.spin_nms_radius)

        self.chk_exclude_edge = QCheckBox("排除边缘区域")
        form.addRow(self.chk_exclude_edge)

        self.spin_edge_margin = QSpinBox()
        self.spin_edge_margin.setRange(0, 500)
        self.spin_edge_margin.setSuffix(" px")
        form.addRow("边缘宽度:", self.spin_edge_margin)

        self.tabs.addTab(tab, "🔍 检测参数")

    # ── AI 模型 ──

    def _init_ai_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 模型路径
        grp_model = QGroupBox("模型文件")
        model_form = QFormLayout(grp_model)

        model_path_layout = QHBoxLayout()
        self.edit_model_path = QLineEdit()
        self.edit_model_path.setReadOnly(True)
        self.btn_browse_model = QPushButton("浏览...")
        self.btn_browse_model.clicked.connect(self._browse_model)
        model_path_layout.addWidget(self.edit_model_path)
        model_path_layout.addWidget(self.btn_browse_model)
        model_form.addRow("模型路径:", model_path_layout)

        self.spin_confidence = QDoubleSpinBox()
        self.spin_confidence.setRange(0.0, 1.0)
        self.spin_confidence.setDecimals(2)
        self.spin_confidence.setSingleStep(0.05)
        model_form.addRow("置信度阈值:", self.spin_confidence)

        self.spin_patch_size = QSpinBox()
        self.spin_patch_size.setRange(16, 256)
        self.spin_patch_size.setSingleStep(16)
        self.spin_patch_size.setSuffix(" px")
        model_form.addRow("切片大小:", self.spin_patch_size)

        layout.addWidget(grp_model)

        # 推理参数
        grp_infer = QGroupBox("推理参数")
        infer_form = QFormLayout(grp_infer)

        self.spin_batch_size = QSpinBox()
        self.spin_batch_size.setRange(1, 512)
        infer_form.addRow("批量大小:", self.spin_batch_size)

        self.combo_device = QComboBox()
        self.combo_device.addItems(["auto", "cpu", "cuda"])
        infer_form.addRow("计算设备:", self.combo_device)

        layout.addWidget(grp_infer)
        layout.addStretch()

        self.tabs.addTab(tab, "🧠 AI 模型")

    # ── 保存/路径 ──

    def _init_paths_tab(self) -> None:
        tab = QWidget()
        form = QFormLayout(tab)

        # 默认保存路径
        save_layout = QHBoxLayout()
        self.edit_save_dir = QLineEdit()
        self.btn_browse_save = QPushButton("浏览...")
        self.btn_browse_save.clicked.connect(self._browse_save_dir)
        save_layout.addWidget(self.edit_save_dir)
        save_layout.addWidget(self.btn_browse_save)
        form.addRow("默认保存路径:", save_layout)

        # MPCORB 路径
        mpcorb_layout = QHBoxLayout()
        self.edit_mpcorb_path = QLineEdit()
        self.btn_browse_mpcorb = QPushButton("浏览...")
        self.btn_browse_mpcorb.clicked.connect(self._browse_mpcorb)
        mpcorb_layout.addWidget(self.edit_mpcorb_path)
        mpcorb_layout.addWidget(self.btn_browse_mpcorb)
        form.addRow("MPCORB 文件:", mpcorb_layout)

        # 数据库路径
        db_layout = QHBoxLayout()
        self.edit_db_path = QLineEdit()
        self.btn_browse_db = QPushButton("浏览...")
        db_layout.addWidget(self.edit_db_path)
        db_layout.addWidget(self.btn_browse_db)
        form.addRow("数据库路径:", db_layout)

        # 保存格式
        self.combo_save_format = QComboBox()
        self.combo_save_format.addItems(["FITS (16-bit)", "FITS (32-bit)", "PNG (8-bit)"])
        form.addRow("保存格式:", self.combo_save_format)

        self.tabs.addTab(tab, "📁 保存/路径")

    # ── 高级 ──

    def _init_advanced_tab(self) -> None:
        tab = QWidget()
        form = QFormLayout(tab)

        self.spin_max_threads = QSpinBox()
        self.spin_max_threads.setRange(1, 32)
        form.addRow("最大线程数:", self.spin_max_threads)

        self.chk_auto_save = QCheckBox("退出时自动保存标记")
        form.addRow(self.chk_auto_save)

        self.chk_auto_collapse = QCheckBox("窗口 < 1200px 时自动折叠侧边栏")
        self.chk_auto_collapse.setChecked(True)
        form.addRow(self.chk_auto_collapse)

        self.spin_recent_max = QSpinBox()
        self.spin_recent_max.setRange(5, 50)
        form.addRow("最近打开数量:", self.spin_recent_max)

        self.chk_confirm_close = QCheckBox("关闭前确认")
        self.chk_confirm_close.setChecked(True)
        form.addRow(self.chk_confirm_close)

        self.tabs.addTab(tab, "⚙ 高级")

    # ── 事件 ──

    def _load_from_config(self) -> None:
        """从 Config 对象加载当前设置"""
        cfg = self.config
        self.edit_obs_code.setText(getattr(cfg, "obs_code", ""))
        self.edit_obs_name.setText(getattr(cfg, "obs_name", ""))
        self.spin_sigma.setValue(getattr(cfg, "sigma_threshold", 5.0))
        self.spin_min_area.setValue(getattr(cfg, "min_area", 3))
        self.spin_confidence.setValue(getattr(cfg, "ai_confidence", 0.5))

    def _save_to_config(self) -> None:
        """将 UI 设置写回 Config"""
        cfg = self.config
        cfg.obs_code = self.edit_obs_code.text()
        cfg.obs_name = self.edit_obs_name.text()
        cfg.sigma_threshold = self.spin_sigma.value()
        cfg.min_area = self.spin_min_area.value()
        cfg.ai_confidence = self.spin_confidence.value()
        cfg.save()

    def _on_ok(self) -> None:
        self._save_to_config()
        self.settings_changed.emit()
        self.accept()

    def _on_apply(self) -> None:
        self._save_to_config()
        self.settings_changed.emit()

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "PyTorch Model (*.pth *.pt)"
        )
        if path:
            self.edit_model_path.setText(path)

    def _browse_save_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if path:
            self.edit_save_dir.setText(path)

    def _browse_mpcorb(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 MPCORB 文件", "", "DAT Files (*.dat);;All (*)"
        )
        if path:
            self.edit_mpcorb_path.setText(path)
