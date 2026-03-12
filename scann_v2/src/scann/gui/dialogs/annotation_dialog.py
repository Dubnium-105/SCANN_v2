"""标注对话框 — 双模式标注工具

支持 v1 三联图分类 和 v2 FITS 全图检测两种模式，
通过 AnnotationBackend 策略模式实现数据格式解耦。

触发: 菜单 AI > 标注工具 或 Ctrl+L
模式: 非模态 (不阻断主窗口操作)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtWidgets import (
    QAction,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QShortcut,
    QSlider,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from scann.core.annotation_backend import AnnotationBackend
from scann.core.triplet_backend import TripletAnnotationBackend
from scann.core.fits_annotation_backend import FitsAnnotationBackend
from scann.core.annotation_models import (
    AnnotationLabel,
    DETAIL_TYPE_DISPLAY,
    DETAIL_TYPE_TO_LABEL,
    SHORTCUT_TO_DETAIL_TYPE,
    AnnotationSample,
    BBox,
)
from scann.core.candidate_detector import DetectionParams
from scann.core.models import AppConfig
from scann.gui.widgets.annotation_list import AnnotationListWidget
from scann.gui.widgets.annotation_stats import AnnotationStatsPanel
from scann.gui.widgets.annotation_viewer import AnnotationViewer
from scann.gui.widgets.draw_toolbar import DrawToolBar
from scann.gui.widgets.triplet_preview import TripletPreviewPanel
from scann.gui.widgets.histogram_panel import HistogramPanel
from scann.gui.widgets.overlay_label import OverlayLabel
from scann.core.image_processor import histogram_stretch
from scann.services.detection_pipeline import DetectionPipeline


class AnnotationDialog(QDialog):
    """标注工具对话框

    双模式:
    - v1: 三联图快速分类 (TripletAnnotationBackend)
    - v2: FITS 全图检测标注 (FitsAnnotationBackend)
    """

    def __init__(self, parent: QWidget | None = None, config: AppConfig | None = None):
        super().__init__(parent)
        self.setWindowTitle("标注工具")
        self.setMinimumSize(900, 650)
        self.setModal(False)
        # 启用窗口最小化/最大化按钮
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)

        # 配置引用
        self._config: AppConfig = config if config is not None else AppConfig()

        # 状态
        self._current_mode: str = "v1"
        self._backend: Optional[AnnotationBackend] = None
        self._samples: list[AnnotationSample] = []
        self._current_index: int = 0
        self._sample_count: int = 0
        self._auto_advance: bool = self._config.ann_auto_advance
        self._dataset_path: str = ""
        self._last_detail_type: Optional[str] = "asteroid"  # 上次标注类型，供新框沿用

        # v2: 新旧图和直方图拉伸状态
        self._new_image_data: Optional[np.ndarray] = None
        self._old_image_data: Optional[np.ndarray] = None
        self._current_view: str = "new"  # "new" | "old"
        self._histogram_panel: Optional[HistogramPanel] = None
        self._current_sample: Optional[AnnotationSample] = None  # 当前样本用于获取文件信息

        # 快捷键→标签按钮映射
        self._label_buttons: dict[str, QPushButton] = {}

        self._init_ui()
        self._init_shortcuts()
        self._connect_signals()

        # 从配置恢复 UI 状态
        self._restore_from_config()

    # ─── UI 初始化 ───

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # ── 顶栏: 模式选择 + 数据集路径 ──
        top_bar = QHBoxLayout()

        top_bar.addWidget(QLabel("模式:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["v1 三联图分类", "v2 FITS全图检测"])
        self._mode_combo.setFixedWidth(160)
        top_bar.addWidget(self._mode_combo)

        top_bar.addWidget(QLabel("数据集:"))
        self._path_label = QLabel("(未加载)")
        self._path_label.setStyleSheet("color: #808080;")
        top_bar.addWidget(self._path_label, stretch=1)

        self._btn_browse = QPushButton("浏览...")
        self._btn_browse.setFixedWidth(70)
        top_bar.addWidget(self._btn_browse)

        # 目录格式帮助
        self._btn_help = QPushButton("?")
        self._btn_help.setFixedSize(20, 20)
        self._btn_help.setStyleSheet("""
            QPushButton {
                background: #444444; color: #D4D4D4;
                border-radius: 10px; font-size: 14px; font-weight: bold;
                padding: 0;
            }
            QPushButton:hover { background: #555555; }
        """)
        self._btn_help.setToolTip("查看目录格式要求")
        self._btn_help.clicked.connect(self._on_show_dir_help)
        top_bar.addWidget(self._btn_help)

        main_layout.addLayout(top_bar)

        # ── 主内容区: 可拆分面板 ──
        self._splitter = QSplitter(Qt.Horizontal)

        # 左侧: 图像预览区域
        viewer_panel = QWidget()
        viewer_layout = QVBoxLayout(viewer_panel)
        viewer_layout.setContentsMargins(0, 0, 0, 0)

        # v1: 三联图预览
        self._triplet_preview = TripletPreviewPanel()
        viewer_layout.addWidget(self._triplet_preview)

        # v2: 标注图像查看器
        self._annotation_viewer = AnnotationViewer()
        viewer_layout.addWidget(self._annotation_viewer)

        # v2: 绘制工具栏
        self._draw_toolbar = DrawToolBar()
        viewer_layout.addWidget(self._draw_toolbar)

        # v2: 新旧图切换和直方图控制栏
        self._viewer_ctrl_bar = self._create_viewer_control_bar()
        viewer_layout.addWidget(self._viewer_ctrl_bar)

        self._splitter.addWidget(viewer_panel)

        # 右侧: 侧边面板
        side_panel = QWidget()
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(4)

        # v2: 标注列表
        self._annotation_list = AnnotationListWidget()
        side_layout.addWidget(self._annotation_list)

        # 通用: 统计面板
        self._stats_panel = AnnotationStatsPanel()
        side_layout.addWidget(self._stats_panel)

        # v2: 边框粗细调节
        self._bbox_width_slider = self._create_bbox_width_slider()
        side_layout.addWidget(self._bbox_width_slider)

        self._splitter.addWidget(side_panel)
        self._splitter.setStretchFactor(0, 5)
        self._splitter.setStretchFactor(1, 1)

        main_layout.addWidget(self._splitter, stretch=3)

        # ── 快速标签面板 ──
        label_panel = self._create_label_panel()
        main_layout.addWidget(label_panel, stretch=0)

        # ── 操作栏 ──
        ops_bar = self._create_ops_bar()
        main_layout.addLayout(ops_bar)

        # ── 筛选栏 ──
        filter_bar = self._create_filter_bar()
        main_layout.addLayout(filter_bar)

        # ── 底栏 ──
        bottom_bar = QHBoxLayout()
        self._btn_export = QPushButton("导出数据集...")
        self._btn_ai_prelabel = QPushButton("批量AI预标注...")
        self._btn_close = QPushButton("关闭")
        self._btn_close.setFixedWidth(80)

        bottom_bar.addWidget(self._btn_export)
        bottom_bar.addWidget(self._btn_ai_prelabel)
        bottom_bar.addStretch()
        bottom_bar.addWidget(self._btn_close)
        main_layout.addLayout(bottom_bar)

        # 样式
        self.setStyleSheet("""
            QDialog {
                background: #1E1E1E;
                color: #D4D4D4;
            }
            QLabel { color: #D4D4D4; }
            QComboBox {
                background: #333333;
                color: #D4D4D4;
                border: 1px solid #3C3C3C;
                padding: 2px 4px;
            }
            QPushButton {
                background: #333333;
                color: #D4D4D4;
                border: 1px solid #3C3C3C;
                border-radius: 3px;
                padding: 4px 12px;
                min-height: 28px;
            }
            QPushButton:hover { background: #444444; }
            QPushButton:pressed { background: #555555; }
        """)

    def _create_label_panel(self) -> QWidget:
        """创建快速标签面板 (Y1-Y3, N1-N5)"""
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # A. 真类
        real_group = QGroupBox("A. 真")
        real_group.setStyleSheet("""
            QGroupBox {
                color: #4CAF50; font-weight: bold; font-size: 12px;
                border: 1px solid #4CAF50; border-radius: 4px;
                margin-top: 8px; padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 4px;
            }
        """)
        real_layout = QHBoxLayout(real_group)
        real_layout.setSpacing(4)

        real_buttons = {
            "Y1": ("小行星 ★  Y1", "asteroid"),
            "Y2": ("超新星 💥  Y2", "supernova"),
            "Y3": ("变星 ✦  Y3", "variable_star"),
        }
        for key, (text, _detail) in real_buttons.items():
            btn = QPushButton(text)
            btn.setStyleSheet("""
                QPushButton {
                    background: #2E7D32; color: white;
                    border: none; border-radius: 3px;
                    font-size: 11px; min-height: 22px; padding: 2px 8px;
                }
                QPushButton:hover { background: #388E3C; }
            """)
            btn.clicked.connect(lambda checked, k=key: self._on_label_button(k))
            self._label_buttons[key] = btn
            real_layout.addWidget(btn)

        layout.addWidget(real_group)

        # B. 假类
        bogus_group = QGroupBox("B. 假")
        bogus_group.setStyleSheet("""
            QGroupBox {
                color: #F44336; font-weight: bold; font-size: 12px;
                border: 1px solid #F44336; border-radius: 4px;
                margin-top: 8px; padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 4px;
            }
        """)
        bogus_layout = QHBoxLayout(bogus_group)
        bogus_layout.setSpacing(4)

        bogus_buttons = {
            "N1": ("卫星线 🛰️  N1", "satellite_trail"),
            "N2": ("噪点 ⚡  N2", "noise"),
            "N3": ("星芒 ✨  N3", "diffraction_spike"),
            "N4": ("CMOS结霜 ❄️  N4", "cmos_condensation"),
            "N5": ("有对应 🔀  N5", "corresponding"),
        }
        for key, (text, _detail) in bogus_buttons.items():
            btn = QPushButton(text)
            btn.setStyleSheet("""
                QPushButton {
                    background: #C62828; color: white;
                    border: none; border-radius: 3px;
                    font-size: 11px; min-height: 22px; padding: 2px 8px;
                }
                QPushButton:hover { background: #D32F2F; }
            """)
            btn.clicked.connect(lambda checked, k=key: self._on_label_button(k))
            self._label_buttons[key] = btn
            bogus_layout.addWidget(btn)

        layout.addWidget(bogus_group)

        return panel

    def _create_bbox_width_slider(self) -> QWidget:
        """创建边框粗细调节滑块"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 8, 4, 4)
        layout.setSpacing(4)

        label = QLabel("边框粗细")
        label.setStyleSheet("color: #D4D4D4; font-size: 11px;")
        layout.addWidget(label)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(1, 5)
        slider.setValue(2)
        slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px;
                background: #333333;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #2196F3;
                width: 14px; height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QSlider::handle:hover {
                background: #1976D2;
            }
        """)
        slider.valueChanged.connect(self._on_bbox_width_changed)
        layout.addWidget(slider)

        return panel

    def _on_bbox_width_changed(self, width: int) -> None:
        """边框粗细变更处理"""
        self._annotation_viewer.set_bbox_width(width)

    def _create_viewer_control_bar(self) -> QWidget:
        """创建图像查看器控制栏（新旧图切换+直方图拉伸）"""
        from PyQt5.QtWidgets import QLabel, QDockWidget

        panel = QWidget()
        panel.setFixedHeight(36)
        panel.setStyleSheet("background-color: #252526; border-top: 1px solid #3C3C3C;")
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        # 新/旧图切换按钮
        self._btn_show_new = QPushButton("[1] 新图")
        self._btn_show_old = QPushButton("[2] 旧图")
        self._btn_show_new.setCheckable(True)
        self._btn_show_old.setCheckable(True)
        self._btn_show_new.setChecked(True)
        layout.addWidget(self._btn_show_new)
        layout.addWidget(self._btn_show_old)

        # 分隔
        sep = QLabel("|")
        sep.setStyleSheet("color: #3C3C3C;")
        layout.addWidget(sep)

        # 反色按钮
        self._btn_invert = QPushButton("🔄 反色 (I)")
        self._btn_invert.setCheckable(True)
        layout.addWidget(self._btn_invert)

        # 直方图拉伸按钮
        self._btn_histogram = QPushButton("📊 拉伸")
        layout.addWidget(self._btn_histogram)

        # 状态标签（新图/旧图）
        self._overlay_state = OverlayLabel("NEW", parent=panel)
        self._overlay_state.set_state("new")
        layout.addWidget(self._overlay_state)

        layout.addStretch()

        # 连接信号
        self._btn_show_new.clicked.connect(self._on_show_new)
        self._btn_show_old.clicked.connect(self._on_show_old)
        self._btn_invert.clicked.connect(self._on_invert_toggle)
        self._btn_histogram.clicked.connect(self._on_toggle_histogram)

        # 初始化直方图面板
        self._init_histogram_dock()

        return panel

    # ─── 直方图面板 ───

    def _init_histogram_dock(self) -> None:
        """初始化直方图拉伸面板"""
        # 注：由于 AnnotationDialog 是 QDialog，不能直接使用 addDockWidget
        # 这里我们使用独立窗口模式显示直方图面板
        self._histogram_panel = HistogramPanel(self)
        self._histogram_panel.setWindowFlags(
            self._histogram_panel.windowFlags() | Qt.Tool
        )
        self._histogram_panel.setVisible(False)
        self._histogram_panel.stretch_changed.connect(self._on_stretch_changed)

    def _on_toggle_histogram(self) -> None:
        """切换直方图面板显示"""
        if self._histogram_panel is None:
            return
        visible = not self._histogram_panel.isVisible()
        if visible:
            self._histogram_panel.show()
            self._histogram_panel.move(
                self.x() + self.width() - self._histogram_panel.width() - 20,
                self.y() + 100
            )
        else:
            self._histogram_panel.hide()

    def _on_show_new(self) -> None:
        """显示新图"""
        self._btn_show_new.setChecked(True)
        self._btn_show_old.setChecked(False)
        self._current_view = "new"
        self._show_image("new")

    def _on_show_old(self) -> None:
        """显示旧图"""
        self._btn_show_new.setChecked(False)
        self._btn_show_old.setChecked(True)
        self._current_view = "old"
        self._show_image("old")

    def _show_image(self, which: str) -> None:
        """统一的图像显示逻辑"""
        if which == "new":
            self._current_view = "new"
        else:
            self._current_view = "old"

        self._refresh_current_image()

    def _on_invert_toggle(self) -> None:
        """切换反色"""
        self._annotation_viewer.toggle_invert()
        self._btn_invert.setChecked(self._annotation_viewer._inverted)

    def _on_stretch_changed(self, black: float, white: float) -> None:
        """直方图拉伸参数变化"""
        # 确定当前显示的图像
        data = self._new_image_data if self._current_view == "new" else self._old_image_data
        if data is None:
            return

        # 使用 histogram_stretch 执行线性拉伸
        stretched = histogram_stretch(data, black_point=black, white_point=white)
        # 使用新的 set_display_data 方法直接显示拉伸后的数据
        self._annotation_viewer.set_display_data(stretched)

    def _create_ops_bar(self) -> QHBoxLayout:
        """创建操作栏"""
        layout = QHBoxLayout()

        self._btn_skip = QPushButton("⏭ 跳过 (S)")
        self._btn_prev = QPushButton("◀ 上一个 (Z)")
        self._btn_next = QPushButton("▶ 下一个 (X)")
        self._btn_undo = QPushButton("↩ 撤销")
        self._btn_redo = QPushButton("↪ 重做")

        self._chk_auto_advance = QCheckBox("标注后自动下一个")
        self._chk_auto_advance.setChecked(True)
        self._chk_auto_advance.setStyleSheet("color: #D4D4D4;")

        layout.addWidget(self._btn_skip)
        layout.addWidget(self._btn_prev)
        layout.addWidget(self._btn_next)
        layout.addSpacing(16)
        layout.addWidget(self._btn_undo)
        layout.addWidget(self._btn_redo)
        layout.addStretch()
        layout.addWidget(self._chk_auto_advance)

        return layout

    def _create_filter_bar(self) -> QHBoxLayout:
        """创建筛选栏"""
        layout = QHBoxLayout()

        layout.addWidget(QLabel("筛选:"))
        self._filter_all = QRadioButton("全部")
        self._filter_all.setChecked(True)
        self._filter_unlabeled = QRadioButton("未标注")
        self._filter_real = QRadioButton("A.真")
        self._filter_bogus = QRadioButton("B.假")

        for rb in (self._filter_all, self._filter_unlabeled,
                   self._filter_real, self._filter_bogus):
            rb.setStyleSheet("color: #D4D4D4;")
            layout.addWidget(rb)

        layout.addStretch()

        layout.addWidget(QLabel("排序:"))
        self._sort_combo = QComboBox()
        self._sort_combo.addItems(["默认", "AI置信度", "文件名"])
        self._sort_combo.setFixedWidth(100)
        layout.addWidget(self._sort_combo)

        return layout

    # ─── 快捷键 ───

    def _init_shortcuts(self) -> None:
        """绑定标注快捷键"""
        shortcuts = {
            "S": self.skip_current,
            "Z": self.go_previous,
            "X": self.go_next,
            "Y1": lambda: self._on_label_button("Y1"),
            "Y2": lambda: self._on_label_button("Y2"),
            "Y3": lambda: self._on_label_button("Y3"),
            "N1": lambda: self._on_label_button("N1"),
            "N2": lambda: self._on_label_button("N2"),
            "N3": lambda: self._on_label_button("N3"),
            "N4": lambda: self._on_label_button("N4"),
            "N5": lambda: self._on_label_button("N5"),
            "1": self._on_show_new,
            "2": self._on_show_old,
            "I": self._on_invert_toggle,
        }
        for key, func in shortcuts.items():
            action = QAction(self)
            action.setShortcut(key)
            action.setShortcutContext(Qt.WindowShortcut)
            action.triggered.connect(func)
            self.addAction(action)

        # Ctrl+Z / Ctrl+Y
        undo_action = QAction(self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.setShortcutContext(Qt.WindowShortcut)
        undo_action.triggered.connect(self.undo)
        self.addAction(undo_action)

        redo_action = QAction(self)
        redo_action.setShortcut("Ctrl+Y")
        redo_action.setShortcutContext(Qt.WindowShortcut)
        redo_action.triggered.connect(self.redo)
        self.addAction(redo_action)

        # Ctrl+S 保存
        save_action = QAction(self)
        save_action.setShortcut("Ctrl+S")
        save_action.setShortcutContext(Qt.WindowShortcut)
        save_action.triggered.connect(self._save_annotations)
        self.addAction(save_action)

    # ─── 键盘事件 ───

    def keyPressEvent(self, event) -> None:
        """处理键盘事件 - 上下键在标注列表中切换"""
        if self._current_mode == "v2":
            if event.key() == Qt.Key_Up:
                self._select_prev_bbox()
                event.accept()
                return
            elif event.key() == Qt.Key_Down:
                self._select_next_bbox()
                event.accept()
                return
        super().keyPressEvent(event)

    # ─── 信号连接 ───

    def _connect_signals(self) -> None:
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._btn_browse.clicked.connect(self._on_browse)
        self._btn_skip.clicked.connect(self.skip_current)
        self._btn_prev.clicked.connect(self.go_previous)
        self._btn_next.clicked.connect(self.go_next)
        self._btn_undo.clicked.connect(self.undo)
        self._btn_redo.clicked.connect(self.redo)
        self._btn_close.clicked.connect(self.close)
        self._chk_auto_advance.toggled.connect(self._on_auto_advance_changed)
        self._btn_export.clicked.connect(self._on_export)
        self._btn_ai_prelabel.clicked.connect(self._on_ai_prelabel)

        # 绘制工具栏 → 标注查看器
        self._draw_toolbar.tool_changed.connect(self._annotation_viewer.set_tool)

        # 标注查看器信号
        self._annotation_viewer.box_drawn.connect(self._on_box_drawn)
        self._annotation_viewer.box_selected.connect(self._on_bbox_selected)
        self._annotation_viewer.point_clicked.connect(self._on_point_clicked)

        # 标注列表
        self._annotation_list.bbox_selected.connect(
            self._annotation_viewer.select_bbox
        )
        self._annotation_list.bbox_delete_requested.connect(self._on_delete_bbox)

        # 筛选
        self._filter_all.toggled.connect(lambda: self.set_filter("all"))
        self._filter_unlabeled.toggled.connect(lambda: self.set_filter("unlabeled"))
        self._filter_real.toggled.connect(lambda: self.set_filter("real"))
        self._filter_bogus.toggled.connect(lambda: self.set_filter("bogus"))

    # ─── 模式切换 ───

    @property
    def current_mode(self) -> str:
        return self._current_mode

    def set_mode(self, mode: str) -> None:
        """切换标注模式

        Args:
            mode: "v1" 或 "v2"
        """
        self._current_mode = mode

        is_v1 = mode == "v1"
        is_v2 = mode == "v2"

        # v1 组件
        self._triplet_preview.setVisible(is_v1)

        # v2 组件
        self._annotation_viewer.setVisible(is_v2)
        self._draw_toolbar.setVisible(is_v2)
        self._annotation_list.setVisible(is_v2)

        # 更新下拉框 (不触发信号)
        self._mode_combo.blockSignals(True)
        self._mode_combo.setCurrentIndex(0 if is_v1 else 1)
        self._mode_combo.blockSignals(False)

        # 创建对应后端
        if is_v1:
            self._backend = TripletAnnotationBackend()
        else:
            self._backend = FitsAnnotationBackend()

    def _on_mode_changed(self, index: int) -> None:
        self.set_mode("v1" if index == 0 else "v2")

    # ─── 数据集加载 ───

    def load_dataset(self, path: str) -> None:
        """加载标注数据集"""
        if self._backend is None:
            return

        self._dataset_path = path
        self._samples = self._backend.load_samples(path)
        self._sample_count = len(self._samples)
        self._current_index = 0

        self._path_label.setText(path)
        self._update_display()
        self._update_stats()

    def _on_browse(self) -> None:
        """浏览选择数据集目录"""
        path = QFileDialog.getExistingDirectory(self, "选择数据集目录")
        if path:
            self.load_dataset(path)

    # ─── 标注操作 ───

    def mark_current(self, label: str, detail_type: Optional[str] = None) -> None:
        """标注当前样本"""
        if self._backend is None or not self._samples:
            return
        if self._current_index >= len(self._samples):
            return

        sample = self._samples[self._current_index]
        self._backend.save_annotation(
            sample.id,
            label,
            detail_type=detail_type,
        )

        self._update_stats()

        if self._auto_advance:
            self.go_next()

    def skip_current(self) -> None:
        """跳过当前样本"""
        self.go_next()

    def go_next(self) -> None:
        """前进到下一样本"""
        if self._current_index < self._sample_count - 1:
            self._current_index += 1
            self._update_display()

    def go_previous(self) -> None:
        """回退到上一样本"""
        if self._current_index > 0:
            self._current_index -= 1
            self._update_display()

    def undo(self) -> None:
        """撤销"""
        if self._backend:
            self._backend.undo()
            self._update_stats()
            self._update_display()

    def redo(self) -> None:
        """重做"""
        if self._backend:
            self._backend.redo()
            self._update_stats()
            self._update_display()

    # ─── 快速标签按钮处理 ───

    def _on_label_button(self, shortcut_key: str) -> None:
        """处理标签按钮点击"""
        detail_type = SHORTCUT_TO_DETAIL_TYPE.get(shortcut_key)
        if detail_type is None:
            return
        label = DETAIL_TYPE_TO_LABEL.get(detail_type)

        # 记录上次类型，供新框沿用
        if label is not None:
            self._last_detail_type = detail_type


        if self._current_mode == "v1":
            # v1: 直接分类标注
            if label is not None:
                self.mark_current(label, detail_type=detail_type)
        elif self._current_mode == "v2":
            # v2: 为选中的 bbox 设置标签
            idx = self._annotation_viewer.selected_bbox_index
            if idx >= 0 and self._samples and self._current_index < len(self._samples):
                sample = self._samples[self._current_index]
                if idx < len(sample.bboxes):
                    # 通过后端保存修改，确保数据和持久化同步
                    sample.bboxes[idx].label = label
                    sample.bboxes[idx].detail_type = detail_type
                    # 保存当前选中的框索引
                    selected_idx = self._annotation_viewer.selected_bbox_index
                    # 重新加载以保持同步（因为修改的是同一个对象引用）
                    self._update_display()
                    self._update_stats()
                    # 恢复选中状态，确保可以连续选择标签
                    if selected_idx >= 0:
                        self._annotation_viewer.select_bbox(selected_idx)

    # ─── 绘制事件 ───

    def _on_box_drawn(self, bbox_drawn: 'BBox') -> None:
        """v2: 处理新绘制的边界框"""
        if self._backend is None or not self._samples:
            return
        if self._current_index >= len(self._samples):
            return

        bbox = BBox(
            x=bbox_drawn.x,
            y=bbox_drawn.y,
            width=bbox_drawn.width,
            height=bbox_drawn.height,
            label=None,  # 未标注
            detail_type=self._last_detail_type,  # 沿用上次类型
        )
        sample = self._samples[self._current_index]
        self._backend.save_annotation(sample.id, None, bbox=bbox)

        # 保存新添加框的索引，用于选中
        new_bbox_index = len(sample.bboxes) - 1

        self._update_display()
        self._update_stats()
        # 自动选中刚创建的框（最后一个）
        self._annotation_viewer.select_bbox(new_bbox_index)

    def _on_bbox_selected(self, index: int) -> None:
        """标注框被选中"""
        self._annotation_list.table.selectRow(index)

    def _on_delete_bbox(self, index: int) -> None:
        """删除标注框"""
        if not self._samples or self._current_index >= len(self._samples):
            return
        sample = self._samples[self._current_index]
        if 0 <= index < len(sample.bboxes):
            sample.bboxes.pop(index)
            self._update_display()
            self._update_stats()

    def _on_point_clicked(self, px: int, py: int) -> None:
        """点标模式: 标记点击位置"""
        if self._current_mode != "v2":
            return
        if not self._samples or self._current_index >= len(self._samples):
            return

        # 创建小尺寸的点标注框
        bbox = BBox(
            x=px - 2,
            y=py - 2,
            width=4,
            height=4,
            label=None,
            detail_type=self._last_detail_type,
        )
        sample = self._samples[self._current_index]
        self._backend.save_annotation(sample.id, None, bbox=bbox)

        # 保存新添加框的索引，用于选中
        new_bbox_index = len(sample.bboxes) - 1

        self._update_display()
        self._update_stats()
        # 自动选中刚创建的框（最后一个）
        self._annotation_viewer.select_bbox(new_bbox_index)

    def _select_next_bbox(self) -> None:
        """选中下一个标注框"""
        if self._current_mode != "v2":
            return
        if not self._samples or self._current_index >= len(self._samples):
            return
        sample = self._samples[self._current_index]
        current_idx = self._annotation_viewer.selected_bbox_index
        if current_idx < len(sample.bboxes) - 1:
            self._annotation_viewer.select_bbox(current_idx + 1)
        elif len(sample.bboxes) > 0 and current_idx < 0:
            self._annotation_viewer.select_bbox(0)

    def _select_prev_bbox(self) -> None:
        """选中上一个标注框"""
        if self._current_mode != "v2":
            return
        if not self._samples or self._current_index >= len(self._samples):
            return
        sample = self._samples[self._current_index]
        current_idx = self._annotation_viewer.selected_bbox_index
        if current_idx > 0:
            self._annotation_viewer.select_bbox(current_idx - 1)

    # ─── 筛选 ───

    def set_filter(self, filter_type: str) -> None:
        """设置样本筛选"""
        if self._backend is None or not self._dataset_path:
            return
        self._samples = self._backend.load_samples(self._dataset_path, filter=filter_type)
        self._sample_count = len(self._samples)
        self._current_index = 0
        self._update_display()
        self._update_stats()

    # ─── 显示更新 ───

    def _update_display(self) -> None:
        """根据当前索引更新图像和信息显示"""
        if not self._samples or self._current_index >= len(self._samples):
            return

        sample = self._samples[self._current_index]

        if self._current_mode == "v1":
            self._update_v1_display(sample)
        elif self._current_mode == "v2":
            self._update_v2_display(sample)

    def _update_v1_display(self, sample: AnnotationSample) -> None:
        """更新 v1 三联图显示"""
        try:
            img_data = self._backend.get_image_data(sample)
            self._triplet_preview.set_triplet_image(img_data)
        except Exception:
            pass

        self._triplet_preview.set_file_info(sample.display_name)

        if sample.ai_suggestion and sample.ai_confidence:
            self._triplet_preview.set_ai_suggestion(
                sample.ai_suggestion, sample.ai_confidence
            )

    def _update_v2_display(self, sample: AnnotationSample) -> None:
        """更新 v2 FITS 显示"""
        try:
            # 保存当前样本引用
            self._current_sample = sample

            # 加载新图数据
            img_data = self._backend.get_image_data(sample)
            self._new_image_data = img_data.copy() if img_data is not None else None

            # 加载旧图数据（如果后端支持）
            try:
                if self._backend is not None and hasattr(self._backend, 'get_image_data'):
                    old_data = self._backend.get_image_data(sample, image_type="old")
                    self._old_image_data = old_data.copy() if old_data is not None else None
                else:
                    self._old_image_data = None
            except Exception:
                self._old_image_data = None

            # 填充 metadata 供 _refresh_current_image 使用
            if hasattr(self._backend, '_image_paths'):
                paths = self._backend._image_paths.get(sample.id, {})
                sample.metadata["new_path"] = paths.get("new", "")
                sample.metadata["old_path"] = paths.get("old", "")
                sample.metadata["has_old"] = bool(paths.get("old", ""))

            # 根据当前视图显示对应的图像
            self._refresh_current_image()

            # 更新直方图数据（使用新图）
            if self._histogram_panel is not None and self._new_image_data is not None:
                self._histogram_panel.set_image_data(self._new_image_data)

        except Exception:
            pass

        # 更新标注框
        self._annotation_viewer.set_bboxes(sample.bboxes[:])
        self._annotation_list.set_bboxes(sample.bboxes[:])

    def _refresh_current_image(self) -> None:
        """根据当前视图刷新图像显示"""
        if self._current_sample is None:
            return
        sample = self._current_sample

        if self._current_view == "old" and self._old_image_data is not None:
            self._annotation_viewer.set_image(self._old_image_data, is_new=False, view="old")
            self._btn_show_new.setChecked(False)
            self._btn_show_old.setChecked(True)
            self._overlay_state.setText("OLD")
            self._overlay_state.set_state("old")
            # 显示旧图文件名和匹配状态
            old_path = sample.metadata.get("old_path", "")
            if old_path:
                filename = Path(old_path).name
                has_old = sample.metadata.get("has_old", False)
                self._overlay_state.set_file_name(filename, match_found=has_old)
        elif self._new_image_data is not None:
            self._annotation_viewer.set_image(self._new_image_data, is_new=True, view="new")
            self._btn_show_new.setChecked(True)
            self._btn_show_old.setChecked(False)
            self._overlay_state.setText("NEW")
            self._overlay_state.set_state("new")
            # 显示新图文件名和匹配状态（检查是否找到对应的旧图）
            new_path = sample.metadata.get("new_path", "") or sample.source_path
            if new_path:
                filename = Path(new_path).name
                has_old = sample.metadata.get("has_old", False)
                self._overlay_state.set_file_name(filename, match_found=has_old)

    def _update_stats(self) -> None:
        """更新统计面板"""
        if self._backend:
            stats = self._backend.get_statistics()
            self._stats_panel.update_stats(stats)

    # ─── 其他 ───

    def _on_auto_advance_changed(self, checked: bool) -> None:
        self._auto_advance = checked
        self._config.ann_auto_advance = checked

    # ─── 配置持久化 ───

    def _restore_from_config(self) -> None:
        """从 AppConfig 恢复标注工具的 UI 状态"""
        cfg = self._config

        # 窗口大小
        self.resize(cfg.ann_window_width, cfg.ann_window_height)

        # 模式
        self.set_mode(cfg.ann_mode)

        # 自动前进
        self._chk_auto_advance.setChecked(cfg.ann_auto_advance)
        self._auto_advance = cfg.ann_auto_advance

        # 筛选
        filter_map = {
            "all": self._filter_all,
            "unlabeled": self._filter_unlabeled,
            "real": self._filter_real,
            "bogus": self._filter_bogus,
        }
        rb = filter_map.get(cfg.ann_filter)
        if rb is not None:
            rb.setChecked(True)

        # 排序
        idx = self._sort_combo.findText(cfg.ann_sort)
        if idx >= 0:
            self._sort_combo.setCurrentIndex(idx)

        # 边框粗细
        # _create_bbox_width_slider 创建的是 QWidget, slider 是其子控件
        slider = self._bbox_width_slider.findChild(QSlider)
        if slider is not None:
            slider.setValue(cfg.ann_bbox_width)
            self._annotation_viewer.set_bbox_width(cfg.ann_bbox_width)

        # 反色
        if cfg.ann_invert:
            self._btn_invert.setChecked(True)
            self._annotation_viewer.toggle_invert()

        # 分割面板比例
        if cfg.ann_splitter_sizes:
            self._splitter.setSizes(cfg.ann_splitter_sizes)

        # 直方图拉伸参数
        if self._histogram_panel is not None:
            # 拉伸模式
            mode_idx = self._histogram_panel.combo_mode.findText(cfg.ann_stretch_mode)
            if mode_idx >= 0:
                self._histogram_panel.combo_mode.setCurrentIndex(mode_idx)
            # 黑白点 (需要在加载图片后才能生效，先记录到 spin)
            self._histogram_panel.spin_black.setValue(int(cfg.ann_stretch_black))
            self._histogram_panel.spin_white.setValue(int(cfg.ann_stretch_white))
            # 面板可见性
            if cfg.ann_histogram_visible:
                self._histogram_panel.show()

        # 数据集路径 (最后加载，因为它会触发 load_samples)
        if cfg.ann_dataset_path:
            if Path(cfg.ann_dataset_path).exists():
                self.load_dataset(cfg.ann_dataset_path)
            else:
                # 路径不存在，清除旧配置，让用户重新选择
                import logging
                logging.getLogger(__name__).warning(
                    f"数据集路径不存在，已清除: {cfg.ann_dataset_path}"
                )
                cfg.ann_dataset_path = ""

    def _save_to_config(self) -> None:
        """将当前 UI 状态写回 AppConfig"""
        cfg = self._config

        # 模式
        cfg.ann_mode = self._current_mode

        # 数据集路径
        cfg.ann_dataset_path = self._dataset_path

        # 自动前进
        cfg.ann_auto_advance = self._auto_advance

        # 筛选
        if self._filter_all.isChecked():
            cfg.ann_filter = "all"
        elif self._filter_unlabeled.isChecked():
            cfg.ann_filter = "unlabeled"
        elif self._filter_real.isChecked():
            cfg.ann_filter = "real"
        elif self._filter_bogus.isChecked():
            cfg.ann_filter = "bogus"

        # 排序
        cfg.ann_sort = self._sort_combo.currentText()

        # 边框粗细
        slider = self._bbox_width_slider.findChild(QSlider)
        if slider is not None:
            cfg.ann_bbox_width = slider.value()

        # 反色
        cfg.ann_invert = self._btn_invert.isChecked()

        # 分割面板比例
        cfg.ann_splitter_sizes = self._splitter.sizes()

        # 直方图拉伸参数
        if self._histogram_panel is not None:
            cfg.ann_stretch_black = self._histogram_panel.black_point
            cfg.ann_stretch_white = self._histogram_panel.white_point
            cfg.ann_stretch_mode = self._histogram_panel.combo_mode.currentText()
            cfg.ann_histogram_visible = self._histogram_panel.isVisible()

        # 窗口大小
        cfg.ann_window_width = self.width()
        cfg.ann_window_height = self.height()

    def closeEvent(self, event) -> None:
        """关闭时保存状态到配置"""
        self._save_to_config()
        super().closeEvent(event)

    def _save_annotations(self) -> None:
        """保存标注 (v2 FITS 模式自动持久化，此处为显式保存)"""
        pass  # FitsAnnotationBackend 自动持久化到 JSON

    def _on_ai_prelabel(self) -> None:
        """批量运行 v2 AI 预标注。"""
        if self._current_mode != "v2":
            self._show_status_message("AI预标注仅支持 v2 FITS 模式", level="WARNING")
            return
        if not isinstance(self._backend, FitsAnnotationBackend) or not self._samples:
            self._show_status_message("请先加载 v2 标注数据集", level="WARNING")
            return

        inference_engine = self._resolve_inference_engine()
        if inference_engine is None or not getattr(inference_engine, "is_ready", False):
            self._show_status_message("请先在主窗口加载可用的 AI 模型", level="WARNING")
            return

        processed_count, bbox_count = self._apply_batch_ai_prelabel(inference_engine)
        self._update_display()
        self._update_stats()
        self._show_status_message(
            f"AI预标注完成: 处理 {processed_count} 张，生成 {bbox_count} 个候选框",
            timeout=5000,
        )

    def _apply_batch_ai_prelabel(self, inference_engine) -> tuple[int, int]:
        """对当前数据集中的未标注样本批量执行 AI 预标注。"""
        if not isinstance(self._backend, FitsAnnotationBackend):
            return 0, 0

        pipeline = DetectionPipeline(
            detection_params=self._build_detection_params(),
            inference_engine=inference_engine,
            patch_size=self._config.slice_size,
        )

        processed_count = 0
        bbox_count = 0
        for sample in self._samples:
            if sample.label is not None or sample.bboxes:
                continue

            new_data = self._backend.get_image_data(sample, image_type="new")
            try:
                old_data = self._backend.get_image_data(sample, image_type="old")
            except Exception:
                old_data = np.zeros_like(new_data)

            result = pipeline.process_pair(
                pair_name=sample.id,
                new_data=new_data,
                old_data=old_data,
                image_path=sample.source_path,
            )

            bboxes = [
                self._candidate_to_bbox(candidate, new_data.shape)
                for candidate in getattr(result, "candidates", [])
            ]
            ai_confidence = max((bbox.confidence for bbox in bboxes), default=None)
            self._backend.apply_ai_preannotations(
                sample.id,
                bboxes,
                ai_suggestion="real" if bboxes else None,
                ai_confidence=ai_confidence,
            )
            processed_count += 1
            bbox_count += len(bboxes)

        return processed_count, bbox_count

    def _build_detection_params(self) -> DetectionParams:
        """从当前配置构造检测参数。"""
        return DetectionParams(
            thresh=self._config.thresh,
            min_area=self._config.min_area,
            max_area=self._config.max_area,
            sharpness_min=self._config.sharpness,
            sharpness_max=self._config.max_sharpness,
            contrast_min=self._config.contrast,
            edge_margin=self._config.edge_margin,
            dynamic_thresh=self._config.dynamic_thresh,
            kill_flat=self._config.kill_flat,
            kill_dipole=self._config.kill_dipole,
            aspect_ratio_max=self._config.aspect_ratio_max,
            extent_max=self._config.extent_max,
            topk=self._config.topk,
        )

    def _candidate_to_bbox(self, candidate, image_shape: tuple[int, ...]) -> BBox:
        """将候选体中心点转换为固定大小的标注框。"""
        height, width = image_shape[:2]
        patch_size = int(self._config.slice_size)
        half_size = patch_size // 2

        left = max(0, int(candidate.x) - half_size)
        top = max(0, int(candidate.y) - half_size)
        bbox_width = min(patch_size, width - left)
        bbox_height = min(patch_size, height - top)

        return BBox(
            x=left,
            y=top,
            width=max(1, bbox_width),
            height=max(1, bbox_height),
            label=None,
            confidence=float(getattr(candidate, "ai_score", 0.0)),
        )

    def _resolve_inference_engine(self):
        """从父窗口解析当前活动的推理引擎。"""
        parent = self.parentWidget()
        if parent is None:
            return None
        return getattr(parent, "_inference_engine", None)

    def _show_status_message(
        self,
        message: str,
        timeout: int = 3000,
        level: str = "INFO",
    ) -> None:
        """优先复用主窗口状态栏消息，缺失时回退到消息框。"""
        parent = self.parentWidget()
        show_message = getattr(parent, "_show_message", None) if parent is not None else None
        if callable(show_message):
            show_message(message, timeout, level=level)
            return

        if level in {"WARNING", "ERROR"}:
            QMessageBox.warning(self, "AI预标注", message)
        else:
            QMessageBox.information(self, "AI预标注", message)

    def _on_show_dir_help(self) -> None:
        """显示目录格式要求说明"""
        msg = """
        <h3>目录格式要求</h3>
        <p><b>v1 三联图模式:</b></p>
        <ul>
            <li>目录内需包含以下子目录（至少一个）:</li>
            <li><code>positive/</code> - 已标注为真类</li>
            <li><code>negative/</code> - 已标注为假类</li>
            <li><code>unlabeled/</code> - 未标注</li>
        </ul>
        <p>支持格式: <code>*.png</code>, <code>*.jpg</code>, <code>*.jpeg</code></p>

        <p><b>v2 FITS 模式:</b></p>
        <ul>
            <li>目录内需包含以下子目录:</li>
            <li><code>new/</code> - 新图像</li>
            <li><code>old/</code> - 参考图像</li>
        </ul>
        <p>支持格式: <code>*.fits</code></p>
        """
        QMessageBox.information(self, "目录格式要求", msg)

    def _on_export(self) -> None:
        """导出数据集"""
        if self._backend is None:
            return
        output_dir = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if output_dir:
            fmt = "native" if self._current_mode == "v1" else "json"
            self._backend.export_dataset(output_dir, format=fmt)
