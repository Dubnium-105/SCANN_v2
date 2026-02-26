"""SCANN v2 主窗口

UI/UX 设计实现:
- 菜单栏: 文件 | 处理 | AI | 查询 | 视图 | 设置 | 帮助
- 可折叠侧边栏 (240px, Ctrl+B 切换)
- 图像区域 (最大化, ≥ 75% 窗口面积)
- 浮层状态标签 (NEW/OLD/INV)
- 控制栏 (40px): 切换/闪烁/反色/拉伸/标记
- 状态栏: 当前图类型 | 像素坐标 | 天球坐标 | 缩放百分比
- 快捷键: r=闪烁, n=假, y=真, i=反色, 1/2=新旧图, F=适配,
          Space=下一候选, ←→=上下配对, Ctrl+B=侧边栏
- 快捷键非全局，窗口焦点在程序内才有效
- 暗色主题 (#1E1E1E)
"""

from __future__ import annotations

from typing import Optional
from pathlib import Path
import math

import logging

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from scann.core.astrometry import pixel_to_wcs, format_ra_hms, format_dec_dms
from scann.core.fits_io import read_fits, write_fits
from scann.core.image_aligner import align
from scann.core.image_processor import histogram_stretch, denoise, pseudo_flat_field
from scann.core.models import (
    AppConfig,
    Candidate,
    FitsHeader,
    TargetVerdict,
)
from scann.core.observation_report import generate_mpc_report, Observation
from scann.logger_config import get_logger
from scann.services.query_service import QueryService, QueryResult
from scann.gui.dialogs.query_result_popup import QueryResultPopup
from scann.data.file_manager import scan_fits_folder, match_new_old_pairs
from scann.ai.inference import InferenceEngine
from scann.services.detection_service import DetectionPipeline
from scann.gui.image_viewer import FitsImageViewer
from scann.gui.widgets.blink_speed_slider import BlinkSpeedSlider
from scann.gui.widgets.collapsible_sidebar import CollapsibleSidebar
from scann.gui.widgets.coordinate_label import CoordinateLabel
from scann.gui.widgets.histogram_panel import HistogramPanel
from scann.gui.widgets.no_scroll_spinbox import NoScrollDoubleSpinBox, NoScrollSpinBox
from scann.gui.widgets.overlay_label import OverlayLabel
from scann.gui.widgets.suspect_table import SuspectTableWidget
from scann.services.blink_service import BlinkService, BlinkState


# ─── 暗色主题样式表 ───
DARK_THEME_QSS = """
QMainWindow {
    background-color: #1E1E1E;
}
QWidget {
    background-color: #1E1E1E;
    color: #D4D4D4;
    font-size: 12px;
}
QMenuBar {
    background-color: #333333;
    color: #D4D4D4;
    border-bottom: 1px solid #3C3C3C;
}
QMenuBar::item:selected {
    background-color: #094771;
}
QMenu {
    background-color: #252526;
    color: #D4D4D4;
    border: 1px solid #3C3C3C;
}
QMenu::item:selected {
    background-color: #094771;
}
QMenu::separator {
    height: 1px;
    background-color: #3C3C3C;
}
QPushButton {
    background-color: #333333;
    color: #D4D4D4;
    border: 1px solid #3C3C3C;
    border-radius: 3px;
    padding: 4px 8px;
    min-height: 24px;
}
QPushButton:hover {
    background-color: #3C3C3C;
}
QPushButton:pressed {
    background-color: #094771;
}
QPushButton:checked {
    background-color: #094771;
    border-color: #2196F3;
}
QPushButton:disabled {
    background-color: #2A2A2A;
    color: #555555;
}
QListWidget {
    background-color: #252526;
    border: 1px solid #3C3C3C;
    color: #D4D4D4;
}
QListWidget::item:selected {
    background-color: #094771;
}
QProgressBar {
    background-color: #333333;
    border: 1px solid #3C3C3C;
    border-radius: 2px;
    text-align: center;
    color: #D4D4D4;
}
QProgressBar::chunk {
    background-color: #2196F3;
}
QStatusBar {
    background-color: #007ACC;
    color: white;
    font-size: 11px;
}
QSplitter::handle {
    background-color: #3C3C3C;
    width: 2px;
}
QLabel {
    color: #D4D4D4;
}
"""


class MainWindow(QMainWindow):
    """SCANN v2 主窗口

    分区:
    ┌─────────────────────────────────────────────┐
    │ 菜单栏                                       │
    ├──────────┬──────────────────────────────────┤
    │ 侧边栏   │  [OverlayLabel]                   │
    │ (可折叠) │  FitsImageViewer (弹性填充)        │
    │          │  [控制栏 40px]                     │
    ├──────────┴──────────────────────────────────┤
    │ 状态栏                                       │
    └─────────────────────────────────────────────┘
    """

    def __init__(self):
        super().__init__()

        # ── 配置 (最先加载，后续初始化依赖它) ──
        from scann.core.config import load_config
        self._config = load_config()

        self.setWindowTitle("SCANN v2 - Star/Source Classification and Analysis Neural Network")
        self.resize(self._config.window_width, self._config.window_height)
        self.setMinimumSize(1024, 768)

        # 暗色主题
        self.setStyleSheet(DARK_THEME_QSS)

        # ── 定时器 ──
        self.blink_timer = QTimer(self)
        self.blink_timer.timeout.connect(self._on_blink_tick)

        # ── 数据状态 ──
        self._candidates: list[Candidate] = []
        self._current_candidate_idx: int = -1
        self._new_image_data: Optional[np.ndarray] = None
        self._old_image_data: Optional[np.ndarray] = None

        # ── 文件管理 ──
        self._new_folder: str = ""
        self._old_folder: str = ""
        self._image_pairs: list = []  # FitsImagePair 列表
        self._current_pair_idx: int = -1
        self._current_pair_using_aligned: bool = False
        self._new_fits_header: Optional[FitsHeader] = None
        self._old_fits_header: Optional[FitsHeader] = None
        
        # ── 候选目标缓存：key 是配对索引，value 是候选目标列表
        self._candidates_cache: dict[int, list[Candidate]] = {}

        # ── AI/推理 ──
        self._inference_engine = None

        # ── 日志 ──
        self._logger = get_logger(__name__)

        # ── 用持久化配置初始化服务 ──
        self.blink_service = BlinkService(speed_ms=self._config.blink_speed_ms)

        # ── 构建 UI ──
        self._init_menu_bar()
        self._init_central_ui()
        self._init_status_bar()
        self._init_histogram_dock()
        self._connect_signals()
        self._init_shortcuts()

        # ── 从配置恢复文件夹路径 ──
        self._new_folder = self._config.new_folder
        self._old_folder = self._config.old_folder

        # ── 从配置恢复 UI 状态 ──
        self._restore_ui_state()

    # ══════════════════════════════════════════════
    #  日志和消息输出
    # ══════════════════════════════════════════════

    def _show_message(self, message: str, timeout: int = 3000, level: str = 'INFO') -> None:
        """统一的消息输出方法，同时输出到状态栏、终端和日志

        Args:
            message: 消息内容
            timeout: 状态栏显示超时时间（毫秒）
            level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        # 输出到状态栏（左下角）
        self.statusBar().showMessage(message, timeout)

        # 输出到终端和日志
        log_level = getattr(logging, level.upper(), logging.INFO)
        self._logger.log(log_level, message)

    # ══════════════════════════════════════════════
    #  菜单栏
    # ══════════════════════════════════════════════

    def _init_menu_bar(self) -> None:
        """初始化菜单栏: 文件 | 处理 | AI | 查询 | 视图 | 设置 | 帮助"""
        mb = self.menuBar()

        # ── 文件 ──
        file_menu = mb.addMenu("文件(&F)")

        self.act_open_new = file_menu.addAction("打开新图文件夹")
        self.act_open_new.setShortcut(QKeySequence("Ctrl+O"))

        self.act_open_old = file_menu.addAction("打开旧图文件夹")
        self.act_open_old.setShortcut(QKeySequence("Ctrl+Shift+O"))

        file_menu.addSeparator()

        self.act_save = file_menu.addAction("保存当前图像")
        self.act_save.setShortcut(QKeySequence("Ctrl+S"))

        self.act_save_marked = file_menu.addAction("另存为标记图...")
        self.act_save_marked.setShortcut(QKeySequence("Ctrl+Shift+S"))

        file_menu.addSeparator()

        self.menu_recent = file_menu.addMenu("最近打开")

        file_menu.addSeparator()
        self.act_exit = file_menu.addAction("退出")
        self.act_exit.setShortcut(QKeySequence("Alt+F4"))
        self.act_exit.triggered.connect(self.close)

        # ── 处理 ──
        proc_menu = mb.addMenu("处理(&P)")
        self.act_align = proc_menu.addAction("批量对齐")
        proc_menu.addSeparator()
        self.act_batch_process = proc_menu.addAction("批量降噪/伪平场...")
        proc_menu.addSeparator()
        self.act_histogram = proc_menu.addAction("直方图拉伸")

        # ── AI ──
        ai_menu = mb.addMenu("AI(&A)")
        self.act_detect = ai_menu.addAction("批量检测")
        self.act_detect.setShortcut(QKeySequence("F5"))
        ai_menu.addSeparator()
        self.act_train = ai_menu.addAction("训练模型...")
        self.act_load_model = ai_menu.addAction("加载模型...")
        self.act_model_info = ai_menu.addAction("模型信息")
        ai_menu.addSeparator()
        self.act_annotation = ai_menu.addAction("🏷️ 标注工具...")
        self.act_annotation.setShortcut(QKeySequence("Ctrl+L"))

        # ── 查询 ──
        query_menu = mb.addMenu("查询(&Q)")
        self.act_query_vsx = query_menu.addAction("查询 VSX")
        self.act_query_mpc = query_menu.addAction("查询 MPC")
        self.act_query_simbad = query_menu.addAction("查询 SIMBAD")
        self.act_query_tns = query_menu.addAction("查询 TNS")
        self.act_query_satellite = query_menu.addAction("人造卫星查询")
        query_menu.addSeparator()
        self.act_mpc_report = query_menu.addAction("生成 MPC 80列报告")
        self.act_mpc_report.setShortcut(QKeySequence("Ctrl+E"))

        # ── 视图 ──
        view_menu = mb.addMenu("视图(&V)")
        self.act_toggle_sidebar = view_menu.addAction("切换侧边栏")
        self.act_toggle_sidebar.setShortcut(QKeySequence("Ctrl+B"))

        view_menu.addSeparator()

        self.act_fit_view = view_menu.addAction("适配窗口")
        self.act_zoom_actual = view_menu.addAction("实际大小")
        self.act_zoom_actual.setShortcut(QKeySequence("Ctrl+0"))
        self.act_zoom_in = view_menu.addAction("放大")
        self.act_zoom_in.setShortcut(QKeySequence("Ctrl++"))
        self.act_zoom_out = view_menu.addAction("缩小")
        self.act_zoom_out.setShortcut(QKeySequence("Ctrl+-"))

        view_menu.addSeparator()

        self.act_show_markers = view_menu.addAction("显示候选标记")
        self.act_show_markers.setCheckable(True)
        self.act_show_markers.setChecked(True)

        self.act_show_mpcorb = view_menu.addAction("显示 MPCORB 叠加")
        self.act_show_mpcorb.setCheckable(True)
        self.act_show_mpcorb.setChecked(True)

        self.act_show_known = view_menu.addAction("显示已知天体")
        self.act_show_known.setCheckable(True)
        self.act_show_known.setChecked(True)

        # ── 设置 ──
        settings_menu = mb.addMenu("设置(&S)")
        self.act_preferences = settings_menu.addAction("首选项...")
        self.act_preferences.setShortcut(QKeySequence("Ctrl+,"))
        settings_menu.addSeparator()
        self.act_mpcorb_file = settings_menu.addAction("MPCORB 文件...")
        self.act_scheduler = settings_menu.addAction("计划任务...")

        # ── 帮助 ──
        help_menu = mb.addMenu("帮助(&H)")
        self.act_shortcut_help = help_menu.addAction("快捷键列表")
        self.act_docs = help_menu.addAction("使用文档")
        help_menu.addSeparator()
        self.act_about = help_menu.addAction("关于 SCANN v2")

    # ══════════════════════════════════════════════
    #  中央区域
    # ══════════════════════════════════════════════

    def _init_central_ui(self) -> None:
        """初始化中央布局: 侧边栏 | 图像区域"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(6)

        # ── 可折叠侧边栏 ──
        self.sidebar = CollapsibleSidebar()
        sidebar_layout = self.sidebar.content_layout

        # 文件夹按钮
        btn_layout = QHBoxLayout()
        self.btn_new_folder = QPushButton("📂 新图")
        self.btn_old_folder = QPushButton("📂 旧图")
        btn_layout.addWidget(self.btn_new_folder)
        btn_layout.addWidget(self.btn_old_folder)
        sidebar_layout.addLayout(btn_layout)

        # 功能按钮
        func_layout = QHBoxLayout()
        self.btn_align = QPushButton("🔗 对齐")
        self.btn_detect = QPushButton("⚡ 检测")
        self.btn_detect.setStyleSheet(
            "QPushButton { background-color: #FFEB3B; color: #1E1E1E; font-weight: bold; }"
            "QPushButton:hover { background-color: #FFF176; }"
        )
        func_layout.addWidget(self.btn_align)
        func_layout.addWidget(self.btn_detect)
        sidebar_layout.addLayout(func_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(16)
        sidebar_layout.addWidget(self.progress_bar)

        # 图像配对列表
        lbl_pairs = QLabel("📁 图像配对:")
        lbl_pairs.setStyleSheet("font-weight: bold;")
        sidebar_layout.addWidget(lbl_pairs)
        self.file_list = QListWidget()
        sidebar_layout.addWidget(self.file_list, 2)

        # 可疑目标表格
        self.suspect_table = SuspectTableWidget()
        sidebar_layout.addWidget(self.suspect_table, 3)
        self.main_splitter.addWidget(self.sidebar)

        # ── 右侧图像区域 ──
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # 图像查看器 (弹性填充)
        self.image_viewer = FitsImageViewer()
        right_layout.addWidget(self.image_viewer, 1)

        # 浮层标签 (覆盖在 image_viewer 上)
        self.overlay_state = OverlayLabel("准备就绪", parent=self.image_viewer)
        self.overlay_state.move(10, 10)
        self.overlay_state.set_state("new")

        self.overlay_inv = OverlayLabel("INV", parent=self.image_viewer)
        self.overlay_inv.set_state("inv")
        self.overlay_inv.hide_label()

        self.overlay_blink = OverlayLabel("⚡", parent=self.image_viewer)
        self.overlay_blink.set_state("blink")
        self.overlay_blink.hide_label()

        # ── 控制栏 (固定 40px) ──
        ctrl_widget = QWidget()
        ctrl_widget.setFixedHeight(40)
        ctrl_widget.setStyleSheet("background-color: #252526; border-top: 1px solid #3C3C3C;")
        ctrl_layout = QHBoxLayout(ctrl_widget)
        ctrl_layout.setContentsMargins(4, 2, 4, 2)
        ctrl_layout.setSpacing(4)

        # 新/旧图切换
        self.btn_show_new = QPushButton("[1] 新图")
        self.btn_show_old = QPushButton("[2] 旧图")
        self.btn_show_new.setCheckable(True)
        self.btn_show_old.setCheckable(True)
        self.btn_show_new.setChecked(True)
        ctrl_layout.addWidget(self.btn_show_new)
        ctrl_layout.addWidget(self.btn_show_old)

        # 分隔
        sep1 = QLabel("|")
        sep1.setStyleSheet("color: #3C3C3C;")
        ctrl_layout.addWidget(sep1)

        # 闪烁
        self.btn_blink = QPushButton("✨ 闪烁 (R)")
        self.btn_blink.setCheckable(True)
        ctrl_layout.addWidget(self.btn_blink)

        # 闪烁速度
        self.blink_speed = BlinkSpeedSlider()
        ctrl_layout.addWidget(self.blink_speed)

        # 分隔
        sep2 = QLabel("|")
        sep2.setStyleSheet("color: #3C3C3C;")
        ctrl_layout.addWidget(sep2)

        # 反色
        self.btn_invert = QPushButton("🔄 反色 (I)")
        self.btn_invert.setCheckable(True)
        ctrl_layout.addWidget(self.btn_invert)

        # 直方图拉伸
        self.btn_histogram = QPushButton("📊 拉伸")
        ctrl_layout.addWidget(self.btn_histogram)

        # 弹性空间
        ctrl_layout.addStretch()

        # 标记按钮
        self.btn_mark_real = QPushButton("✅ 真 (Y)")
        self.btn_mark_real.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }"
            "QPushButton:hover { background-color: #66BB6A; }"
            "QPushButton:disabled { background-color: #2A2A2A; color: #555; }"
        )
        self.btn_mark_bogus = QPushButton("❌ 假 (N)")
        self.btn_mark_bogus.setStyleSheet(
            "QPushButton { background-color: #F44336; color: white; font-weight: bold; }"
            "QPushButton:hover { background-color: #EF5350; }"
            "QPushButton:disabled { background-color: #2A2A2A; color: #555; }"
        )
        self.btn_next_candidate = QPushButton("➡ 下一个")

        ctrl_layout.addWidget(self.btn_mark_real)
        ctrl_layout.addWidget(self.btn_mark_bogus)
        ctrl_layout.addWidget(self.btn_next_candidate)

        right_layout.addWidget(ctrl_widget)

        self.main_splitter.addWidget(right_panel)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes(
            [self.sidebar.preferred_width, max(1, self.width() - self.sidebar.preferred_width)]
        )
        self.main_splitter.splitterMoved.connect(self._on_main_splitter_moved)
        main_layout.addWidget(self.main_splitter, 1)

        # ── 信号连接 ──
        # 连接已移至 __init__，确保依赖的 Dock 已初始化

    # ══════════════════════════════════════════════
    #  状态栏
    # ══════════════════════════════════════════════

    def _init_status_bar(self) -> None:
        """初始化状态栏: 当前图 | 像素坐标 | 天球坐标 | 缩放"""
        sb = QStatusBar()
        self.setStatusBar(sb)

        self.status_image_type = QLabel("准备就绪")
        self.status_image_type.setMinimumWidth(80)
        sb.addWidget(self.status_image_type)

        sep = QLabel("|")
        sep.setStyleSheet("color: rgba(255,255,255,0.3);")
        sb.addWidget(sep)

        self.status_pixel_coord = CoordinateLabel("X: --  Y: --")
        self.status_pixel_coord.setMinimumWidth(120)
        sb.addWidget(self.status_pixel_coord)

        sep2 = QLabel("|")
        sep2.setStyleSheet("color: rgba(255,255,255,0.3);")
        sb.addWidget(sep2)

        self.status_wcs_coord = CoordinateLabel("RA: --  Dec: --")
        self.status_wcs_coord.setMinimumWidth(200)
        sb.addWidget(self.status_wcs_coord)

        self.status_zoom = QLabel("100%")
        sb.addPermanentWidget(self.status_zoom)

    # ══════════════════════════════════════════════
    #  直方图 Dock
    # ══════════════════════════════════════════════

    def _init_histogram_dock(self) -> None:
        """初始化直方图拉伸面板 (可停靠 DockWidget)"""
        self.histogram_panel = HistogramPanel(self)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.histogram_panel)
        self.histogram_panel.setVisible(False)

    # ══════════════════════════════════════════════
    #  信号连接
    # ══════════════════════════════════════════════

    def _connect_signals(self) -> None:
        """连接所有信号与槽"""
        # ── 控制栏按钮 ──
        self.btn_show_new.clicked.connect(self._on_show_new)
        self.btn_show_old.clicked.connect(self._on_show_old)
        self.btn_blink.clicked.connect(self._on_blink_toggle)
        self.btn_invert.clicked.connect(self._on_invert_toggle)
        self.btn_mark_real.clicked.connect(self._on_mark_real)
        self.btn_mark_bogus.clicked.connect(self._on_mark_bogus)
        self.btn_next_candidate.clicked.connect(self._on_next_candidate)
        self.btn_histogram.clicked.connect(self._on_toggle_histogram)

        # ── 闪烁速度 ──
        self.blink_speed.speed_changed.connect(self._on_blink_speed_changed)

        # ── 侧边栏按钮 ──
        self.btn_new_folder.clicked.connect(self._on_open_new_folder)
        self.btn_old_folder.clicked.connect(self._on_open_old_folder)
        self.btn_align.clicked.connect(self._on_batch_align)
        self.btn_detect.clicked.connect(self._on_batch_detect)

        # ── 文件菜单 ──
        self.act_open_new.triggered.connect(self._on_open_new_folder)
        self.act_open_old.triggered.connect(self._on_open_old_folder)
        self.act_save.triggered.connect(self._on_save_image)
        self.act_save_marked.triggered.connect(self._on_save_marked_image)

        # ── 处理菜单 ──
        self.act_align.triggered.connect(self._on_batch_align)
        self.act_batch_process.triggered.connect(self._on_batch_process)
        self.act_histogram.triggered.connect(self._on_toggle_histogram)

        # ── AI 菜单 ──
        self.act_detect.triggered.connect(self._on_batch_detect)
        self.act_train.triggered.connect(self._on_open_training)
        self.act_load_model.triggered.connect(self._on_load_model)
        self.act_model_info.triggered.connect(self._on_model_info)
        self.act_annotation.triggered.connect(self._on_open_annotation)

        # ── 查询菜单 ──
        self.act_query_vsx.triggered.connect(lambda: self._on_menu_query("vsx"))
        self.act_query_mpc.triggered.connect(lambda: self._on_menu_query("mpc"))
        self.act_query_simbad.triggered.connect(lambda: self._on_menu_query("simbad"))
        self.act_query_tns.triggered.connect(lambda: self._on_menu_query("tns"))
        self.act_query_satellite.triggered.connect(lambda: self._on_menu_query("satellite"))
        self.act_mpc_report.triggered.connect(self._on_mpc_report)

        # ── 视图菜单 ──
        self.act_toggle_sidebar.triggered.connect(self.sidebar.toggle)
        self.act_fit_view.triggered.connect(self.image_viewer.fit_in_view)
        self.act_zoom_actual.triggered.connect(self._on_zoom_actual)
        self.act_zoom_in.triggered.connect(self._on_zoom_in)
        self.act_zoom_out.triggered.connect(self._on_zoom_out)
        self.act_show_markers.toggled.connect(lambda _: self._update_markers())
        self.act_show_mpcorb.toggled.connect(self._on_toggle_mpcorb)
        self.act_show_known.toggled.connect(self._on_toggle_known)

        # ── 设置菜单 ──
        self.act_preferences.triggered.connect(self._on_open_preferences)
        self.act_mpcorb_file.triggered.connect(self._on_select_mpcorb_file)
        self.act_scheduler.triggered.connect(self._on_open_scheduler)

        # ── 帮助菜单 ──
        self.act_shortcut_help.triggered.connect(self._on_shortcut_help)
        self.act_docs.triggered.connect(self._on_open_docs)
        self.act_about.triggered.connect(self._on_about)

        # ── 可疑目标表格 ──
        self.suspect_table.candidate_selected.connect(self._on_candidate_selected)
        self.suspect_table.candidate_double_clicked.connect(self._on_candidate_double_clicked)

        # ── 文件列表 ──
        self.file_list.currentRowChanged.connect(self._on_pair_selected)

        # ── 图像查看器 ──
        self.image_viewer.point_clicked.connect(self._on_image_clicked)
        self.image_viewer.right_click.connect(self._on_image_right_click)
        self.image_viewer.mouse_moved.connect(self._on_mouse_moved)
        self.image_viewer.zoom_changed.connect(self._on_zoom_changed)

        # ── 直方图 ──
        self.histogram_panel.stretch_changed.connect(self._on_stretch_changed)

    # ══════════════════════════════════════════════
    #  快捷键
    # ══════════════════════════════════════════════

    def _init_shortcuts(self) -> None:
        """初始化快捷键 (非全局，仅窗口焦点内)"""
        shortcuts = {
            "R": self._on_blink_toggle,
            "I": self._on_invert_toggle,
            "Y": self._on_mark_real,
            "N": self._on_mark_bogus,
            "1": self._on_show_new,
            "2": self._on_show_old,
            "F": self.image_viewer.fit_in_view,
            "Space": self._on_next_candidate,
            "Left": self._on_prev_pair,
            "Right": self._on_next_pair,
        }
        for key, handler in shortcuts.items():
            action = QAction(self)
            action.setShortcut(key)
            action.setShortcutContext(Qt.WindowShortcut)  # 非全局
            action.triggered.connect(handler)
            self.addAction(action)

    # ══════════════════════════════════════════════
    #  事件处理
    # ══════════════════════════════════════════════

    def _on_main_splitter_moved(self, _pos: int, _index: int) -> None:
        """记录用户通过拖动分割器调整后的侧边栏宽度。"""
        if not hasattr(self, "main_splitter"):
            return
        sizes = self.main_splitter.sizes()
        if len(sizes) >= 2 and sizes[0] > 0 and not self.sidebar.is_collapsed:
            self.sidebar.set_preferred_width(sizes[0])

    def _on_blink_toggle(self) -> None:
        """切换闪烁"""
        running = self.blink_service.toggle()
        self.btn_blink.setChecked(running)
        if running:
            self.blink_timer.setInterval(self.blink_service.speed_ms)
            self.blink_timer.start()
            # self.overlay_blink.show_label()  # 不显示⚡图标
            # self.overlay_blink.start_pulse()
        else:
            self.blink_timer.stop()
            # self.overlay_blink.stop_pulse()
            # self.overlay_blink.hide_label()

    def _on_blink_tick(self) -> None:
        """闪烁定时回调"""
        state = self.blink_service.tick()
        if state == BlinkState.NEW:
            self._show_image("new")
        else:
            self._show_image("old")

    def _on_blink_speed_changed(self, speed_ms: int) -> None:
        """闪烁速度变化"""
        self.blink_service.speed_ms = speed_ms
        self._config.blink_speed_ms = speed_ms
        if self.blink_service.is_running:
            self.blink_timer.setInterval(speed_ms)

    def _on_invert_toggle(self) -> None:
        """切换反色 (持久状态: 切换图片不重置)"""
        inverted = self.blink_service.toggle_invert()
        self.btn_invert.setChecked(inverted)

        if inverted:
            self.overlay_inv.show_label()
        else:
            self.overlay_inv.hide_label()

        # 刷新当前显示
        current = "new" if self.blink_service.current_state == BlinkState.NEW else "old"
        self._show_image(current)

    def _on_show_new(self) -> None:
        """显示新图"""
        self.btn_show_new.setChecked(True)
        self.btn_show_old.setChecked(False)
        self.blink_service.set_state(BlinkState.NEW)
        self._show_image("new")

    def _on_show_old(self) -> None:
        """显示旧图"""
        self.btn_show_new.setChecked(False)
        self.btn_show_old.setChecked(True)
        self.blink_service.set_state(BlinkState.OLD)
        self._show_image("old")

    def _show_image(self, which: str) -> None:
        """统一的图像显示逻辑

        Args:
            which: "new" 或 "old"
        """
        if which == "new":
            data = self._new_image_data
            label = "NEW"
            color = "new"
        else:
            data = self._old_image_data
            label = "OLD"
            color = "old"

        if data is None:
            self.overlay_state.setText(f"无{label}")
            return

        # 更新直方图面板，显示当前图像的直方图
        self.histogram_panel.set_image_data(data)

        # 应用当前的直方图拉伸参数
        black = self.histogram_panel.black_point
        white = self.histogram_panel.white_point
        stretched = histogram_stretch(data, black_point=black, white_point=white)

        self.image_viewer.set_image_data(
            stretched, inverted=self.blink_service.is_inverted
        )
        self.overlay_state.setText(label)
        self.overlay_state.set_state(color)
        self.status_image_type.setText(f"当前: {label}")

    def _on_mark_real(self) -> None:
        """标记当前候选为真目标"""
        if not self._candidates or self._current_candidate_idx < 0:
            return
        if self._current_candidate_idx >= len(self._candidates):
            return

        candidate = self._candidates[self._current_candidate_idx]
        candidate.verdict = TargetVerdict.REAL
        self.suspect_table.update_candidate(self._current_candidate_idx)
        self._update_markers()
        self._show_message(f"候选 #{self._current_candidate_idx + 1} → 真目标")

    def _on_mark_bogus(self) -> None:
        """标记当前候选为假目标"""
        if not self._candidates or self._current_candidate_idx < 0:
            return
        if self._current_candidate_idx >= len(self._candidates):
            return

        candidate = self._candidates[self._current_candidate_idx]
        candidate.verdict = TargetVerdict.BOGUS
        self.suspect_table.update_candidate(self._current_candidate_idx)
        self._update_markers()
        self._show_message(f"候选 #{self._current_candidate_idx + 1} → 假目标")

    def _on_next_candidate(self) -> None:
        """跳转到下一个候选体"""
        if not self._candidates:
            return
        self._current_candidate_idx = (
            (self._current_candidate_idx + 1) % len(self._candidates)
        )
        self._focus_candidate(self._current_candidate_idx)

    def _on_candidate_selected(self, index: int) -> None:
        """候选表格单击选中"""
        self._current_candidate_idx = index
        self._focus_candidate(index)

    def _on_candidate_double_clicked(self, index: int) -> None:
        """候选表格双击 → 放大到候选体"""
        if 0 <= index < len(self._candidates):
            cand = self._candidates[index]
            self._current_candidate_idx = index
            self.image_viewer.center_on_point(cand.x, cand.y, zoom_to=200)

    def _focus_candidate(self, index: int) -> None:
        """聚焦某个候选体"""
        if 0 <= index < len(self._candidates):
            cand = self._candidates[index]
            self.image_viewer.center_on_point(cand.x, cand.y)
            self._update_markers()
            self.status_pixel_coord.set_pixel_coordinates(cand.x, cand.y)

    def _update_markers(self) -> None:
        """刷新候选标记"""
        show = self.act_show_markers.isChecked()
        self.image_viewer.draw_markers(
            self._candidates,
            selected_idx=self._current_candidate_idx,
            hide_all=not show,
        )

    def _on_toggle_histogram(self) -> None:
        """切换直方图面板"""
        visible = not self.histogram_panel.isVisible()
        self.histogram_panel.setVisible(visible)

    def _on_stretch_changed(self, black: float, white: float) -> None:
        """直方图拉伸参数变化 (仅影响显示)"""
        # 确定当前显示的图像
        if self.blink_service.current_state == BlinkState.NEW:
            data = self._new_image_data
        else:
            data = self._old_image_data

        if data is None:
            return

        # 使用 ImageProcessor 执行线性拉伸
        stretched = histogram_stretch(
            data, black_point=black, white_point=white
        )
        self.image_viewer.set_image_data(
            stretched, inverted=self.blink_service.is_inverted
        )

    def _on_image_clicked(self, x: int, y: int) -> None:
        """图像左键点击"""
        self.status_pixel_coord.set_pixel_coordinates(x, y)

    def _on_image_right_click(self, x: int, y: int) -> None:
        """图像右键点击 → 上下文查询菜单"""
        menu = QMenu(self)

        queries = [
            ("🔍 查询 VSX", "vsx"),
            ("🔍 查询 MPC", "mpc"),
            ("🔍 查询 SIMBAD", "simbad"),
            ("🔍 查询 TNS", "tns"),
            ("🛰️ 查询人造卫星", "satellite"),
        ]
        for label, qtype in queries:
            action = menu.addAction(label)
            action.triggered.connect(
                lambda checked, t=qtype: self._do_query(t, x, y)
            )

        menu.addSeparator()
        act_mpc = menu.addAction("📝 生成 MPC 80列报告")
        act_mpc.triggered.connect(
            lambda checked, cx=x, cy=y: self._on_context_mpc_report(cx, cy)
        )
        menu.addSeparator()
        act_add_cand = menu.addAction("➕ 手动添加候选体")
        act_add_cand.triggered.connect(
            lambda checked, cx=x, cy=y: self._on_context_add_candidate(cx, cy)
        )
        menu.addSeparator()

        act_copy_pixel = menu.addAction("📋 复制像素坐标")
        act_copy_pixel.triggered.connect(
            lambda: QApplication.clipboard().setText(f"{x}, {y}")
        )
        act_copy_wcs = menu.addAction("📋 复制天球坐标")
        act_copy_wcs.triggered.connect(
            lambda checked, cx=x, cy=y: self._on_copy_wcs_coordinates(cx, cy)
        )

        menu.exec_(self.image_viewer.mapToGlobal(
            self.image_viewer.mapFromScene(float(x), float(y))
        ))

    def _do_query(self, query_type: str, x: int, y: int) -> None:
        """执行外部查询"""
        # 若有 WCS 头信息，先转换坐标
        if self._new_fits_header is not None:
            sky = pixel_to_wcs(x, y, self._new_fits_header)
            if sky:
                ra_deg = sky.ra
                dec_deg = sky.dec
                self._show_message(f"正在查询 {query_type} (RA={ra_deg:.4f}, Dec={dec_deg:.4f})...", 5000)

                # 实际查询
                svc = QueryService()
                results: list[QueryResult] = []

                query_map = {
                    "vsx": svc.query_vsx,
                    "mpc": svc.query_mpc,
                    "simbad": svc.query_simbad,
                    "tns": svc.query_tns,
                }
                query_fn = query_map.get(query_type)
                if query_fn:
                    try:
                        results = query_fn(ra_deg, dec_deg)
                    except Exception as e:
                        results = []
                        self._show_message(f"查询失败: {e}", 5000, level='WARNING')

                # 显示结果弹窗
                popup = QueryResultPopup(
                    title=f"{query_type.upper()} 查询结果", parent=self
                )
                if results:
                    lines = []
                    for r in results:
                        lines.append(
                            f"{r.name}  类型={r.object_type}  "
                            f"距离={r.distance_arcsec:.1f}″"
                        )
                    popup.set_content(
                        "\n".join(lines),
                        coords=f"RA={ra_deg:.4f}  Dec={dec_deg:.4f}",
                    )
                    popup.set_success(count=len(results))
                else:
                    popup.set_content(
                        "未找到匹配天体",
                        coords=f"RA={ra_deg:.4f}  Dec={dec_deg:.4f}",
                    )
                popup.show()
                return

        self._show_message(
            f"正在查询 {query_type} ({x}, {y})... (无WCS信息，使用像素坐标)", 5000
        )

    def _on_prev_pair(self) -> None:
        """上一组图像配对"""
        current = self.file_list.currentRow()
        if current > 0:
            self.file_list.setCurrentRow(current - 1)

    def _on_next_pair(self) -> None:
        """下一组图像配对"""
        current = self.file_list.currentRow()
        if current < self.file_list.count() - 1:
            self.file_list.setCurrentRow(current + 1)

    # ══════════════════════════════════════════════
    #  菜单 / 按钮处理方法
    # ══════════════════════════════════════════════

    # ── 文件菜单 ──

    def _on_open_new_folder(self) -> None:
        """打开新图文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择新图文件夹")
        if not folder:
            return

        self._new_folder = folder
        files = scan_fits_folder(folder)

        # 清空并重新填充文件列表
        self.file_list.clear()
        self._image_pairs = []
        self._current_pair_idx = -1
        self._candidates_cache.clear()  # 清空候选目标缓存

        for f in files:
            self.file_list.addItem(f.stem)

        # 自动加载第一张图
        if files:
            try:
                fits_img = read_fits(files[0].path)
                self._new_image_data = fits_img.data
                self._new_fits_header = fits_img.header
                self._on_show_new()
                self.histogram_panel.set_image_data(fits_img.data)
            except Exception as e:
                self._show_message(f"加载失败: {e}", 5000, level='ERROR')
                return

        self._show_message(f"已加载新图文件夹: {folder} ({len(files)} 个文件)")

        # 同步到配置并加入最近打开
        self._config.new_folder = folder
        self._add_recent_folder(folder)

    def _add_recent_folder(self, folder: str) -> None:
        """添加文件夹到最近打开列表"""
        if folder in self._config.recent_folders:
            self._config.recent_folders.remove(folder)
        self._config.recent_folders.insert(0, folder)
        # 限制数量
        max_count = self._config.max_recent_count
        self._config.recent_folders = self._config.recent_folders[:max_count]
        self._on_update_recent_menu()

    def _on_open_old_folder(self) -> None:
        """打开旧图文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择旧图文件夹")
        if not folder:
            return

        self._old_folder = folder
        self._config.old_folder = folder
        self._add_recent_folder(folder)
        old_files = scan_fits_folder(folder)

        # 如果已有新图文件夹，自动配对
        if self._new_folder:
            pairs, only_new, only_old = match_new_old_pairs(
                self._new_folder, folder
            )
            self._image_pairs = pairs

            # 清空候选目标缓存，因为配对已改变
            self._candidates_cache.clear()

            # 更新文件列表显示配对状态
            self.file_list.clear()
            for p in pairs:
                self.file_list.addItem(f"✅ {p.name}")
            for n in only_new:
                self.file_list.addItem(f"🆕 {n} (仅新图)")
            for o in only_old:
                self.file_list.addItem(f"📁 {o} (仅旧图)")

            # 自动加载第一对
            if pairs:
                self._load_pair(0)

            self._show_message(
                f"已配对: {len(pairs)} 对, 仅新图: {len(only_new)}, 仅旧图: {len(only_old)}", 5000
            )
        else:
            self._show_message(f"已选择旧图文件夹: {folder} ({len(old_files)} 个文件)")

    def _on_save_image(self) -> None:
        """保存当前图像"""
        data = self._new_image_data
        if data is None:
            self._show_message("无图像数据可保存")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "保存图像", "", "FITS (*.fits);;所有文件 (*)"
        )
        if not path:
            return

        try:
            write_fits(
                path, data,
                header=self._new_fits_header,
            )
            self._show_message(f"已保存: {path}")
        except Exception as e:
            self._show_message(f"保存失败: {e}", 5000, level='ERROR')

    def _on_save_marked_image(self) -> None:
        """另存为带标记的图像"""
        if self._new_image_data is None:
            self._show_message("无图像数据可保存")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "另存为标记图", "", "PNG (*.png);;FITS (*.fits)"
        )
        if not path:
            return

        try:
            # 获取带标记的渲染图像
            pixmap = self.image_viewer.grab()
            pixmap.save(path)
            self._show_message(f"已保存标记图: {path}")
        except Exception as e:
            self._show_message(f"保存失败: {e}", 5000, level='ERROR')

    def _on_update_recent_menu(self) -> None:
        """更新最近打开菜单"""
        self.menu_recent.clear()
        recent = self._config.recent_folders
        if not recent:
            self.menu_recent.addAction("(无最近打开)")
            return
        for folder in recent:
            action = self.menu_recent.addAction(folder)
            action.triggered.connect(
                lambda checked, f=folder: self._open_recent_folder(f)
            )

    def _open_recent_folder(self, folder: str) -> None:
        """从最近打开列表恢复文件夹"""
        from pathlib import Path
        if not Path(folder).exists():
            self._show_message(f"文件夹不存在: {folder}", 5000, level='WARNING')
            return
        # 按新图文件夹打开
        self._new_folder = folder
        self._config.new_folder = folder
        files = scan_fits_folder(folder)
        self.file_list.clear()
        self._image_pairs = []
        self._current_pair_idx = -1
        for f in files:
            self.file_list.addItem(f.stem)
        if files:
            try:
                fits_img = read_fits(files[0].path)
                self._new_image_data = fits_img.data
                self._new_fits_header = fits_img.header
                self._on_show_new()
                self.histogram_panel.set_image_data(fits_img.data)
            except Exception as e:
                self._show_message(f"加载失败: {e}", 5000, level='ERROR')
                return
        self._show_message(f"已加载: {folder} ({len(files)} 个文件)")

    # ── 处理菜单 ──

    def _on_batch_align(self) -> None:
        """批量对齐"""
        if not self._image_pairs:
            self._show_message("请先加载新旧图文件夹配对")
            return

        success_count = 0
        fail_count = 0
        skip_count = 0
        total = len(self._image_pairs)

        # 进度条可视化，避免大量处理时窗口“假死”
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(0)
        self.btn_align.setEnabled(False)
        self.act_align.setEnabled(False)

        self._show_message(f"开始批量对齐: 共 {total} 对", 3000)

        try:
            for idx, pair in enumerate(self._image_pairs, start=1):
                try:
                    if self._pair_has_aligned_artifacts(pair):
                        skip_count += 1
                        self._logger.info("[%s/%s] 已有对齐裁剪标记，跳过: %s", idx, total, pair.name)
                        self._show_message(f"[{idx}/{total}] 跳过已对齐: {pair.name}", 1000)
                        self.progress_bar.setValue(idx)
                        QApplication.processEvents()
                        continue

                    new_fits = read_fits(pair.new_path)
                    old_fits = read_fits(pair.old_path)

                    # 输入有效性校验：常量图（常见为全零）无法进行星点或特征对齐
                    new_data = np.nan_to_num(new_fits.data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                    old_data = np.nan_to_num(old_fits.data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                    new_span = float(np.percentile(new_data, 99.5) - np.percentile(new_data, 0.5))
                    old_span = float(np.percentile(old_data, 99.5) - np.percentile(old_data, 0.5))
                    if new_span <= 1e-6 or old_span <= 1e-6:
                        fail_count += 1
                        self._logger.error(
                            "[%s/%s] 对齐前校验失败: %s; new_span=%.6g, old_span=%.6g; new=%s old=%s",
                            idx,
                            total,
                            pair.name,
                            new_span,
                            old_span,
                            pair.new_path,
                            pair.old_path,
                        )
                        self._show_message(
                            f"[{idx}/{total}] 跳过无效图像: {pair.name} (新图/旧图近乎常量)",
                            2000,
                            level='WARNING',
                        )
                        self.progress_bar.setValue(idx)
                        QApplication.processEvents()
                        continue

                    self._logger.info("[%s/%s] 开始 Siril 对齐: %s", idx, total, pair.name)
                    self._show_message(f"[{idx}/{total}] Siril 对齐: {pair.name}", 1000)

                    # 优先使用 Siril 对齐；失败时自动回退到内置对齐
                    result = align(new_fits.data, old_fits.data, method="siril")
                    if not result.success or result.aligned_old is None:
                        h, w = new_fits.data.shape[:2]
                        fallback_max_shift = max(100, int(min(h, w) * 0.45))
                        self._logger.warning(
                            "[%s/%s] Siril 对齐失败，回退 auto: %s; reason=%s; fallback_max_shift=%s",
                            idx,
                            total,
                            pair.name,
                            result.error_message,
                            fallback_max_shift,
                        )
                        self._show_message(f"[{idx}/{total}] Siril 失败，回退内置对齐: {pair.name}", 1500, level='WARNING')
                        result = align(
                            new_fits.data,
                            old_fits.data,
                            method="auto",
                            max_shift=fallback_max_shift,
                        )

                    if result.success and result.aligned_old is not None:
                        h, w = new_data.shape[:2]
                        crop_bounds = self._calc_overlap_crop_bounds(
                            w=w,
                            h=h,
                            dx=result.dx,
                            dy=result.dy,
                            aligned_old=result.aligned_old,
                        )
                        if crop_bounds is None:
                            fail_count += 1
                            self._logger.error(
                                "[%s/%s] 对齐后无有效重叠区域: %s; dx=%.3f dy=%.3f",
                                idx,
                                total,
                                pair.name,
                                result.dx,
                                result.dy,
                            )
                            self._show_message(
                                f"[{idx}/{total}] 对齐失败(无重叠区域): {pair.name}",
                                2000,
                                level='WARNING',
                            )
                            self.progress_bar.setValue(idx)
                            QApplication.processEvents()
                            continue

                        x0, x1, y0, y1 = crop_bounds
                        cropped_new = new_data[y0:y1, x0:x1]
                        cropped_old = result.aligned_old[y0:y1, x0:x1]

                        new_aligned_path, old_aligned_path, new_marker_path, old_marker_path = self._aligned_artifact_paths(pair)
                        write_fits(new_aligned_path, cropped_new, new_fits.header)
                        write_fits(old_aligned_path, cropped_old, old_fits.header)
                        marker_text = (
                            "aligned=1\n"
                            f"dx={result.dx:.6f}\n"
                            f"dy={result.dy:.6f}\n"
                            f"crop={x0},{x1},{y0},{y1}\n"
                        )
                        new_marker_path.write_text(marker_text, encoding="utf-8")
                        old_marker_path.write_text(marker_text, encoding="utf-8")

                        success_count += 1
                        self._logger.info(
                            "[%s/%s] 对齐成功并保存裁剪重叠图: %s; new=%s old=%s",
                            idx,
                            total,
                            pair.name,
                            new_aligned_path,
                            old_aligned_path,
                        )
                    else:
                        fail_count += 1
                        self._logger.error(
                            "[%s/%s] 对齐失败: %s; reason=%s",
                            idx,
                            total,
                            pair.name,
                            result.error_message,
                        )

                except Exception as e:
                    fail_count += 1
                    self._logger.exception("[%s/%s] 对齐异常: %s", idx, total, pair.name)
                    self._show_message(f"[{idx}/{total}] 对齐异常: {pair.name} ({e})", 2000, level='ERROR')

                self.progress_bar.setValue(idx)
                QApplication.processEvents()
        finally:
            self.progress_bar.setVisible(False)
            self.btn_align.setEnabled(True)
            self.act_align.setEnabled(True)

        self._show_message(f"对齐完成: 成功 {success_count}, 跳过 {skip_count}, 失败 {fail_count}", 5000)

        # 重新加载当前显示的配对
        if self._current_pair_idx >= 0:
            self._load_pair(self._current_pair_idx)

    def _on_batch_process(self) -> None:
        """打开批量处理对话框"""
        from scann.gui.dialogs.batch_process_dialog import BatchProcessDialog
        dlg = BatchProcessDialog(self)
        dlg.process_started.connect(self._run_batch_process)
        self._batch_dialog = dlg
        dlg.exec_()

    def _run_batch_process(self, params: dict) -> None:
        """执行批量处理 (降噪/伪平场)"""
        input_dir = params.get("input_dir", self._new_folder)
        output_dir = params.get("output_dir", "")
        if not input_dir:
            self._show_message("未指定输入文件夹")
            return

        from pathlib import Path
        if not output_dir:
            output_dir = str(Path(input_dir) / "processed")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        fits_files = scan_fits_folder(input_dir)
        if not fits_files:
            self._show_message("输入文件夹中未找到 FITS 文件")
            return

        success_count = 0
        fail_count = 0
        denoise_method_map = {
            "中值滤波": "median",
            "高斯滤波": "gaussian",
            "双边滤波": "bilateral",
        }

        for i, fits_path in enumerate(fits_files):
            try:
                fits_img = read_fits(str(fits_path))
                data = fits_img.data

                # 降噪
                if params.get("denoise", False):
                    method = denoise_method_map.get(
                        params.get("denoise_method", "中值滤波"), "median"
                    )
                    kernel = params.get("kernel_size", 3)
                    data = denoise(data, method=method, kernel_size=kernel)

                # 伪平场
                if params.get("flat_field", False):
                    sigma = params.get("flat_sigma", 100.0)
                    kernel_size = max(3, int(sigma) * 2 + 1)
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    data = pseudo_flat_field(data, kernel_size=kernel_size)

                # 保存
                out_path = str(Path(output_dir) / fits_path.name)
                write_fits(data, out_path)
                success_count += 1

                # 更新对话框进度
                try:
                    if self._batch_dialog is not None:
                        self._batch_dialog.update_progress(
                            i + 1, len(fits_files), fits_path.name
                        )
                except (AttributeError, RuntimeError):
                    pass
            except Exception:
                fail_count += 1

        try:
            if self._batch_dialog is not None:
                self._batch_dialog.processing_finished()
        except (AttributeError, RuntimeError):
            pass

        self._show_message(f"批量处理完成: 成功 {success_count}, 失败 {fail_count}", 5000)

    def _build_detection_params(self):
        """从 AppConfig 构造 DetectionParams"""
        from scann.core.candidate_detector import DetectionParams
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

    # ── AI 菜单 ──

    def _on_batch_detect(self) -> None:
        """批量检测"""
        if self._new_image_data is None:
            self._show_message("请先加载图像数据")
            return

        old_data = self._old_image_data
        if old_data is None:
            old_data = np.zeros_like(self._new_image_data)

        pipeline = DetectionPipeline(
            detection_params=self._build_detection_params(),
            inference_engine=self._inference_engine,
            patch_size=self._config.slice_size,
        )
        result = pipeline.process_pair(
            pair_name="current",
            new_data=self._new_image_data,
            old_data=old_data,
            # 仅当当前加载的是已对齐裁剪产物时才跳过对齐
            skip_align=bool(self.__dict__.get("_current_pair_using_aligned", False)),
        )

        if result.candidates:
            self.set_candidates(result.candidates)
            # 将检测结果缓存到当前配对索引
            self._candidates_cache[self._current_pair_idx] = result.candidates
            self._show_message(f"检测完成: 发现 {len(result.candidates)} 个候选体", 5000)
        else:
            self._show_message(f"检测完成: 未发现候选体 {result.error or ''}", 5000)
            # 即使没有候选体，也缓存空列表，表示已检测过
            self._candidates_cache[self._current_pair_idx] = []

    def _on_open_training(self) -> None:
        """打开训练对话框"""
        from scann.gui.dialogs.training_dialog import TrainingDialog
        dlg = TrainingDialog(self)
        dlg.training_started.connect(self._on_training_started)
        dlg.training_stopped.connect(self._on_training_stopped)
        self._training_dialog = dlg
        self._training_worker = None
        dlg.exec_()

    def _on_open_annotation(self) -> None:
        """打开标注工具对话框 (非模态)"""
        from scann.gui.dialogs.annotation_dialog import AnnotationDialog
        dlg = AnnotationDialog(self, config=self._config)
        self._annotation_dialog = dlg
        dlg.show()

    def _on_training_started(self, params: dict) -> None:
        """训练开始信号处理: 接收超参数并启动训练线程"""
        self._show_message(
            f"训练已开始: epochs={params.get('epochs', '?')}, "
            f"lr={params.get('lr', '?')}, backbone={params.get('backbone', '?')}, "
            f"device={params.get('device', 'auto')}", 5000
        )
        # 保存训练参数到实例以便后续使用
        self._training_params = params

        # 创建并启动训练工作线程
        from scann.ai.training_worker import TrainingWorker

        self._training_worker = TrainingWorker(params, parent=self)
        self._training_worker.progress.connect(self._on_training_progress)
        self._training_worker.finished.connect(self._on_training_finished)
        self._training_worker.error.connect(self._on_training_error)
        self._training_worker.start()

    def _on_training_progress(self, epoch: int, total: int, loss: float, val_loss: float) -> None:
        """训练进度更新"""
        if self._training_dialog:
            self._training_dialog.update_progress(epoch, total, loss, val_loss)

    def _on_training_finished(self, model_path: str, metrics: dict) -> None:
        """训练完成"""
        if self._training_dialog:
            self._training_dialog.training_finished(model_path)
        self._training_worker = None
        best_f2 = metrics.get('best_f2', 0)
        best_threshold = metrics.get('best_threshold', 0.5)
        self._show_message(
            f"训练完成! 最佳 F2={best_f2:.4f}, 阈值={best_threshold:.3f}", 5000
        )

    def _on_training_error(self, message: str) -> None:
        """训练出错"""
        if self._training_dialog:
            self._training_dialog.log_text.appendPlainText(f"❌ 错误: {message}")
        self._training_worker = None
        self._show_message(f"训练失败: {message}", 5000, level='ERROR')

    def _on_training_stopped(self) -> None:
        """训练停止信号处理"""
        if self._training_worker:
            self._training_worker.stop()
        self._training_worker = None
        self._show_message("训练已停止")

    def _on_load_model(self) -> None:
        """加载 AI 模型 (支持 v1/v2 格式自动检测)"""
        path, _ = QFileDialog.getOpenFileName(
            self, "加载模型", "", "PyTorch 模型 (*.pth *.pt)"
        )
        if not path:
            return

        try:
            from scann.ai.inference import InferenceConfig
            config = InferenceConfig(
                batch_size=self._config.batch_size,
                device=self._config.compute_device,
                model_format=self._config.model_format,
            )
            self._inference_engine = InferenceEngine(model_path=path, config=config)

            # 模型内阈值仅作为参考；实际运行优先采用 GUI 配置阈值
            model_threshold = float(self._inference_engine.threshold)
            gui_threshold = float(self._config.ai_confidence)
            self._inference_engine.threshold = gui_threshold

            # 回写夹紧后的运行时阈值
            self._config.ai_confidence = float(self._inference_engine.threshold)
            self._config.model_path = path
            fmt_info = getattr(self._inference_engine, '_model_format', None)
            fmt_str = fmt_info.value if fmt_info else 'unknown'
            ch_order = getattr(self._inference_engine, '_channel_order', (0, 1, 2))
            self._logger.info(
                "模型已加载: %s (格式=%s, 模型阈值=%.4f, GUI阈值=%.4f, 生效阈值=%.4f, 通道=%s)",
                path,
                fmt_str,
                model_threshold,
                gui_threshold,
                self._inference_engine.threshold,
                ch_order,
            )
            self._show_message(
                f"模型已加载: {path} (格式={fmt_str}, 阈值={self._inference_engine.threshold:.2f})", 5000
            )
        except Exception as e:
            self._inference_engine = None
            self._show_message(f"模型加载失败: {e}", 5000, level='ERROR')

    def _on_model_info(self) -> None:
        """显示模型信息"""
        if self._inference_engine is None or not self._inference_engine.is_ready:
            self._show_message("尚未加载模型")
            return

        model = self._inference_engine.model
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        threshold = self._inference_engine.threshold
        fmt_info = getattr(self._inference_engine, '_model_format', None)
        fmt_str = fmt_info.value if fmt_info else 'unknown'
        ch_order = getattr(self._inference_engine, '_channel_order', (0, 1, 2))

        QMessageBox.information(
            self,
            "模型信息",
            f"<h3>AI 模型信息</h3>"
            f"<p>架构: {model.__class__.__name__}</p>"
            f"<p>模型格式: {fmt_str}</p>"
            f"<p>参数量: {total_params:,}</p>"
            f"<p>检测阈值: {threshold:.4f}</p>"
            f"<p>通道顺序: {ch_order}</p>"
            f"<p>设备: {self._inference_engine.device}</p>",
        )

    # ── 查询菜单 ──

    def _on_menu_query(self, query_type: str) -> None:
        """从菜单栏触发的查询 (无坐标上下文)"""
        # 若有选中候选体则使用其坐标查询，否则提示用户
        if self._candidates and 0 <= self._current_candidate_idx < len(self._candidates):
            cand = self._candidates[self._current_candidate_idx]
            self._do_query(query_type, int(cand.x), int(cand.y))
        else:
            self._show_message("请先选中一个候选体，或在图像上右键进行坐标查询")

    def _on_mpc_report(self) -> None:
        """打开 MPC 80列报告对话框"""
        from scann.gui.dialogs.mpc_report_dialog import MpcReportDialog

        dlg = MpcReportDialog(self)

        # 如果有候选体和 WCS 头信息，生成报告
        if self._candidates and self._new_fits_header is not None:
            from datetime import datetime

            observations = []
            header = self._new_fits_header
            obs_dt = header.observation_datetime or datetime.utcnow()
            obs_code = header.raw.get("OBSERVAT", "")[:3] if header.raw.get("OBSERVAT") else ""

            for cand in self._candidates:
                if cand.verdict == TargetVerdict.BOGUS:
                    continue

                sky = pixel_to_wcs(int(cand.x), int(cand.y), header)
                if sky is None:
                    continue

                observations.append(Observation(
                    designation="",
                    discovery=False,
                    obs_datetime=obs_dt,
                    ra_deg=sky.ra,
                    dec_deg=sky.dec,
                    magnitude=0.0,
                    mag_band="C",
                    observatory_code=obs_code,
                ))

            if observations:
                report = generate_mpc_report(observations)
                dlg.set_report(report)
        elif not self._candidates:
            pass  # 空对话框
        elif self._new_fits_header is None:
            self._show_message("无 WCS 头信息，无法生成 MPC 报告坐标")

        dlg.exec_()

    # ── 视图菜单 ──

    def _on_zoom_actual(self) -> None:
        """重置缩放到 100%"""
        self.image_viewer.resetTransform()
        self.image_viewer._zoom_level = 1.0
        self.image_viewer._emit_zoom()

    def _on_zoom_in(self) -> None:
        """放大"""
        factor = self.image_viewer.ZOOM_FACTOR
        self.image_viewer.scale(factor, factor)
        self.image_viewer._zoom_level *= factor
        self.image_viewer._emit_zoom()

    def _on_zoom_out(self) -> None:
        """缩小"""
        factor = 1.0 / self.image_viewer.ZOOM_FACTOR
        self.image_viewer.scale(factor, factor)
        self.image_viewer._zoom_level *= factor
        self.image_viewer._emit_zoom()

    def _on_toggle_mpcorb(self, checked: bool) -> None:
        """切换 MPCORB 叠加显示"""
        self.image_viewer.set_mpcorb_visible(checked)
        self._show_message(f"MPCORB 叠加: {'开启' if checked else '关闭'}", 2000)

    def _on_toggle_known(self, checked: bool) -> None:
        """切换已知天体显示"""
        self.image_viewer.set_known_objects_visible(checked)
        self._show_message(f"已知天体标记: {'开启' if checked else '关闭'}", 2000)

    # ── 设置菜单 ──

    def _on_open_preferences(self) -> None:
        """打开首选项对话框"""
        from scann.gui.dialogs.settings_dialog import SettingsDialog
        from scann.core.config import save_config
        dlg = SettingsDialog(self._config, parent=self)
        if dlg.exec_():
            # 保存配置到磁盘
            try:
                save_config(self._config)
            except Exception as e:
                self._logger.error(f"保存配置失败: {e}")
            # 同步运行时状态
            self.blink_service.speed_ms = self._config.blink_speed_ms
            if self._inference_engine is not None and self._inference_engine.is_ready:
                # 使 GUI 配置对当前会话立即生效
                self._inference_engine.threshold = self._config.ai_confidence
                self._inference_engine.config.batch_size = self._config.batch_size
                self._logger.info(
                    "已应用GUI推理参数: threshold=%.4f, batch_size=%d",
                    self._inference_engine.threshold,
                    self._inference_engine.config.batch_size,
                )
            self._show_message("设置已保存")

    def _on_select_mpcorb_file(self) -> None:
        """选择 MPCORB 数据文件"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 MPCORB 文件", "", "DAT 文件 (*.dat);;所有文件 (*)"
        )
        if not path:
            return

        self._config.mpcorb_path = path
        # 立即持久化保存 MPCORB 路径
        try:
            from scann.core.config import save_config as _save_cfg
            _save_cfg(self._config)
        except Exception:
            pass
        try:
            from scann.core.mpcorb import MpcorbParser
            parser = MpcorbParser(path)
            count = parser.load()
            self._show_message(f"已加载 MPCORB: {count} 个小行星", 5000)
        except Exception as e:
            self._show_message(f"MPCORB 加载失败: {e}", 5000, level='ERROR')

    def _on_open_scheduler(self) -> None:
        """打开计划任务设置"""
        self._show_message("计划任务功能开发中，敬请期待")

    # ── 帮助菜单 ──

    def _on_shortcut_help(self) -> None:
        """显示快捷键帮助对话框"""
        from scann.gui.dialogs.shortcut_help_dialog import ShortcutHelpDialog
        dlg = ShortcutHelpDialog(self)
        dlg.exec_()

    def _on_open_docs(self) -> None:
        """打开使用文档"""
        import webbrowser
        webbrowser.open("https://github.com/Dubnium-105/SCANN_v2/wiki")

    def _on_about(self) -> None:
        """显示关于对话框"""
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.about(
            self,
            "关于 SCANN v2",
            "<h3>SCANN v2</h3>"
            "<p>Star/Source Classification and Analysis Neural Network</p>"
            "<p>版本: 2.0.0-dev</p>"
            "<p>基于深度学习的天文瞬变源自动检测工具</p>",
        )

    # ── 图像查看器信号处理 ──

    def _on_mouse_moved(self, x: int, y: int) -> None:
        """鼠标在图像上移动 → 更新状态栏像素坐标"""
        self.status_pixel_coord.set_pixel_coordinates(x, y)

        # 若已加载 WCS 头信息，同步更新天球坐标
        if self._new_fits_header is not None:
            sky = pixel_to_wcs(x, y, self._new_fits_header)
            if sky:
                self.status_wcs_coord.set_wcs_coordinates(
                    format_ra_hms(sky.ra), format_dec_dms(sky.dec)
                )

    def _on_zoom_changed(self, zoom_pct: float) -> None:
        """缩放比例变化 → 更新状态栏"""
        self.status_zoom.setText(f"{zoom_pct:.0f}%")

    # ── 右键上下文菜单处理 ──

    def _on_context_mpc_report(self, x: int, y: int) -> None:
        """右键菜单 → 生成 MPC 报告"""
        # 尝试定位最近的候选体
        best_idx = -1
        best_dist = float('inf')
        for i, c in enumerate(self._candidates):
            dist = (c.x - x) ** 2 + (c.y - y) ** 2
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx >= 0 and best_dist < 50 ** 2:  # 50像素范围内
            self._current_candidate_idx = best_idx
            self._focus_candidate(best_idx)

        self._on_mpc_report()

    def _on_context_add_candidate(self, x: int, y: int) -> None:
        """右键菜单 → 手动添加候选体"""
        candidate = Candidate(
            x=x, y=y, is_manual=True,
            verdict=TargetVerdict.UNKNOWN,
        )
        self._candidates.append(candidate)
        self._current_candidate_idx = len(self._candidates) - 1
        self.suspect_table.set_candidates(self._candidates)
        self._update_markers()
        self._show_message(f"已添加手动候选体 ({x}, {y})")

    def _on_copy_wcs_coordinates(self, x: int, y: int) -> None:
        """右键菜单 → 复制天球坐标"""
        if self._new_fits_header is None:
            self._show_message("无 WCS 头信息，无法转换坐标")
            return

        sky = pixel_to_wcs(x, y, self._new_fits_header)
        if sky:
            text = f"{format_ra_hms(sky.ra)}  {format_dec_dms(sky.dec)}"
            QApplication.clipboard().setText(text)
            self._show_message(f"已复制: {text}")
        else:
            self._show_message("WCS 转换失败")

    # ══════════════════════════════════════════════
    #  图像配对加载
    # ══════════════════════════════════════════════

    def _load_pair(self, index: int) -> None:
        """加载指定索引的图像配对"""
        if index < 0 or index >= len(self._image_pairs):
            return

        pair = self._image_pairs[index]
        self._current_pair_idx = index

        # 切换图像配对时，先清空当前候选目标
        self._candidates = []
        self._current_candidate_idx = -1
        self._update_markers()
        self.suspect_table.set_candidates([])

        try:
            new_path, old_path, using_aligned = self._resolve_pair_image_paths(pair)
            new_fits = read_fits(new_path)
            old_fits = read_fits(old_path)
            self._new_image_data = new_fits.data
            self._old_image_data = old_fits.data
            self._current_pair_using_aligned = bool(using_aligned)

            # 防御性修复：历史对齐产物可能仍残留 L 型黑边。
            if using_aligned:
                bounds = self._calc_nonzero_valid_bounds(self._old_image_data)
                if bounds is not None:
                    x0, x1, y0, y1 = bounds
                    if (x1 - x0) >= 16 and (y1 - y0) >= 16:
                        self._new_image_data = self._new_image_data[y0:y1, x0:x1]
                        self._old_image_data = self._old_image_data[y0:y1, x0:x1]

            self._new_fits_header = new_fits.header
            self._old_fits_header = old_fits.header
            self._on_show_new()
            self.histogram_panel.set_image_data(self._new_image_data)

            if using_aligned:
                self._logger.info("加载已对齐裁剪图像: %s", pair.name)
            
            # 从缓存中恢复该配对的候选目标（如果有）
            if index in self._candidates_cache:
                self.set_candidates(self._candidates_cache[index])
        except Exception as e:
            self._show_message(f"加载失败: {e}", 5000, level='ERROR')

    def _aligned_artifact_paths(self, pair) -> tuple[Path, Path, Path, Path]:
        """返回配对图像的对齐裁剪产物路径。

        Returns:
            (new_aligned_path, old_aligned_path, new_marker_path, old_marker_path)
        """
        new_path = Path(pair.new_path)
        old_path = Path(pair.old_path)

        new_aligned_path = new_path.with_name(f"{new_path.stem}__aligned_crop{new_path.suffix}")
        old_aligned_path = old_path.with_name(f"{old_path.stem}__aligned_crop{old_path.suffix}")
        new_marker_path = new_path.with_name(f"{new_path.stem}__aligned.marker")
        old_marker_path = old_path.with_name(f"{old_path.stem}__aligned.marker")
        return new_aligned_path, old_aligned_path, new_marker_path, old_marker_path

    def _pair_has_aligned_artifacts(self, pair) -> bool:
        """配对是否已有可复用的对齐裁剪结果。"""
        new_aligned_path, old_aligned_path, new_marker_path, old_marker_path = self._aligned_artifact_paths(pair)
        return (
            new_aligned_path.is_file()
            and old_aligned_path.is_file()
            and new_marker_path.is_file()
            and old_marker_path.is_file()
        )

    def _resolve_pair_image_paths(self, pair) -> tuple[Path, Path, bool]:
        """解析配对应使用的图像路径（优先使用对齐裁剪图）。"""
        if self._pair_has_aligned_artifacts(pair):
            new_aligned_path, old_aligned_path, _, _ = self._aligned_artifact_paths(pair)
            return new_aligned_path, old_aligned_path, True
        return Path(pair.new_path), Path(pair.old_path), False

    def _calc_nonzero_valid_bounds(self, image: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        """估计旧图有效区域边界（用于移除对齐后黑边）。"""
        if image is None or image.size == 0:
            return None

        arr = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        mask = np.abs(arr) > 1e-6
        if not np.any(mask):
            return None

        row_ratio = np.mean(mask, axis=1)
        col_ratio = np.mean(mask, axis=0)

        row_valid = row_ratio > 0.98
        col_valid = col_ratio > 0.98
        # 如果阈值过严（如存在窄黑边），回退为“该行/列存在任意有效像素”
        if not np.any(row_valid):
            row_valid = np.any(mask, axis=1)
        if not np.any(col_valid):
            col_valid = np.any(mask, axis=0)
        if not np.any(row_valid) or not np.any(col_valid):
            return None

        ys = np.where(row_valid)[0]
        xs = np.where(col_valid)[0]
        y0, y1 = int(ys[0]), int(ys[-1] + 1)
        x0, x1 = int(xs[0]), int(xs[-1] + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, x1, y0, y1

    def _calc_overlap_crop_bounds(
        self,
        w: int,
        h: int,
        dx: float,
        dy: float,
        aligned_old: Optional[np.ndarray] = None,
    ) -> Optional[tuple[int, int, int, int]]:
        """根据平移量 + 旧图有效区域，计算新旧图重叠裁剪区域。"""
        x0 = max(0, int(math.ceil(dx)))
        x1 = min(w, int(math.floor(w + dx)))
        y0 = max(0, int(math.ceil(dy)))
        y1 = min(h, int(math.floor(h + dy)))

        if aligned_old is not None:
            valid = self._calc_nonzero_valid_bounds(aligned_old)
            if valid is not None:
                vx0, vx1, vy0, vy1 = valid
                x0 = max(x0, vx0)
                x1 = min(x1, vx1)
                y0 = max(y0, vy0)
                y1 = min(y1, vy1)

        if x1 <= x0 or y1 <= y0:
            return None
        return x0, x1, y0, y1

    def _on_pair_selected(self, index: int) -> None:
        """配对列表选择事件"""
        if index < 0 or index >= len(self._image_pairs):
            return
        self._load_pair(index)

    # ══════════════════════════════════════════════
    #  公共 API
    # ══════════════════════════════════════════════

    def set_image_data(
        self, new_data: Optional[np.ndarray], old_data: Optional[np.ndarray]
    ) -> None:
        """设置当前图像配对数据"""
        self._new_image_data = new_data
        self._old_image_data = old_data
        self._on_show_new()

        if new_data is not None:
            self.histogram_panel.set_image_data(new_data)

    def set_candidates(self, candidates: list[Candidate]) -> None:
        """设置检测到的候选体列表"""
        self._candidates = candidates
        self._current_candidate_idx = 0 if candidates else -1
        self.suspect_table.set_candidates(candidates)
        self._update_markers()

    # ══════════════════════════════════════════════
    #  窗口事件
    # ══════════════════════════════════════════════

    def closeEvent(self, event) -> None:
        """窗口关闭 → 自动保存配置"""
        if self._config.confirm_before_close:
            from PyQt5.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self, "确认退出",
                "确定要退出 SCANN v2 吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return

        # 将运行时状态回写到配置
        self._save_runtime_state()

        # 持久化保存到磁盘
        try:
            from scann.core.config import save_config
            save_config(self._config)
            self._logger.info("配置已自动保存")
        except Exception as e:
            self._logger.error(f"退出时保存配置失败: {e}")

        super().closeEvent(event)

    def _save_runtime_state(self) -> None:
        """将运行时状态同步到配置对象"""
        self._config.new_folder = self._new_folder
        self._config.old_folder = self._old_folder
        self._config.blink_speed_ms = self.blink_service.speed_ms

        # 直方图拉伸参数
        self._config.stretch_black_point = self.histogram_panel.black_point
        self._config.stretch_white_point = self.histogram_panel.white_point
        mode_names = ["线性", "对数", "平方根", "Asinh", "自动拉伸"]
        mode_idx = self.histogram_panel.combo_mode.currentIndex()
        if 0 <= mode_idx < len(mode_names):
            self._config.stretch_mode = mode_names[mode_idx]

        # 视图开关
        self._config.show_markers = self.act_show_markers.isChecked()
        self._config.show_mpcorb = self.act_show_mpcorb.isChecked()
        self._config.show_known_objects = self.act_show_known.isChecked()
        self._config.histogram_visible = self.histogram_panel.isVisible()
        self._config.sidebar_collapsed = self.sidebar.is_collapsed
        self._config.sidebar_width = self.sidebar.preferred_width

        # 窗口几何
        self._config.window_width = self.width()
        self._config.window_height = self.height()

    def _restore_ui_state(self) -> None:
        """从配置恢复 UI 状态 (在构建 UI 后调用)"""
        cfg = self._config

        # 闪烁速度滑块
        self.blink_speed.speed_ms = cfg.blink_speed_ms

        # 视图菜单开关
        self.act_show_markers.setChecked(cfg.show_markers)
        self.act_show_mpcorb.setChecked(cfg.show_mpcorb)
        self.act_show_known.setChecked(cfg.show_known_objects)

        # 直方图面板可见性
        self.histogram_panel.setVisible(cfg.histogram_visible)

        # 直方图拉伸预设模式
        mode_names = ["线性", "对数", "平方根", "Asinh", "自动拉伸"]
        if cfg.stretch_mode in mode_names:
            self.histogram_panel.combo_mode.setCurrentIndex(
                mode_names.index(cfg.stretch_mode)
            )

        # 侧边栏折叠状态
        self.sidebar.set_preferred_width(cfg.sidebar_width)
        if hasattr(self, "main_splitter"):
            self.main_splitter.setSizes(
                [cfg.sidebar_width, max(1, self.width() - cfg.sidebar_width)]
            )
        if cfg.sidebar_collapsed:
            self.sidebar.collapse()

    def resizeEvent(self, event) -> None:
        """窗口大小变化 → 自动折叠侧边栏"""
        super().resizeEvent(event)
        if self._config.auto_collapse_sidebar:
            self.sidebar.auto_collapse_check(self.width())
        else:
            pass  # 不自动折叠

        # 重新定位浮层标签
        self.overlay_state.move(10, 10)
        vw = self.image_viewer.width()
        self.overlay_inv.move(vw - 60, self.image_viewer.height() - 36)
        self.overlay_blink.move(vw - 100, self.image_viewer.height() - 36)
