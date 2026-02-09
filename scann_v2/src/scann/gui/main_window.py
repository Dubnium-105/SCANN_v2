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
    QProgressBar,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from scann.core.models import Candidate, TargetVerdict
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
        self.setWindowTitle("SCANN v2 - Star/Source Classification and Analysis Neural Network")
        self.resize(1600, 1000)
        self.setMinimumSize(1024, 768)

        # 暗色主题
        self.setStyleSheet(DARK_THEME_QSS)

        # ── 服务 ──
        self.blink_service = BlinkService(speed_ms=500)

        # ── 定时器 ──
        self.blink_timer = QTimer(self)
        self.blink_timer.timeout.connect(self._on_blink_tick)

        # ── 数据状态 ──
        self._candidates: list[Candidate] = []
        self._current_candidate_idx: int = -1
        self._new_image_data: Optional[np.ndarray] = None
        self._old_image_data: Optional[np.ndarray] = None

        # ── 构建 UI ──
        self._init_menu_bar()
        self._init_central_ui()
        self._init_status_bar()
        self._init_histogram_dock()
        self._connect_signals()
        self._init_shortcuts()

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

        main_layout.addWidget(self.sidebar)

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
        main_layout.addWidget(right_panel, 1)

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

    def _on_blink_toggle(self) -> None:
        """切换闪烁"""
        running = self.blink_service.toggle()
        self.btn_blink.setChecked(running)
        if running:
            self.blink_timer.setInterval(self.blink_service.speed_ms)
            self.blink_timer.start()
            self.overlay_blink.show_label()
            self.overlay_blink.start_pulse()
        else:
            self.blink_timer.stop()
            self.overlay_blink.stop_pulse()
            self.overlay_blink.hide_label()

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
        self._show_image("new")

    def _on_show_old(self) -> None:
        """显示旧图"""
        self.btn_show_new.setChecked(False)
        self.btn_show_old.setChecked(True)
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

        self.image_viewer.set_image_data(
            data, inverted=self.blink_service.is_inverted
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
        self.statusBar().showMessage(
            f"候选 #{self._current_candidate_idx + 1} → 真目标", 3000
        )

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
        self.statusBar().showMessage(
            f"候选 #{self._current_candidate_idx + 1} → 假目标", 3000
        )

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
        # TODO: 通过 ImageProcessor 对当前图像执行线性拉伸
        #       使用 black/white 点映射像素范围，刷新 image_viewer 显示
        pass

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
        self.statusBar().showMessage(f"正在查询 {query_type} ({x}, {y})...", 5000)
        # TODO: 通过 QueryService 实现远程查询
        #       1. 将像素坐标 (x, y) 转换为天球坐标 (RA, Dec)
        #       2. 调用 QueryService.query_{query_type}(ra, dec)
        #       3. 将结果展示在弹出窗口或侧边栏中

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
    #  TODO: 待完成的菜单 / 按钮处理方法
    # ══════════════════════════════════════════════

    # ── 文件菜单 ──

    def _on_open_new_folder(self) -> None:
        """打开新图文件夹"""
        # TODO: 加载文件夹中的 FITS 文件到 file_list，
        #       为每个文件创建配对，并设置 _new_image_data
        folder = QFileDialog.getExistingDirectory(self, "选择新图文件夹")
        if folder:
            self.statusBar().showMessage(f"已选择新图文件夹: {folder}", 3000)

    def _on_open_old_folder(self) -> None:
        """打开旧图文件夹"""
        # TODO: 加载文件夹中的 FITS 文件，
        #       与新图配对并设置 _old_image_data
        folder = QFileDialog.getExistingDirectory(self, "选择旧图文件夹")
        if folder:
            self.statusBar().showMessage(f"已选择旧图文件夹: {folder}", 3000)

    def _on_save_image(self) -> None:
        """保存当前图像"""
        # TODO: 通过 FitsIO 将当前显示的图像数据保存为 FITS 文件
        self.statusBar().showMessage("TODO: 保存当前图像", 3000)

    def _on_save_marked_image(self) -> None:
        """另存为带标记的图像"""
        # TODO: 将当前图像连同候选标记一起导出为 PNG/FITS
        path, _ = QFileDialog.getSaveFileName(
            self, "另存为标记图", "", "PNG (*.png);;FITS (*.fits)"
        )
        if path:
            self.statusBar().showMessage(f"TODO: 保存标记图到 {path}", 3000)

    def _on_update_recent_menu(self) -> None:
        """更新最近打开菜单"""
        # TODO: 从 AppConfig 读取最近打开的文件夹列表，
        #       填充 menu_recent 子菜单项并连接点击事件
        self.menu_recent.clear()
        self.menu_recent.addAction("(无最近打开)")

    # ── 处理菜单 ──

    def _on_batch_align(self) -> None:
        """批量对齐"""
        # TODO: 调用 ImageAligner 对当前文件夹中的图像进行批量对齐
        self.statusBar().showMessage("TODO: 批量对齐 — 需要集成 ImageAligner", 3000)

    def _on_batch_process(self) -> None:
        """打开批量处理对话框"""
        # TODO: 打开 BatchProcessDialog，获取参数后调用 ImageProcessor
        from scann.gui.dialogs.batch_process_dialog import BatchProcessDialog
        dlg = BatchProcessDialog(self)
        dlg.exec_()

    # ── AI 菜单 ──

    def _on_batch_detect(self) -> None:
        """批量检测"""
        # TODO: 调用 DetectionService 对当前图像配对执行 AI 检测，
        #       将结果通过 set_candidates() 设置到界面
        self.statusBar().showMessage("TODO: 批量检测 — 需要集成 DetectionService", 3000)

    def _on_open_training(self) -> None:
        """打开训练对话框"""
        # TODO: 打开 TrainingDialog，配置并启动模型训练
        from scann.gui.dialogs.training_dialog import TrainingDialog
        dlg = TrainingDialog(self)
        dlg.exec_()

    def _on_load_model(self) -> None:
        """加载 AI 模型"""
        # TODO: 通过 InferenceEngine 加载 .pth 模型文件
        path, _ = QFileDialog.getOpenFileName(
            self, "加载模型", "", "PyTorch 模型 (*.pth *.pt)"
        )
        if path:
            self.statusBar().showMessage(f"TODO: 加载模型 {path}", 3000)

    def _on_model_info(self) -> None:
        """显示模型信息"""
        # TODO: 显示当前已加载模型的架构、参数量、训练信息
        self.statusBar().showMessage("TODO: 模型信息 — 需要 InferenceEngine 提供元数据", 3000)

    # ── 查询菜单 ──

    def _on_menu_query(self, query_type: str) -> None:
        """从菜单栏触发的查询 (无坐标上下文)"""
        # TODO: 若有选中候选体则使用其坐标查询，否则提示用户
        if self._candidates and 0 <= self._current_candidate_idx < len(self._candidates):
            cand = self._candidates[self._current_candidate_idx]
            self._do_query(query_type, int(cand.x), int(cand.y))
        else:
            self.statusBar().showMessage(
                "请先选中一个候选体，或在图像上右键进行坐标查询", 3000
            )

    def _on_mpc_report(self) -> None:
        """打开 MPC 80列报告对话框"""
        # TODO: 传入当前候选列表和观测信息
        from scann.gui.dialogs.mpc_report_dialog import MpcReportDialog
        dlg = MpcReportDialog(self)
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
        # TODO: 根据 checked 状态显示/隐藏 MPCORB 小行星轨道叠加层
        self.statusBar().showMessage(
            f"MPCORB 叠加: {'开启' if checked else '关闭'}", 2000
        )

    def _on_toggle_known(self, checked: bool) -> None:
        """切换已知天体显示"""
        # TODO: 根据 checked 状态显示/隐藏已知天体 (变星、小行星等) 标记
        self.statusBar().showMessage(
            f"已知天体标记: {'开启' if checked else '关闭'}", 2000
        )

    # ── 设置菜单 ──

    def _on_open_preferences(self) -> None:
        """打开首选项对话框"""
        # TODO: 保存用户修改后重新加载配置
        from scann.gui.dialogs.settings_dialog import SettingsDialog
        dlg = SettingsDialog(self)
        if dlg.exec_():
            self.statusBar().showMessage("设置已保存", 3000)

    def _on_select_mpcorb_file(self) -> None:
        """选择 MPCORB 数据文件"""
        # TODO: 更新配置并通过 MpcorbParser 重新加载小行星数据
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 MPCORB 文件", "", "DAT 文件 (*.dat);;所有文件 (*)"
        )
        if path:
            self.statusBar().showMessage(f"TODO: 加载 MPCORB 文件 {path}", 3000)

    def _on_open_scheduler(self) -> None:
        """打开计划任务设置"""
        # TODO: 实现计划任务管理界面 (定时检测、自动下载等)
        self.statusBar().showMessage("TODO: 计划任务 — 功能待设计", 3000)

    # ── 帮助菜单 ──

    def _on_shortcut_help(self) -> None:
        """显示快捷键帮助对话框"""
        from scann.gui.dialogs.shortcut_help_dialog import ShortcutHelpDialog
        dlg = ShortcutHelpDialog(self)
        dlg.exec_()

    def _on_open_docs(self) -> None:
        """打开使用文档"""
        # TODO: 替换为项目实际文档 URL
        import webbrowser
        webbrowser.open("https://github.com/your-repo/scann-v2/wiki")

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
        # TODO: 若已加载 WCS 头信息，同步更新天球坐标
        # wcs_coord = self._pixel_to_wcs(x, y)
        # if wcs_coord:
        #     self.status_wcs_coord.set_wcs_coordinates(*wcs_coord)

    def _on_zoom_changed(self, zoom_pct: float) -> None:
        """缩放比例变化 → 更新状态栏"""
        self.status_zoom.setText(f"{zoom_pct:.0f}%")

    # ── 右键上下文菜单处理 ──

    def _on_context_mpc_report(self, x: int, y: int) -> None:
        """右键菜单 → 生成 MPC 报告"""
        # TODO: 使用点击坐标定位候选体后打开 MPC 报告对话框
        self._on_mpc_report()

    def _on_context_add_candidate(self, x: int, y: int) -> None:
        """右键菜单 → 手动添加候选体"""
        # TODO: 在 (x, y) 位置创建手动候选体，添加到 _candidates 列表
        self.statusBar().showMessage(f"TODO: 在 ({x}, {y}) 添加手动候选体", 3000)

    def _on_copy_wcs_coordinates(self, x: int, y: int) -> None:
        """右键菜单 → 复制天球坐标"""
        # TODO: 将像素 (x, y) 通过 Astrometry 转换为 RA/Dec 并复制到剪贴板
        self.statusBar().showMessage("TODO: 复制天球坐标 — 需要 WCS 信息", 3000)

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

    def resizeEvent(self, event) -> None:
        """窗口大小变化 → 自动折叠侧边栏"""
        super().resizeEvent(event)
        self.sidebar.auto_collapse_check(self.width())

        # 重新定位浮层标签
        self.overlay_state.move(10, 10)
        vw = self.image_viewer.width()
        self.overlay_inv.move(vw - 60, self.image_viewer.height() - 36)
        self.overlay_blink.move(vw - 100, self.image_viewer.height() - 36)
