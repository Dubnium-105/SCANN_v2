"""Main window UI composition helpers."""

from __future__ import annotations

from dataclasses import dataclass, fields

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QAction,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMenu,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from scann.gui.image_viewer import FitsImageViewer
from scann.gui.widgets.blink_speed_slider import BlinkSpeedSlider
from scann.gui.widgets.collapsible_sidebar import CollapsibleSidebar
from scann.gui.widgets.coordinate_label import CoordinateLabel
from scann.gui.widgets.histogram_panel import HistogramPanel
from scann.gui.widgets.overlay_label import OverlayLabel
from scann.gui.widgets.suspect_table import SuspectTableWidget


@dataclass(frozen=True)
class MenuBarParts:
    act_open_new: QAction
    act_open_old: QAction
    act_save: QAction
    act_save_marked: QAction
    menu_recent: QMenu
    act_exit: QAction
    act_align: QAction
    act_batch_process: QAction
    act_histogram: QAction
    act_detect: QAction
    act_train: QAction
    act_load_model: QAction
    act_model_info: QAction
    act_annotation: QAction
    act_query_vsx: QAction
    act_query_mpc: QAction
    act_query_simbad: QAction
    act_query_tns: QAction
    act_query_satellite: QAction
    act_mpc_report: QAction
    act_toggle_sidebar: QAction
    act_fit_view: QAction
    act_zoom_actual: QAction
    act_zoom_in: QAction
    act_zoom_out: QAction
    act_show_markers: QAction
    act_show_mpcorb: QAction
    act_show_known: QAction
    act_preferences: QAction
    act_mpcorb_file: QAction
    act_scheduler: QAction
    act_shortcut_help: QAction
    act_docs: QAction
    act_about: QAction

    def attach(self, window: QMainWindow) -> None:
        for field_info in fields(self):
            setattr(window, field_info.name, getattr(self, field_info.name))


@dataclass(frozen=True)
class CentralUiParts:
    main_splitter: QSplitter
    sidebar: CollapsibleSidebar
    btn_new_folder: QPushButton
    btn_old_folder: QPushButton
    btn_align: QPushButton
    btn_detect: QPushButton
    progress_bar: QProgressBar
    file_list: QListWidget
    suspect_table: SuspectTableWidget
    image_viewer: FitsImageViewer
    overlay_state: OverlayLabel
    overlay_inv: OverlayLabel
    overlay_blink: OverlayLabel
    btn_show_new: QPushButton
    btn_show_old: QPushButton
    btn_blink: QPushButton
    blink_speed: BlinkSpeedSlider
    btn_invert: QPushButton
    btn_histogram: QPushButton
    btn_mark_real: QPushButton
    btn_mark_bogus: QPushButton
    btn_next_candidate: QPushButton

    def attach(self, window: QMainWindow) -> None:
        for field_info in fields(self):
            setattr(window, field_info.name, getattr(self, field_info.name))


@dataclass(frozen=True)
class StatusBarParts:
    status_image_type: QLabel
    status_pixel_coord: CoordinateLabel
    status_wcs_coord: CoordinateLabel
    status_zoom: QLabel

    def attach(self, window: QMainWindow) -> None:
        for field_info in fields(self):
            setattr(window, field_info.name, getattr(self, field_info.name))


@dataclass(frozen=True)
class MainWindowUiParts:
    menu: MenuBarParts
    central: CentralUiParts
    status: StatusBarParts
    histogram_panel: HistogramPanel

    def attach(self, window: QMainWindow) -> None:
        self.menu.attach(window)
        self.central.attach(window)
        self.status.attach(window)
        window.histogram_panel = self.histogram_panel


class MainWindowBuilder:
    """Build the static UI shell for MainWindow."""

    def __init__(self, window: QMainWindow):
        self._window = window

    def build(self) -> MainWindowUiParts:
        menu_parts = self._build_menu_bar()
        central_parts = self._build_central_ui()
        status_parts = self._build_status_bar()
        histogram_panel = self._build_histogram_dock()
        return MainWindowUiParts(
            menu=menu_parts,
            central=central_parts,
            status=status_parts,
            histogram_panel=histogram_panel,
        )

    def _build_menu_bar(self) -> MenuBarParts:
        mb = self._window.menuBar()

        file_menu = mb.addMenu("文件(&F)")

        act_open_new = file_menu.addAction("打开新图文件夹")
        act_open_new.setShortcut(QKeySequence("Ctrl+O"))

        act_open_old = file_menu.addAction("打开旧图文件夹")
        act_open_old.setShortcut(QKeySequence("Ctrl+Shift+O"))

        file_menu.addSeparator()

        act_save = file_menu.addAction("保存当前图像")
        act_save.setShortcut(QKeySequence("Ctrl+S"))

        act_save_marked = file_menu.addAction("另存为标记图...")
        act_save_marked.setShortcut(QKeySequence("Ctrl+Shift+S"))

        file_menu.addSeparator()

        menu_recent = file_menu.addMenu("最近打开")

        file_menu.addSeparator()
        act_exit = file_menu.addAction("退出")
        act_exit.setShortcut(QKeySequence("Alt+F4"))
        act_exit.triggered.connect(self._window.close)

        proc_menu = mb.addMenu("处理(&P)")
        act_align = proc_menu.addAction("批量对齐")
        proc_menu.addSeparator()
        act_batch_process = proc_menu.addAction("批量降噪/伪平场...")
        proc_menu.addSeparator()
        act_histogram = proc_menu.addAction("直方图拉伸")

        ai_menu = mb.addMenu("AI(&A)")
        act_detect = ai_menu.addAction("批量检测")
        act_detect.setShortcut(QKeySequence("F5"))
        ai_menu.addSeparator()
        act_train = ai_menu.addAction("训练模型...")
        act_load_model = ai_menu.addAction("加载模型...")
        act_model_info = ai_menu.addAction("模型信息")
        ai_menu.addSeparator()
        act_annotation = ai_menu.addAction("🏷️ 标注工具...")
        act_annotation.setShortcut(QKeySequence("Ctrl+L"))

        query_menu = mb.addMenu("查询(&Q)")
        act_query_vsx = query_menu.addAction("查询 VSX")
        act_query_mpc = query_menu.addAction("查询 MPC")
        act_query_simbad = query_menu.addAction("查询 SIMBAD")
        act_query_tns = query_menu.addAction("查询 TNS")
        act_query_satellite = query_menu.addAction("人造卫星查询")
        query_menu.addSeparator()
        act_mpc_report = query_menu.addAction("生成 MPC 80列报告")
        act_mpc_report.setShortcut(QKeySequence("Ctrl+E"))

        view_menu = mb.addMenu("视图(&V)")
        act_toggle_sidebar = view_menu.addAction("切换侧边栏")
        act_toggle_sidebar.setShortcut(QKeySequence("Ctrl+B"))

        view_menu.addSeparator()

        act_fit_view = view_menu.addAction("适配窗口")
        act_zoom_actual = view_menu.addAction("实际大小")
        act_zoom_actual.setShortcut(QKeySequence("Ctrl+0"))
        act_zoom_in = view_menu.addAction("放大")
        act_zoom_in.setShortcut(QKeySequence("Ctrl++"))
        act_zoom_out = view_menu.addAction("缩小")
        act_zoom_out.setShortcut(QKeySequence("Ctrl+-"))

        view_menu.addSeparator()

        act_show_markers = view_menu.addAction("显示候选标记")
        act_show_markers.setCheckable(True)
        act_show_markers.setChecked(True)

        act_show_mpcorb = view_menu.addAction("显示 MPCORB 叠加")
        act_show_mpcorb.setCheckable(True)
        act_show_mpcorb.setChecked(True)

        act_show_known = view_menu.addAction("显示已知天体")
        act_show_known.setCheckable(True)
        act_show_known.setChecked(True)

        settings_menu = mb.addMenu("设置(&S)")
        act_preferences = settings_menu.addAction("首选项...")
        act_preferences.setShortcut(QKeySequence("Ctrl+,"))
        settings_menu.addSeparator()
        act_mpcorb_file = settings_menu.addAction("MPCORB 文件...")
        act_scheduler = settings_menu.addAction("计划任务...")

        help_menu = mb.addMenu("帮助(&H)")
        act_shortcut_help = help_menu.addAction("快捷键列表")
        act_docs = help_menu.addAction("使用文档")
        help_menu.addSeparator()
        act_about = help_menu.addAction("关于 SCANN v2")

        return MenuBarParts(
            act_open_new=act_open_new,
            act_open_old=act_open_old,
            act_save=act_save,
            act_save_marked=act_save_marked,
            menu_recent=menu_recent,
            act_exit=act_exit,
            act_align=act_align,
            act_batch_process=act_batch_process,
            act_histogram=act_histogram,
            act_detect=act_detect,
            act_train=act_train,
            act_load_model=act_load_model,
            act_model_info=act_model_info,
            act_annotation=act_annotation,
            act_query_vsx=act_query_vsx,
            act_query_mpc=act_query_mpc,
            act_query_simbad=act_query_simbad,
            act_query_tns=act_query_tns,
            act_query_satellite=act_query_satellite,
            act_mpc_report=act_mpc_report,
            act_toggle_sidebar=act_toggle_sidebar,
            act_fit_view=act_fit_view,
            act_zoom_actual=act_zoom_actual,
            act_zoom_in=act_zoom_in,
            act_zoom_out=act_zoom_out,
            act_show_markers=act_show_markers,
            act_show_mpcorb=act_show_mpcorb,
            act_show_known=act_show_known,
            act_preferences=act_preferences,
            act_mpcorb_file=act_mpcorb_file,
            act_scheduler=act_scheduler,
            act_shortcut_help=act_shortcut_help,
            act_docs=act_docs,
            act_about=act_about,
        )

    def _build_central_ui(self) -> CentralUiParts:
        central = QWidget()
        self._window.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.setHandleWidth(6)

        sidebar = CollapsibleSidebar()
        sidebar_layout = sidebar.content_layout

        btn_layout = QHBoxLayout()
        btn_new_folder = QPushButton("📂 新图")
        btn_old_folder = QPushButton("📂 旧图")
        btn_layout.addWidget(btn_new_folder)
        btn_layout.addWidget(btn_old_folder)
        sidebar_layout.addLayout(btn_layout)

        func_layout = QHBoxLayout()
        btn_align = QPushButton("🔗 对齐")
        btn_detect = QPushButton("⚡ 检测")
        btn_detect.setStyleSheet(
            "QPushButton { background-color: #FFEB3B; color: #1E1E1E; font-weight: bold; }"
            "QPushButton:hover { background-color: #FFF176; }"
        )
        func_layout.addWidget(btn_align)
        func_layout.addWidget(btn_detect)
        sidebar_layout.addLayout(func_layout)

        progress_bar = QProgressBar()
        progress_bar.setVisible(False)
        progress_bar.setFixedHeight(16)
        sidebar_layout.addWidget(progress_bar)

        lbl_pairs = QLabel("📁 图像配对:")
        lbl_pairs.setStyleSheet("font-weight: bold;")
        sidebar_layout.addWidget(lbl_pairs)
        file_list = QListWidget()
        sidebar_layout.addWidget(file_list, 2)

        suspect_table = SuspectTableWidget()
        sidebar_layout.addWidget(suspect_table, 3)
        main_splitter.addWidget(sidebar)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        image_viewer = FitsImageViewer()
        right_layout.addWidget(image_viewer, 1)

        overlay_state = OverlayLabel("准备就绪", parent=image_viewer)
        overlay_state.move(10, 10)
        overlay_state.set_state("new")

        overlay_inv = OverlayLabel("INV", parent=image_viewer)
        overlay_inv.set_state("inv")
        overlay_inv.hide_label()

        overlay_blink = OverlayLabel("⚡", parent=image_viewer)
        overlay_blink.set_state("blink")
        overlay_blink.hide_label()

        ctrl_widget = QWidget()
        ctrl_widget.setFixedHeight(40)
        ctrl_widget.setStyleSheet("background-color: #252526; border-top: 1px solid #3C3C3C;")
        ctrl_layout = QHBoxLayout(ctrl_widget)
        ctrl_layout.setContentsMargins(4, 2, 4, 2)
        ctrl_layout.setSpacing(4)

        btn_show_new = QPushButton("[1] 新图")
        btn_show_old = QPushButton("[2] 旧图")
        btn_show_new.setCheckable(True)
        btn_show_old.setCheckable(True)
        btn_show_new.setChecked(True)
        ctrl_layout.addWidget(btn_show_new)
        ctrl_layout.addWidget(btn_show_old)

        sep1 = QLabel("|")
        sep1.setStyleSheet("color: #3C3C3C;")
        ctrl_layout.addWidget(sep1)

        btn_blink = QPushButton("✨ 闪烁 (R)")
        btn_blink.setCheckable(True)
        ctrl_layout.addWidget(btn_blink)

        blink_speed = BlinkSpeedSlider()
        ctrl_layout.addWidget(blink_speed)

        sep2 = QLabel("|")
        sep2.setStyleSheet("color: #3C3C3C;")
        ctrl_layout.addWidget(sep2)

        btn_invert = QPushButton("🔄 反色 (I)")
        btn_invert.setCheckable(True)
        ctrl_layout.addWidget(btn_invert)

        btn_histogram = QPushButton("📊 拉伸")
        ctrl_layout.addWidget(btn_histogram)

        ctrl_layout.addStretch()

        btn_mark_real = QPushButton("✅ 真 (Y)")
        btn_mark_real.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }"
            "QPushButton:hover { background-color: #66BB6A; }"
            "QPushButton:disabled { background-color: #2A2A2A; color: #555; }"
        )
        btn_mark_bogus = QPushButton("❌ 假 (N)")
        btn_mark_bogus.setStyleSheet(
            "QPushButton { background-color: #F44336; color: white; font-weight: bold; }"
            "QPushButton:hover { background-color: #EF5350; }"
            "QPushButton:disabled { background-color: #2A2A2A; color: #555; }"
        )
        btn_next_candidate = QPushButton("➡ 下一个")

        ctrl_layout.addWidget(btn_mark_real)
        ctrl_layout.addWidget(btn_mark_bogus)
        ctrl_layout.addWidget(btn_next_candidate)

        right_layout.addWidget(ctrl_widget)

        main_splitter.addWidget(right_panel)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes(
            [sidebar.preferred_width, max(1, self._window.width() - sidebar.preferred_width)]
        )
        splitter_moved_handler = getattr(self._window, "_on_main_splitter_moved", None)
        if splitter_moved_handler is not None:
            main_splitter.splitterMoved.connect(splitter_moved_handler)
        main_layout.addWidget(main_splitter, 1)

        return CentralUiParts(
            main_splitter=main_splitter,
            sidebar=sidebar,
            btn_new_folder=btn_new_folder,
            btn_old_folder=btn_old_folder,
            btn_align=btn_align,
            btn_detect=btn_detect,
            progress_bar=progress_bar,
            file_list=file_list,
            suspect_table=suspect_table,
            image_viewer=image_viewer,
            overlay_state=overlay_state,
            overlay_inv=overlay_inv,
            overlay_blink=overlay_blink,
            btn_show_new=btn_show_new,
            btn_show_old=btn_show_old,
            btn_blink=btn_blink,
            blink_speed=blink_speed,
            btn_invert=btn_invert,
            btn_histogram=btn_histogram,
            btn_mark_real=btn_mark_real,
            btn_mark_bogus=btn_mark_bogus,
            btn_next_candidate=btn_next_candidate,
        )

    def _build_status_bar(self) -> StatusBarParts:
        status_bar = QStatusBar()
        self._window.setStatusBar(status_bar)

        status_image_type = QLabel("准备就绪")
        status_image_type.setMinimumWidth(80)
        status_bar.addWidget(status_image_type)

        sep = QLabel("|")
        sep.setStyleSheet("color: rgba(255,255,255,0.3);")
        status_bar.addWidget(sep)

        status_pixel_coord = CoordinateLabel("X: --  Y: --")
        status_pixel_coord.setMinimumWidth(120)
        status_bar.addWidget(status_pixel_coord)

        sep2 = QLabel("|")
        sep2.setStyleSheet("color: rgba(255,255,255,0.3);")
        status_bar.addWidget(sep2)

        status_wcs_coord = CoordinateLabel("RA: --  Dec: --")
        status_wcs_coord.setMinimumWidth(200)
        status_bar.addWidget(status_wcs_coord)

        status_zoom = QLabel("100%")
        status_bar.addPermanentWidget(status_zoom)

        return StatusBarParts(
            status_image_type=status_image_type,
            status_pixel_coord=status_pixel_coord,
            status_wcs_coord=status_wcs_coord,
            status_zoom=status_zoom,
        )

    def _build_histogram_dock(self) -> HistogramPanel:
        histogram_panel = HistogramPanel(self._window)
        self._window.addDockWidget(Qt.BottomDockWidgetArea, histogram_panel)
        histogram_panel.setVisible(False)
        return histogram_panel