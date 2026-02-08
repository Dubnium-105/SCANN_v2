"""主窗口

需求:
- 界面紧凑，留更大空间显示原图
- 两个按钮显示新旧图
- 可疑目标列表 + AI 评分 + 可复制坐标
- 快捷键: r=闪烁, n=假, y=真, 滚轮=缩放, i=反色
- 快捷键非全局，窗口焦点在程序内才有效
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QAction,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from scann.gui.image_viewer import FitsImageViewer
from scann.gui.widgets.no_scroll_spinbox import NoScrollDoubleSpinBox, NoScrollSpinBox
from scann.services.blink_service import BlinkService, BlinkState


class MainWindow(QMainWindow):
    """SCANN v2 主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SCANN v2 - Star/Source Classification and Analysis Neural Network")
        self.resize(1600, 1000)

        # 服务
        self.blink_service = BlinkService(speed_ms=500)

        # 定时器
        self.blink_timer = QTimer(self)
        self.blink_timer.timeout.connect(self._on_blink_tick)

        self._init_ui()
        self._init_shortcuts()

    def _init_ui(self) -> None:
        """初始化界面"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # 主分割器 (左侧面板 | 图像区域)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # ── 左侧面板 (紧凑) ──
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(2, 2, 2, 2)
        left_layout.setSpacing(4)

        # 文件夹按钮
        btn_layout = QHBoxLayout()
        self.btn_new_folder = QPushButton("📂 新图文件夹")
        self.btn_old_folder = QPushButton("📂 旧图文件夹")
        btn_layout.addWidget(self.btn_new_folder)
        btn_layout.addWidget(self.btn_old_folder)
        left_layout.addLayout(btn_layout)

        # 功能按钮
        self.btn_align = QPushButton("🔗 批量对齐")
        self.btn_detect = QPushButton("⚡ 批量检测")
        self.btn_detect.setStyleSheet("background-color: #ffeb3b; font-weight: bold;")
        left_layout.addWidget(self.btn_align)
        left_layout.addWidget(self.btn_detect)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        # 文件列表
        left_layout.addWidget(QLabel("图像配对列表:"))
        self.file_list = QListWidget()
        left_layout.addWidget(self.file_list, 2)

        # 可疑目标列表
        left_layout.addWidget(QLabel("🔥 可疑目标 (AI 排序):"))
        self.suspect_list = QListWidget()
        left_layout.addWidget(self.suspect_list, 1)

        splitter.addWidget(left_panel)

        # ── 右侧图像区域 (最大化) ──
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(2, 2, 2, 2)
        right_layout.setSpacing(4)

        # 标题
        self.lbl_title = QLabel("准备就绪")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.lbl_title)

        # 图像查看器 (占据最大空间)
        self.image_viewer = FitsImageViewer()
        right_layout.addWidget(self.image_viewer, 1)

        # 底部控制栏
        ctrl_layout = QHBoxLayout()

        self.btn_show_new = QPushButton("[1] 新图")
        self.btn_show_old = QPushButton("[2] 旧图")
        self.btn_blink = QPushButton("✨ 闪烁 (R)")
        self.btn_blink.setCheckable(True)
        self.btn_invert = QPushButton("🔄 反色 (I)")
        self.btn_invert.setCheckable(True)
        self.btn_mark_real = QPushButton("✅ 真 (Y)")
        self.btn_mark_real.setStyleSheet("background-color: #4CAF50; color: white;")
        self.btn_mark_bogus = QPushButton("❌ 假 (N)")
        self.btn_mark_bogus.setStyleSheet("background-color: #f44336; color: white;")

        ctrl_layout.addWidget(self.btn_show_new)
        ctrl_layout.addWidget(self.btn_show_old)
        ctrl_layout.addWidget(self.btn_blink)
        ctrl_layout.addWidget(self.btn_invert)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.btn_mark_real)
        ctrl_layout.addWidget(self.btn_mark_bogus)

        right_layout.addLayout(ctrl_layout)

        splitter.addWidget(right_panel)

        # 分割比例: 左侧 1/4, 右侧 3/4
        splitter.setSizes([300, 900])

        # 状态栏
        self.setStatusBar(QStatusBar())

        # 连接信号
        self.btn_blink.clicked.connect(self._on_blink_toggle)
        self.btn_invert.clicked.connect(self._on_invert_toggle)

    def _init_shortcuts(self) -> None:
        """初始化快捷键 (非全局，仅窗口焦点内)"""
        shortcuts = {
            "R": self._on_blink_toggle,
            "I": self._on_invert_toggle,
            "Y": self._on_mark_real,
            "N": self._on_mark_bogus,
            "1": self._on_show_new,
            "2": self._on_show_old,
        }
        for key, handler in shortcuts.items():
            action = QAction(self)
            action.setShortcut(key)
            action.setShortcutContext(Qt.WindowShortcut)  # 非全局
            action.triggered.connect(handler)
            self.addAction(action)

    # ─── 事件处理 ───

    def _on_blink_toggle(self) -> None:
        running = self.blink_service.toggle()
        self.btn_blink.setChecked(running)
        if running:
            self.blink_timer.setInterval(self.blink_service.speed_ms)
            self.blink_timer.start()
        else:
            self.blink_timer.stop()

    def _on_blink_tick(self) -> None:
        state = self.blink_service.tick()
        # TODO: 根据 state 切换显示的图像

    def _on_invert_toggle(self) -> None:
        inverted = self.blink_service.toggle_invert()
        self.btn_invert.setChecked(inverted)
        # TODO: 刷新当前显示

    def _on_mark_real(self) -> None:
        # TODO: 标记当前候选为真目标
        pass

    def _on_mark_bogus(self) -> None:
        # TODO: 标记当前候选为假目标
        pass

    def _on_show_new(self) -> None:
        # TODO: 显示新图
        pass

    def _on_show_old(self) -> None:
        # TODO: 显示旧图
        pass
