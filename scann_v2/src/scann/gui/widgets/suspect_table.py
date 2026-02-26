"""可疑目标表格 Widget

需求:
- 显示 AI 评分、像素坐标、天球坐标、判决状态
- 按 AI 评分排序
- 坐标可鼠标选择并复制
- 右键弹出外部查询菜单 (VSX/MPC/SIMBAD/TNS/人造卫星/MPC报告)
- 单击选中 → 图像居中; 双击 → 放大
"""

from __future__ import annotations

from typing import List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QAction,
    QApplication,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from scann.core.models import Candidate, TargetVerdict


class SuspectTableWidget(QWidget):
    """可疑目标表格

    Signals:
        candidate_selected(int): 单击选中候选体 (索引)
        candidate_double_clicked(int): 双击候选体 (索引)
        query_requested(str, int, int): 查询请求 (类型, x, y)
        mpc_report_requested(int): MPC 报告请求 (索引)
        copy_coordinates_requested(int): 复制坐标请求 (索引)
    """

    candidate_selected = pyqtSignal(int)
    candidate_double_clicked = pyqtSignal(int)
    query_requested = pyqtSignal(str, int, int)  # query_type, x, y
    mpc_report_requested = pyqtSignal(int)
    copy_coordinates_requested = pyqtSignal(int)

    # 列定义
    COL_INDEX = 0
    COL_SCORE = 1
    COL_PIXEL = 2
    COL_WCS = 3
    COL_VERDICT = 4
    NUM_COLS = 5

    HEADERS = ["#", "AI 评分", "像素坐标", "WCS 坐标", "判决"]

    # 判决显示映射
    VERDICT_DISPLAY = {
        TargetVerdict.REAL: ("✅ 真", QColor("#4CAF50")),
        TargetVerdict.BOGUS: ("❌ 假", QColor("#F44336")),
        TargetVerdict.UNKNOWN: ("──", QColor("#808080")),
    }

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._candidates: List[Candidate] = []
        self._visible_indices: List[int] = []
        self._ai_filter_threshold: float = 0.0

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # 标题行
        header_layout = QHBoxLayout()
        title = QLabel("🔥 可疑目标 (AI 排序)")
        title.setFont(QFont("", -1, QFont.Bold))
        header_layout.addWidget(title)
        header_layout.addStretch()

        lbl_filter = QLabel("AI过滤阈值")
        header_layout.addWidget(lbl_filter)

        self.spin_ai_threshold = QDoubleSpinBox()
        self.spin_ai_threshold.setRange(0.0, 1.0)
        self.spin_ai_threshold.setSingleStep(0.05)
        self.spin_ai_threshold.setDecimals(2)
        self.spin_ai_threshold.setValue(0.0)
        self.spin_ai_threshold.setFixedHeight(24)
        self.spin_ai_threshold.setFixedWidth(72)
        self.spin_ai_threshold.setToolTip("仅显示 AI 评分大于等于该阈值的候选体")
        header_layout.addWidget(self.spin_ai_threshold)
        layout.addLayout(header_layout)

        # 表格
        self.table = QTableWidget()
        self.table.setColumnCount(self.NUM_COLS)
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)

        # 列宽
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setMinimumSectionSize(30)
        header.resizeSection(self.COL_INDEX, 40)
        header.resizeSection(self.COL_SCORE, 90)
        header.resizeSection(self.COL_PIXEL, 140)
        header.resizeSection(self.COL_WCS, 180)
        header.resizeSection(self.COL_VERDICT, 80)

        layout.addWidget(self.table)

        # 坐标复制区
        coord_layout = QHBoxLayout()
        self.lbl_coord = QLabel("📋 坐标: --")
        self.lbl_coord.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
        )
        self.lbl_coord.setCursor(Qt.IBeamCursor)
        coord_layout.addWidget(self.lbl_coord, 1)

        self.btn_copy = QPushButton("复制")
        self.btn_copy.setFixedWidth(50)
        self.btn_copy.setFixedHeight(24)
        coord_layout.addWidget(self.btn_copy)
        layout.addLayout(coord_layout)

        # 信号连接
        self.table.cellClicked.connect(self._on_cell_clicked)
        self.table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        self.table.customContextMenuRequested.connect(self._on_context_menu)
        self.btn_copy.clicked.connect(self._on_copy_coord)
        self.spin_ai_threshold.valueChanged.connect(self._on_threshold_changed)

        # 暗色主题样式
        self.table.setStyleSheet(
            "QTableWidget { background-color: #252526; color: #D4D4D4; "
            "  gridline-color: #3C3C3C; }"
            "QTableWidget::item { background-color: #252526; }"
            "QTableWidget::item:alternate { background-color: #2D2D2D; }"
            "QTableWidget::item:selected { background-color: #094771; }"
            "QHeaderView::section { background-color: #333333; color: #D4D4D4; "
            "  border: 1px solid #3C3C3C; padding: 2px; }"
        )

    def set_candidates(self, candidates: List[Candidate]) -> None:
        """设置候选体列表 (已按 AI 评分排序)"""
        self._candidates = candidates
        self._refresh_table()

    def update_candidate(self, index: int) -> None:
        """更新单个候选体的显示 (例如判决更新后)"""
        if 0 <= index < len(self._candidates):
            # 阈值过滤开启时，候选体可见性可能变化，统一全量刷新更稳妥
            self._refresh_table()

    def _on_threshold_changed(self, value: float) -> None:
        """AI 过滤阈值变化"""
        self._ai_filter_threshold = float(value)
        self._refresh_table()

    def _refresh_table(self) -> None:
        """刷新整个表格"""
        self._visible_indices = [
            idx
            for idx, cand in enumerate(self._candidates)
            if cand.ai_score >= self._ai_filter_threshold
        ]
        self.table.setRowCount(len(self._visible_indices))
        for row, src_idx in enumerate(self._visible_indices):
            self._update_row(row, self._candidates[src_idx])

    def _update_row(self, row: int, cand: Candidate) -> None:
        """更新某一行"""
        # #
        item_idx = QTableWidgetItem(str(row + 1))
        item_idx.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, self.COL_INDEX, item_idx)

        # AI 评分
        score_text = f"⭐ {cand.ai_score:.2f}"
        item_score = QTableWidgetItem(score_text)
        item_score.setTextAlignment(Qt.AlignCenter)
        # 高分高亮
        if cand.ai_score >= 0.9:
            item_score.setForeground(QColor("#FFEB3B"))
        elif cand.ai_score >= 0.7:
            item_score.setForeground(QColor("#4CAF50"))
        self.table.setItem(row, self.COL_SCORE, item_score)

        # 像素坐标
        item_pixel = QTableWidgetItem(f"({cand.x}, {cand.y})")
        item_pixel.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, self.COL_PIXEL, item_pixel)

        # WCS 坐标 (暂用占位)
        item_wcs = QTableWidgetItem("--")
        item_wcs.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, self.COL_WCS, item_wcs)

        # 判决
        verdict = cand.verdict if hasattr(cand, "verdict") else TargetVerdict.UNKNOWN
        display_text, color = self.VERDICT_DISPLAY.get(
            verdict, ("──", QColor("#808080"))
        )
        item_verdict = QTableWidgetItem(display_text)
        item_verdict.setTextAlignment(Qt.AlignCenter)
        item_verdict.setForeground(color)
        self.table.setItem(row, self.COL_VERDICT, item_verdict)

        # 已知天体行灰色
        if cand.is_known:
            for col in range(self.NUM_COLS):
                item = self.table.item(row, col)
                if item:
                    item.setForeground(QColor("#757575"))

    def _on_cell_clicked(self, row: int, _col: int) -> None:
        if 0 <= row < len(self._visible_indices):
            src_idx = self._visible_indices[row]
            cand = self._candidates[src_idx]
            self.lbl_coord.setText(f"📋 坐标: X={cand.x}  Y={cand.y}")
            self.candidate_selected.emit(src_idx)

    def _on_cell_double_clicked(self, row: int, _col: int) -> None:
        if 0 <= row < len(self._visible_indices):
            self.candidate_double_clicked.emit(self._visible_indices[row])

    def _on_copy_coord(self) -> None:
        row = self.table.currentRow()
        if 0 <= row < len(self._visible_indices):
            src_idx = self._visible_indices[row]
            cand = self._candidates[src_idx]
            text = f"{cand.x}, {cand.y}"
            QApplication.clipboard().setText(text)
            self.copy_coordinates_requested.emit(src_idx)

    def _on_context_menu(self, pos) -> None:
        """右键上下文菜单"""
        row = self.table.rowAt(pos.y())
        if row < 0 or row >= len(self._visible_indices):
            return

        src_idx = self._visible_indices[row]
        cand = self._candidates[src_idx]
        menu = QMenu(self)

        # 查询菜单
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
                lambda checked, t=qtype, x=cand.x, y=cand.y: self.query_requested.emit(t, x, y)
            )

        menu.addSeparator()
        act_report = menu.addAction("📝 生成 MPC 80列报告")
        act_report.triggered.connect(lambda: self.mpc_report_requested.emit(src_idx))

        menu.addSeparator()
        act_copy_pixel = menu.addAction("📋 复制像素坐标")
        act_copy_pixel.triggered.connect(
            lambda: QApplication.clipboard().setText(f"{cand.x}, {cand.y}")
        )
        act_copy_wcs = menu.addAction("📋 复制天球坐标")
        act_copy_wcs.triggered.connect(
            lambda: QApplication.clipboard().setText(
                self.table.item(row, self.COL_WCS).text()
            )
        )

        menu.exec_(self.table.viewport().mapToGlobal(pos))

    @property
    def selected_index(self) -> int:
        """当前选中行索引"""
        row = self.table.currentRow()
        if 0 <= row < len(self._visible_indices):
            return self._visible_indices[row]
        return -1
