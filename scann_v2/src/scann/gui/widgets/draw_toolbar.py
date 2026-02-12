"""绘制工具栏

v2 FITS 标注模式下的工具切换：框选 (B) / 点标 (Q) / 移动 (V)
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QPushButton,
    QWidget,
)


class DrawToolBar(QWidget):
    """绘制工具栏

    Signals:
        tool_changed(str): 工具切换 ("box" / "point" / "move")
    """

    tool_changed = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(True)

        self._btn_box = QPushButton("□ 框选 (B)")
        self._btn_box.setCheckable(True)
        self._btn_point = QPushButton("○ 点标 (Q)")
        self._btn_point.setCheckable(True)
        self._btn_move = QPushButton("✋ 移动 (V)")
        self._btn_move.setCheckable(True)
        self._btn_move.setChecked(True)  # 默认移动模式

        self._btn_group.addButton(self._btn_box)
        self._btn_group.addButton(self._btn_point)
        self._btn_group.addButton(self._btn_move)

        layout.addWidget(self._btn_box)
        layout.addWidget(self._btn_point)
        layout.addWidget(self._btn_move)
        layout.addStretch()

        self._btn_box.clicked.connect(lambda: self.tool_changed.emit("box"))
        self._btn_point.clicked.connect(lambda: self.tool_changed.emit("point"))
        self._btn_move.clicked.connect(lambda: self.tool_changed.emit("move"))

    def set_tool(self, tool: str) -> None:
        """程序化切换工具"""
        mapping = {"box": self._btn_box, "point": self._btn_point, "move": self._btn_move}
        btn = mapping.get(tool)
        if btn:
            btn.setChecked(True)
            self.tool_changed.emit(tool)
