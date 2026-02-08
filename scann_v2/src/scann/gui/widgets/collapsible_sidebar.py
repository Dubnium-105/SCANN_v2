"""可折叠侧边栏

需求:
- Ctrl+B 切换显示/隐藏
- 默认宽度 240px，最小 200px
- 窗口宽度 < 1200px 时自动折叠
"""

from __future__ import annotations

from PyQt5.QtCore import QPropertyAnimation, Qt, pyqtSignal
from PyQt5.QtWidgets import QFrame, QSizePolicy, QVBoxLayout, QWidget


class CollapsibleSidebar(QFrame):
    """可折叠侧边栏

    Signals:
        collapsed_changed(bool): 折叠状态变化
    """

    collapsed_changed = pyqtSignal(bool)

    DEFAULT_WIDTH = 240
    MIN_WIDTH = 200

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._collapsed = False
        self._stored_width = self.DEFAULT_WIDTH

        self.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        self.setMinimumWidth(0)  # 折叠时可到 0
        self.setMaximumWidth(400)

        # 内容布局
        self._content_layout = QVBoxLayout(self)
        self._content_layout.setContentsMargins(2, 2, 2, 2)
        self._content_layout.setSpacing(4)

        # 动画
        self._animation = QPropertyAnimation(self, b"maximumWidth")
        self._animation.setDuration(200)

        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setFixedWidth(self.DEFAULT_WIDTH)

        # 暗色主题
        self.setStyleSheet(
            "CollapsibleSidebar {"
            "  background-color: #252526;"
            "  border-right: 1px solid #3C3C3C;"
            "}"
        )

    @property
    def content_layout(self) -> QVBoxLayout:
        """获取内容布局以添加子组件"""
        return self._content_layout

    @property
    def is_collapsed(self) -> bool:
        return self._collapsed

    def toggle(self) -> None:
        """切换折叠/展开"""
        if self._collapsed:
            self.expand()
        else:
            self.collapse()

    def collapse(self) -> None:
        """折叠侧边栏"""
        if self._collapsed:
            return
        self._collapsed = True
        self._stored_width = self.width()

        self._animation.setStartValue(self._stored_width)
        self._animation.setEndValue(0)
        self._animation.start()
        self.collapsed_changed.emit(True)

    def expand(self) -> None:
        """展开侧边栏"""
        if not self._collapsed:
            return
        self._collapsed = False

        self._animation.setStartValue(0)
        self._animation.setEndValue(self._stored_width)
        self._animation.start()
        self.collapsed_changed.emit(False)

    def auto_collapse_check(self, window_width: int) -> None:
        """根据窗口宽度自动折叠/展开

        Args:
            window_width: 当前窗口宽度
        """
        if window_width < 1200 and not self._collapsed:
            self.collapse()
        elif window_width >= 1200 and self._collapsed:
            self.expand()
