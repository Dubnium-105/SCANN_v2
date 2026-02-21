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
    MAX_WIDTH = 400

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._collapsed = False
        self._stored_width = self.DEFAULT_WIDTH

        self.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        self.setMinimumWidth(self.MIN_WIDTH)
        self.setMaximumWidth(self.MAX_WIDTH)

        # 内容布局
        self._content_layout = QVBoxLayout(self)
        self._content_layout.setContentsMargins(2, 2, 2, 2)
        self._content_layout.setSpacing(4)

        # 动画
        self._animation = QPropertyAnimation(self, b"maximumWidth")
        self._animation.setDuration(200)

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

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

    @property
    def preferred_width(self) -> int:
        return self._stored_width

    def set_preferred_width(self, width: int) -> None:
        """记录用户拖动后的侧边栏宽度。"""
        width = max(self.MIN_WIDTH, min(int(width), self.MAX_WIDTH))
        self._stored_width = width

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
        if self.width() > 0:
            self._stored_width = max(self.MIN_WIDTH, min(self.width(), self.MAX_WIDTH))

        self.setMinimumWidth(0)

        self._animation.setStartValue(self._stored_width)
        self._animation.setEndValue(0)
        self._animation.start()
        self.collapsed_changed.emit(True)

    def expand(self) -> None:
        """展开侧边栏"""
        if not self._collapsed:
            return
        self._collapsed = False

        self.setMinimumWidth(self.MIN_WIDTH)
        self.setMaximumWidth(self.MAX_WIDTH)

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

    def resizeEvent(self, event) -> None:
        """在展开状态下记住当前宽度（用于下次折叠后恢复）。"""
        super().resizeEvent(event)
        if not self._collapsed and self.width() >= self.MIN_WIDTH:
            self._stored_width = min(self.width(), self.MAX_WIDTH)
