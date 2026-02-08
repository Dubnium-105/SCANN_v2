"""禁用滚轮的 SpinBox

需求: 所有输入窗口禁用滚轮调整数字大小！
"""

from PyQt5.QtCore import QEvent
from PyQt5.QtWidgets import QDoubleSpinBox, QSpinBox


class NoScrollSpinBox(QSpinBox):
    """禁用滚轮调整的 SpinBox"""

    def wheelEvent(self, event):
        event.ignore()  # 不处理滚轮事件，让父组件处理


class NoScrollDoubleSpinBox(QDoubleSpinBox):
    """禁用滚轮调整的 DoubleSpinBox"""

    def wheelEvent(self, event):
        event.ignore()
