"""可复制坐标标签 Widget

需求: 可疑目标列表显示目标的坐标，可鼠标选择并复制坐标
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel


class CoordinateLabel(QLabel):
    """可选择并复制文本的坐标标签"""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
        )
        self.setCursor(Qt.IBeamCursor)

    def set_coordinates(self, ra: str, dec: str) -> None:
        """设置显示的坐标"""
        self.setText(f"RA: {ra}  Dec: {dec}")

    def set_wcs_coordinates(self, ra: str, dec: str) -> None:
        """设置 WCS 坐标（兼容旧调用名）。"""
        self.set_coordinates(ra, dec)

    def set_pixel_coordinates(self, x: int, y: int) -> None:
        """设置像素坐标"""
        self.setText(f"X: {x}  Y: {y}")
