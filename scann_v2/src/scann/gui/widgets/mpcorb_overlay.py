"""MPCORB 已知小行星叠加绘制层

在图像上绘制 MPCORB 文件中已知小行星的位置，
使用虚线圆 + 名称标签，青色显示。
"""

from __future__ import annotations

from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont, QPen
from PyQt5.QtWidgets import QGraphicsEllipseItem, QGraphicsScene, QGraphicsTextItem

from scann.core.models import SkyPosition


class MpcorbOverlay:
    """MPCORB 已知小行星叠加层

    管理在 QGraphicsScene 上绘制的已知小行星标记。
    使用虚线圆 + 名称标签，不干扰候选体标记。

    用法:
        overlay = MpcorbOverlay(scene)
        overlay.set_positions(positions, wcs_to_pixel_func)
        overlay.set_visible(True)
    """

    COLOR = QColor("#00BCD4")       # 青色
    RADIUS = 20                      # 圆圈半径
    PEN_WIDTH = 1
    Z_VALUE = 5                      # 低于候选体标记 (z=10)

    def __init__(self, scene: QGraphicsScene):
        self._scene = scene
        self._items: List = []
        self._visible = True

    def set_positions(
        self,
        positions: List[SkyPosition],
        wcs_to_pixel: callable,
    ) -> None:
        """设置已知小行星位置并绘制

        Args:
            positions: 已知小行星天球坐标列表
            wcs_to_pixel: WCS→像素坐标转换函数 (ra, dec) -> (x, y)
        """
        self.clear()
        font = QFont("Arial", 8)
        pen = QPen(self.COLOR, self.PEN_WIDTH, Qt.DashLine)

        for pos in positions:
            try:
                px, py = wcs_to_pixel(pos.ra, pos.dec)
            except Exception:
                continue

            # 虚线圆
            ellipse = self._scene.addEllipse(
                px - self.RADIUS, py - self.RADIUS,
                self.RADIUS * 2, self.RADIUS * 2,
                pen,
            )
            ellipse.setZValue(self.Z_VALUE)
            ellipse.setVisible(self._visible)
            self._items.append(ellipse)

            # 名称标签
            label_text = pos.name
            if pos.mag is not None:
                label_text += f" ({pos.mag:.1f})"
            text = self._scene.addText(label_text, font)
            text.setDefaultTextColor(self.COLOR)
            text.setPos(px + self.RADIUS + 2, py - 6)
            text.setZValue(self.Z_VALUE)
            text.setVisible(self._visible)
            self._items.append(text)

    def clear(self) -> None:
        """清除所有叠加标记"""
        for item in self._items:
            self._scene.removeItem(item)
        self._items.clear()

    def set_visible(self, visible: bool) -> None:
        """设置可见性"""
        self._visible = visible
        for item in self._items:
            item.setVisible(visible)

    @property
    def is_visible(self) -> bool:
        return self._visible

    def toggle(self) -> bool:
        """切换可见性"""
        self._visible = not self._visible
        self.set_visible(self._visible)
        return self._visible
