"""FITS 图像查看器 Widget

需求:
- 中键拖拽 (新旧图同步移动)
- 滚轮缩放
- 方框/十字线标记显示
- 支持反色显示
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from PyQt5.QtCore import QEvent, QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PyQt5.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


class FitsImageViewer(QGraphicsView):
    """FITS 图像查看器

    快捷键 (非全局，仅窗口焦点内有效):
    - 中键拖拽: 拖动图片
    - 滚轮: 缩放
    """

    # 信号
    point_clicked = pyqtSignal(int, int)       # 点击位置 (图像坐标)
    right_click = pyqtSignal(int, int)          # 右键位置 (用于查询菜单)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # 渲染设置
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(20, 20, 20)))

        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        # 状态
        self._is_panning = False
        self._pan_start = QPointF()

    def set_image_data(self, data: np.ndarray, inverted: bool = False) -> None:
        """加载图像数据

        Args:
            data: 显示用的数据 (float32 0~1 或 uint8)
            inverted: 是否反色显示
        """
        if data is None:
            return

        # 转换为 uint8 显示
        if data.dtype == np.float32 or data.dtype == np.float64:
            display = (np.clip(data, 0, 1) * 255).astype(np.uint8)
        elif data.dtype == np.uint16:
            # 简单线性映射到 0~255
            dmin, dmax = float(data.min()), float(data.max())
            if dmax > dmin:
                display = ((data.astype(np.float32) - dmin) / (dmax - dmin) * 255).astype(np.uint8)
            else:
                display = np.zeros_like(data, dtype=np.uint8)
        else:
            display = data.astype(np.uint8)

        if inverted:
            display = 255 - display

        # 转为 QImage (灰度)
        if display.ndim == 2:
            h, w = display.shape
            bytes_per_line = w
            qimg = QImage(display.data.tobytes(), w, h, bytes_per_line, QImage.Format_Grayscale8)
        elif display.ndim == 3 and display.shape[2] == 3:
            h, w, ch = display.shape
            bytes_per_line = ch * w
            if not display.flags["C_CONTIGUOUS"]:
                display = np.ascontiguousarray(display)
            qimg = QImage(display.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        else:
            return

        pixmap = QPixmap.fromImage(qimg)
        self._pixmap_item.setPixmap(pixmap)

        # 首次加载时适配视图
        if self._scene.sceneRect().isEmpty():
            self._scene.setSceneRect(QRectF(pixmap.rect()))
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        else:
            self._scene.setSceneRect(QRectF(pixmap.rect()))

    def draw_markers(
        self,
        candidates: list,
        selected_idx: int = -1,
        hide_all: bool = False,
    ) -> None:
        """绘制候选体标记"""
        # 清除旧标记
        for item in self._scene.items():
            if item != self._pixmap_item:
                self._scene.removeItem(item)

        if hide_all:
            return

        font = QFont("Arial", 10, QFont.Bold)

        for i, cand in enumerate(candidates):
            cx, cy = cand.x, cand.y
            is_selected = i == selected_idx

            # 颜色
            if is_selected:
                color = QColor(255, 0, 0)  # 红色 = 选中
                pen_width = 3
            elif cand.is_manual:
                color = QColor(255, 0, 255)  # 紫色 = 手动
                pen_width = 2
            elif cand.is_known:
                color = QColor(128, 128, 128)  # 灰色 = 已知
                pen_width = 1
            else:
                color = QColor(0, 255, 0)  # 绿色 = 自动检测
                pen_width = 2

            # 画圆
            radius = 15
            self._scene.addEllipse(
                cx - radius, cy - radius, radius * 2, radius * 2,
                QPen(color, pen_width),
            ).setZValue(10)

            # 标签
            text = self._scene.addText(f"{i + 1}", font)
            text.setDefaultTextColor(QColor(255, 255, 0))
            text.setPos(cx + radius + 2, cy - 8)
            text.setZValue(10)

    # ─── 交互事件 ───

    def wheelEvent(self, event: QWheelEvent) -> None:
        """滚轮缩放"""
        factor = 1.25 if event.angleDelta().y() > 0 else 1 / 1.25
        self.scale(factor, factor)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """鼠标按下"""
        if event.button() == Qt.MiddleButton:
            # 中键拖拽
            self._is_panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton:
            # 左键选点
            scene_pos = self.mapToScene(event.pos())
            if self._pixmap_item.boundingRect().contains(scene_pos):
                self.point_clicked.emit(int(scene_pos.x()), int(scene_pos.y()))
        elif event.button() == Qt.RightButton:
            # 右键菜单
            scene_pos = self.mapToScene(event.pos())
            if self._pixmap_item.boundingRect().contains(scene_pos):
                self.right_click.emit(int(scene_pos.x()), int(scene_pos.y()))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """鼠标移动 (中键拖拽)"""
        if self._is_panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """鼠标释放"""
        if event.button() == Qt.MiddleButton:
            self._is_panning = False
            self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)
