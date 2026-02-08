"""FITS 图像查看器 Widget

功能:
- 中键拖拽 (新旧图同步移动)
- 滚轮缩放
- 方框/十字线标记显示 (选中候选用十字线, 已判决显示 ✓/✗)
- 支持反色显示
- 鼠标追踪: 实时报告像素坐标
- center_on_point: 聚焦候选体
- F 键: 适配窗口
- MPCORB 叠加层集成
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
    QTransform,
    QWheelEvent,
)
from PyQt5.QtWidgets import (
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)

from scann.core.models import TargetVerdict


class FitsImageViewer(QGraphicsView):
    """FITS 图像查看器

    交互:
    - 中键拖拽: 拖动图片
    - 滚轮: 缩放 (锚点在鼠标下)
    - 左键: 选点 → point_clicked 信号
    - 右键: 查询菜单 → right_click 信号
    - F 键: 适配窗口 (fitInView)
    - 鼠标移动: 实时坐标 → mouse_moved 信号
    """

    # ── 信号 ──
    point_clicked = pyqtSignal(int, int)       # 左键点击 (图像像素坐标)
    right_click = pyqtSignal(int, int)          # 右键点击 (图像像素坐标)
    mouse_moved = pyqtSignal(int, int)          # 鼠标移动 (图像像素坐标)
    zoom_changed = pyqtSignal(float)            # 缩放比例变化 (百分比)

    # ── 缩放限制 ──
    ZOOM_MIN = 0.05    # 5%
    ZOOM_MAX = 20.0    # 2000%
    ZOOM_FACTOR = 1.25

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

        # 鼠标追踪 (实时坐标)
        self.setMouseTracking(True)

        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        # 标记图层分组 (z-value: pixmap=0, mpcorb=5, markers=10)
        self._marker_items: list = []

        # 状态
        self._is_panning = False
        self._pan_start = QPointF()
        self._zoom_level = 1.0

    # ══════════════════════════════════════════════
    #  图像加载
    # ══════════════════════════════════════════════

    def set_image_data(self, data: np.ndarray, inverted: bool = False) -> None:
        """加载图像数据

        Args:
            data: 显示用的数据 (float32 0~1 或 uint8 或 uint16)
            inverted: 是否反色显示
        """
        if data is None:
            return

        # 转换为 uint8 显示
        if data.dtype == np.float32 or data.dtype == np.float64:
            display = (np.clip(data, 0, 1) * 255).astype(np.uint8)
        elif data.dtype == np.uint16:
            dmin, dmax = float(data.min()), float(data.max())
            if dmax > dmin:
                display = ((data.astype(np.float32) - dmin) / (dmax - dmin) * 255).astype(np.uint8)
            else:
                display = np.zeros_like(data, dtype=np.uint8)
        else:
            display = data.astype(np.uint8)

        if inverted:
            display = 255 - display

        # 转为 QImage
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
            self._emit_zoom()
        else:
            self._scene.setSceneRect(QRectF(pixmap.rect()))

    # ══════════════════════════════════════════════
    #  候选标记绘制
    # ══════════════════════════════════════════════

    def draw_markers(
        self,
        candidates: list,
        selected_idx: int = -1,
        hide_all: bool = False,
    ) -> None:
        """绘制候选体标记

        增强:
        - 选中候选: 红色十字线 (crosshair)
        - 已判决: 显示 ✓ (真) 或 ✗ (假) 图标
        - 手动添加: 紫色
        - 已知天体: 灰色
        - 自动检测: 绿色

        Args:
            candidates: 候选体列表
            selected_idx: 当前选中索引 (-1 表示无)
            hide_all: 隐藏所有标记
        """
        # 清除旧标记 (保留 pixmap 和 z < 10 的叠加层如 MPCORB)
        for item in self._marker_items:
            self._scene.removeItem(item)
        self._marker_items.clear()

        if hide_all:
            return

        font = QFont("Arial", 10, QFont.Bold)
        verdict_font = QFont("Arial", 12, QFont.Bold)

        for i, cand in enumerate(candidates):
            cx, cy = cand.x, cand.y
            is_selected = i == selected_idx

            # 颜色选择
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

            pen = QPen(color, pen_width)

            # 画圆
            radius = 15
            ellipse = self._scene.addEllipse(
                cx - radius, cy - radius, radius * 2, radius * 2, pen,
            )
            ellipse.setZValue(10)
            self._marker_items.append(ellipse)

            # 选中项 → 十字线
            if is_selected:
                cross_len = 25
                cross_pen = QPen(QColor(255, 0, 0, 180), 1, Qt.DashLine)
                h_line = self._scene.addLine(
                    cx - cross_len, cy, cx + cross_len, cy, cross_pen,
                )
                v_line = self._scene.addLine(
                    cx, cy - cross_len, cx, cy + cross_len, cross_pen,
                )
                h_line.setZValue(10)
                v_line.setZValue(10)
                self._marker_items.extend([h_line, v_line])

            # 判决图标
            verdict_text = ""
            verdict_color = QColor(255, 255, 255)
            if hasattr(cand, "verdict") and cand.verdict is not None:
                if cand.verdict == TargetVerdict.REAL:
                    verdict_text = "✓"
                    verdict_color = QColor(76, 175, 80)  # 绿色
                elif cand.verdict == TargetVerdict.BOGUS:
                    verdict_text = "✗"
                    verdict_color = QColor(244, 67, 54)  # 红色

            if verdict_text:
                vtext = self._scene.addText(verdict_text, verdict_font)
                vtext.setDefaultTextColor(verdict_color)
                vtext.setPos(cx - radius - 12, cy - radius - 12)
                vtext.setZValue(11)
                self._marker_items.append(vtext)

            # 编号标签
            text = self._scene.addText(f"{i + 1}", font)
            text.setDefaultTextColor(QColor(255, 255, 0))
            text.setPos(cx + radius + 2, cy - 8)
            text.setZValue(10)
            self._marker_items.append(text)

    # ══════════════════════════════════════════════
    #  导航
    # ══════════════════════════════════════════════

    def center_on_point(
        self, x: float, y: float, zoom_to: Optional[float] = None
    ) -> None:
        """将视图中心移至指定图像坐标

        Args:
            x: 图像 X 坐标
            y: 图像 Y 坐标
            zoom_to: 缩放百分比 (如 200 = 200%), None 表示保持当前缩放
        """
        if zoom_to is not None:
            desired = zoom_to / 100.0
            desired = max(self.ZOOM_MIN, min(self.ZOOM_MAX, desired))
            self.resetTransform()
            self.scale(desired, desired)
            self._zoom_level = desired
            self._emit_zoom()

        self.centerOn(QPointF(x, y))

    def fit_in_view(self) -> None:
        """适配视图 (F 键)"""
        rect = self._scene.sceneRect()
        if not rect.isEmpty():
            self.fitInView(rect, Qt.KeepAspectRatio)
            self._emit_zoom()

    def mapFromScene(self, x: float, y: float) -> QPointF:
        """将场景坐标映射为视图坐标

        Args:
            x: 场景 X 坐标
            y: 场景 Y 坐标

        Returns:
            视图坐标 QPointF
        """
        return super().mapFromScene(QPointF(x, y))

    # ══════════════════════════════════════════════
    #  交互事件
    # ══════════════════════════════════════════════

    def wheelEvent(self, event: QWheelEvent) -> None:
        """滚轮缩放"""
        if event.angleDelta().y() > 0:
            factor = self.ZOOM_FACTOR
        else:
            factor = 1.0 / self.ZOOM_FACTOR

        new_zoom = self._zoom_level * factor
        if self.ZOOM_MIN <= new_zoom <= self.ZOOM_MAX:
            self.scale(factor, factor)
            self._zoom_level = new_zoom
            self._emit_zoom()

    def keyPressEvent(self, event) -> None:
        """键盘事件"""
        if event.key() == Qt.Key_F:
            self.fit_in_view()
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """鼠标按下"""
        if event.button() == Qt.MiddleButton:
            self._is_panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            if self._pixmap_item.boundingRect().contains(scene_pos):
                self.point_clicked.emit(int(scene_pos.x()), int(scene_pos.y()))
        elif event.button() == Qt.RightButton:
            scene_pos = self.mapToScene(event.pos())
            if self._pixmap_item.boundingRect().contains(scene_pos):
                self.right_click.emit(int(scene_pos.x()), int(scene_pos.y()))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """鼠标移动"""
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
            # 实时坐标追踪
            scene_pos = self.mapToScene(event.pos())
            if self._pixmap_item.boundingRect().contains(scene_pos):
                self.mouse_moved.emit(int(scene_pos.x()), int(scene_pos.y()))
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """鼠标释放"""
        if event.button() == Qt.MiddleButton:
            self._is_panning = False
            self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

    # ══════════════════════════════════════════════
    #  内部辅助
    # ══════════════════════════════════════════════

    def _emit_zoom(self) -> None:
        """发送缩放比例信号"""
        t = self.transform()
        self._zoom_level = t.m11()  # 水平缩放因子
        self.zoom_changed.emit(self._zoom_level * 100)

    @property
    def scene_ref(self) -> QGraphicsScene:
        """暴露场景引用 (供 MpcorbOverlay 使用)"""
        return self._scene
