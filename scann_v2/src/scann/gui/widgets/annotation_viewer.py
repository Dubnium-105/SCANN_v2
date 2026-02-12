"""标注图像查看器

支持边界框绘制/编辑的图像查看器，兼容三联图和 FITS 两种模式。
v2 模式下支持框选 (B)、点标 (Q)、移动 (V) 等工具。

交互:
- 滚轮: 缩放
- 右键拖拽: 平移
- 左键: 取决于当前工具 (box/point/move)
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush, QPixmap, QImage, QCursor
from PyQt5.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QGraphicsRectItem,
    QWidget,
    QGraphicsLineItem,
)

import numpy as np

from scann.core.annotation_models import (
    BBox,
    DETAIL_TYPE_COLOR,
    DEFAULT_BBOX_COLOR,
    SELECTED_BBOX_COLOR,
)


class AnnotationViewer(QGraphicsView):
    """标注专用图像查看器

    支持:
    - FITS 图像显示 (numpy → QPixmap)
    - 边界框绘制/编辑/删除
    - 滚轮缩放 / 右键平移
    - 新旧图切换 (标注框位置不变)

    Signals:
        box_drawn(BBox): 新边界框绘制完成
        box_selected(int): 选中标注框 (index)
        bbox_deleted(int): 删除标注框 (index)
        point_clicked(int, int): 鼠标左键点击坐标
    """

    box_drawn = pyqtSignal(object)       # BBox
    box_selected = pyqtSignal(int)       # index
    bbox_deleted = pyqtSignal(int)       # index
    point_clicked = pyqtSignal(int, int)

    # 缩放范围
    _ZOOM_MIN = 0.1
    _ZOOM_MAX = 20.0
    _ZOOM_FACTOR = 1.15

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # 渲染优化
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)

        # 状态
        self._image_data: Optional[np.ndarray] = None
        self._display_data: Optional[np.ndarray] = None  # 拉伸后的显示数据
        self._bboxes: list[BBox] = []
        self._selected_bbox_idx: int = -1
        self._current_tool: str = "move"  # "box", "point", "move"
        self._bbox_pen_width: int = 2
        self._inverted: bool = False

        # 绘制状态
        self._drawing: bool = False
        self._draw_start: Optional[QPointF] = None
        self._draw_rect_item: Optional[QGraphicsRectItem] = None

        # 右键平移状态
        self._panning: bool = False
        self._pan_start: Optional[QPointF] = None

        # 当前缩放倍率
        self._zoom_level: float = 1.0

        # 默认大小
        self.setMinimumSize(200, 200)

    # ─── 属性 ───

    @property
    def selected_bbox_index(self) -> int:
        """当前选中的标注框索引"""
        return self._selected_bbox_idx

    def set_image(self, data: np.ndarray, is_new: bool = True, view: str = "new") -> None:
        """设置显示图像

        Args:
            data: numpy 图像数组
            is_new: 是否为新图
            view: 视图类型 ("new" / "old")
        """
        self._image_data = data
        self._display_data = None  # 清除旧的拉伸数据
        self._update_display()

    def set_bboxes(self, bboxes: list[BBox]) -> None:
        """设置标注框列表"""
        self._bboxes = list(bboxes)
        self._update_display()

    def set_tool(self, tool: str) -> None:
        """切换绘制工具: box / point / move"""
        self._current_tool = tool
        # 更新光标
        if tool == "box":
            self.setCursor(Qt.CrossCursor)
        elif tool == "point":
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def set_bbox_width(self, width: int) -> None:
        """设置边界框画笔宽度"""
        self._bbox_pen_width = max(1, width)
        self._update_display()

    def toggle_invert(self) -> None:
        """切换反色显示"""
        self._inverted = not self._inverted
        self._update_display()

    def set_display_data(self, data: np.ndarray) -> None:
        """设置经拉伸处理后的显示数据 (跳过内部归一化)"""
        self._display_data = data
        self._update_display()

    def get_selected_bbox_index(self) -> int:
        """返回当前选中的标注框索引"""
        return self._selected_bbox_idx

    def select_bbox(self, index: int) -> None:
        """选中指定索引的标注框"""
        if 0 <= index < len(self._bboxes):
            self._selected_bbox_idx = index
            self._update_display()

    def clear(self) -> None:
        """清除图像和标注框"""
        self._image_data = None
        self._display_data = None
        self._bboxes = []
        self._selected_bbox_idx = -1
        self._scene.clear()

    def fit_in_view(self) -> None:
        """自适应窗口大小"""
        if self._scene.sceneRect().isNull():
            return
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self._zoom_level = self.transform().m11()

    # ─── 鼠标事件 ───

    def wheelEvent(self, event) -> None:
        """滚轮缩放"""
        if event.angleDelta().y() > 0:
            factor = self._ZOOM_FACTOR
        else:
            factor = 1.0 / self._ZOOM_FACTOR

        new_zoom = self._zoom_level * factor
        if self._ZOOM_MIN <= new_zoom <= self._ZOOM_MAX:
            self.scale(factor, factor)
            self._zoom_level = new_zoom

    def mousePressEvent(self, event) -> None:
        """鼠标按下"""
        # 右键: 开始平移
        if event.button() == Qt.RightButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        # 左键: 根据工具处理
        if event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())

            if self._current_tool == "box":
                self._start_box_draw(scene_pos)
                event.accept()
                return
            elif self._current_tool == "point":
                self._do_point_click(scene_pos)
                event.accept()
                return
            elif self._current_tool == "move":
                self._try_select_bbox(scene_pos)
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        """鼠标移动"""
        # 右键平移
        if self._panning and self._pan_start is not None:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
            event.accept()
            return

        # 框选绘制中
        if self._drawing and self._draw_start is not None:
            scene_pos = self.mapToScene(event.pos())
            self._update_draw_rect(scene_pos)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        """鼠标释放"""
        # 右键平移结束
        if event.button() == Qt.RightButton and self._panning:
            self._panning = False
            self._pan_start = None
            # 恢复工具光标
            self.set_tool(self._current_tool)
            event.accept()
            return

        # 框选结束
        if event.button() == Qt.LeftButton and self._drawing:
            scene_pos = self.mapToScene(event.pos())
            self._finish_box_draw(scene_pos)
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        """快捷键: Delete 删除选中框"""
        if event.key() == Qt.Key_Delete and self._selected_bbox_idx >= 0:
            self.bbox_deleted.emit(self._selected_bbox_idx)
            event.accept()
            return
        super().keyPressEvent(event)

    # ─── 工具动作 ───

    def _start_box_draw(self, scene_pos: QPointF) -> None:
        """开始框选绘制"""
        self._drawing = True
        self._draw_start = scene_pos
        # 绘制预览矩形
        pen = QPen(QColor("#FFEB3B"), 1, Qt.DashLine)  # 黄色虚线
        self._draw_rect_item = self._scene.addRect(
            QRectF(scene_pos, scene_pos), pen
        )

    def _update_draw_rect(self, current_pos: QPointF) -> None:
        """更新框选预览"""
        if self._draw_rect_item and self._draw_start:
            rect = QRectF(self._draw_start, current_pos).normalized()
            self._draw_rect_item.setRect(rect)

    def _finish_box_draw(self, end_pos: QPointF) -> None:
        """完成框选绘制"""
        if self._draw_rect_item:
            self._scene.removeItem(self._draw_rect_item)
            self._draw_rect_item = None

        if self._draw_start is None:
            self._drawing = False
            return

        rect = QRectF(self._draw_start, end_pos).normalized()
        self._drawing = False
        self._draw_start = None

        # 忽略太小的框 (< 4px)
        if rect.width() < 4 or rect.height() < 4:
            return

        bbox = BBox(
            x=int(rect.x()),
            y=int(rect.y()),
            width=int(rect.width()),
            height=int(rect.height()),
        )
        self.box_drawn.emit(bbox)

    def _do_point_click(self, scene_pos: QPointF) -> None:
        """点标工具: 发射点击坐标"""
        self.point_clicked.emit(int(scene_pos.x()), int(scene_pos.y()))

    def _try_select_bbox(self, scene_pos: QPointF) -> None:
        """移动工具: 点击选中标注框"""
        px, py = scene_pos.x(), scene_pos.y()
        for i, bbox in enumerate(self._bboxes):
            if bbox.contains(int(px), int(py)):
                self._selected_bbox_idx = i
                self.box_selected.emit(i)
                self._update_display()
                return

        # 点击空白处取消选中
        if self._selected_bbox_idx >= 0:
            self._selected_bbox_idx = -1
            self._update_display()

    # ─── 渲染 ───

    def _update_display(self) -> None:
        """重新绘制图像和标注框"""
        self._scene.clear()
        self._draw_rect_item = None  # 清除引用

        # 优先使用 display_data (拉伸后)，否则使用原始数据
        raw = self._display_data if self._display_data is not None else self._image_data
        if raw is not None:
            h, w = raw.shape[:2]
            # 归一化到 0-255
            data = raw.astype(np.float64)
            dmin, dmax = data.min(), data.max()
            if dmax > dmin:
                data = ((data - dmin) / (dmax - dmin) * 255).astype(np.uint8)
            else:
                data = np.zeros_like(data, dtype=np.uint8)

            # 反色处理
            if self._inverted:
                data = 255 - data

            # 确保数据连续
            data = np.ascontiguousarray(data)
            qimg = QImage(data.data, w, h, w, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimg)
            self._scene.addPixmap(pixmap)
            self._scene.setSceneRect(0, 0, w, h)

        # 绘制标注框
        for i, bbox in enumerate(self._bboxes):
            # 优先使用 detail_type 对应的颜色
            if bbox.detail_type:
                from scann.core.annotation_models import DetailType
                try:
                    detail_enum = DetailType(bbox.detail_type)
                    color_hex = DETAIL_TYPE_COLOR.get(detail_enum, DEFAULT_BBOX_COLOR)
                    color = QColor(color_hex)
                except (ValueError, KeyError):
                    # 如果 detail_type 无效，使用默认颜色
                    color = QColor(DEFAULT_BBOX_COLOR)
            else:
                # 降级使用 label 对应的颜色
                color = QColor("#4CAF50") if bbox.label == "real" else QColor("#F44336")
            
            if i == self._selected_bbox_idx:
                color = QColor(SELECTED_BBOX_COLOR)  # 紫色选中
            pen = QPen(color, self._bbox_pen_width)
            self._scene.addRect(
                QRectF(bbox.x, bbox.y, bbox.width, bbox.height), pen
            )
