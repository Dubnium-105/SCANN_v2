# -*- coding: utf-8 -*-
"""
SCANN UI 组件模块
- ImageViewer: 图像查看器 (支持缩放、平移、点选)
- SuspectListWidget: 可疑目标列表组件
- SuspectGlobalKeyFilter: 全局快捷键过滤器
"""

import numpy as np

from PyQt5.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QListWidget
)
from PyQt5.QtCore import Qt, QRectF, QEvent, QObject, pyqtSignal
from PyQt5.QtGui import (
    QPixmap, QImage, QFont, QColor, QPainter, QPen, QBrush,
    QWheelEvent, QMouseEvent
)


class ImageViewer(QGraphicsView):
    """图像查看器组件"""
    
    # 发送点击的图片坐标 (x, y)
    point_selected = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # 交互设置
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(20, 20, 20)))
        
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

    def set_image(self, cv_img):
        """加载 OpenCV 图片"""
        if cv_img is None:
            return
        if not cv_img.flags['C_CONTIGUOUS']:
            cv_img = np.ascontiguousarray(cv_img)
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        qimg = QImage(cv_img.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qimg)
        self.pixmap_item.setPixmap(pixmap)
        
        # 仅在首次加载或场景为空时自动适配
        if self.scene.sceneRect().isEmpty():
            self.scene.setSceneRect(QRectF(pixmap.rect()))
            self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        else:
            # 保持当前视图区域不变
            self.scene.setSceneRect(QRectF(pixmap.rect()))

    def draw_overlays(self, candidates, current_idx, hide_all=False):
        """绘制圆圈和标记"""
        # 清除旧的标记（保留 pixmap_item）
        for item in self.scene.items():
            if item != self.pixmap_item:
                self.scene.removeItem(item)

        if hide_all:
            return

        font = QFont("Arial", 12, QFont.Bold)
        
        for i, cand in enumerate(candidates):
            cx, cy = cand['x'], cand['y']
            is_manual = cand.get('manual', False)
            is_saved = cand.get('saved', False)
            is_selected = (i == current_idx)

            if is_manual:
                color = QColor(255, 0, 255)  # 紫色 (手动)
            else:
                color = QColor(0, 255, 0)    # 绿色 (自动)
            
            # 如果已保存，给一个特殊颜色
            if is_saved:
                color = QColor(0, 255, 255)  # 青色 (已保存)

            pen_width = 3 if is_selected else 2
            if is_selected:
                color = QColor(255, 0, 0)    # 选中变红

            radius = 12
            ellipse = self.scene.addEllipse(cx - radius, cy - radius, radius * 2, radius * 2, QPen(color, pen_width))
            ellipse.setZValue(10)
            
            text = self.scene.addText(str(cand.get('id', i + 1)), font)
            text.setDefaultTextColor(QColor(255, 255, 0))
            text.setPos(cx + 10, cy - 10)
            text.setZValue(10)

    def wheelEvent(self, event: QWheelEvent):
        """滚轮缩放"""
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        if event.angleDelta().y() > 0:
            self.scale(zoom_in_factor, zoom_in_factor)
        else:
            self.scale(zoom_out_factor, zoom_out_factor)

    def mousePressEvent(self, event):
        # 右键触发平移
        if event.button() == Qt.RightButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            fake_event = QMouseEvent(QEvent.MouseButtonPress, event.pos(), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
            super().mousePressEvent(fake_event)
        
        # 左键触发选点
        elif event.button() == Qt.LeftButton:
            self.setDragMode(QGraphicsView.NoDrag)
            scene_pos = self.mapToScene(event.pos())
            if self.pixmap_item.boundingRect().contains(scene_pos):
                self.point_selected.emit(int(scene_pos.x()), int(scene_pos.y()))
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self.setDragMode(QGraphicsView.NoDrag)
            self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)


class SuspectListWidget(QListWidget):
    """可疑目标列表组件，支持快捷键"""
    
    def __init__(self, main_window):
        super().__init__()
        self.main = main_window
        self.setFont(QFont("Arial", 11))

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_S:
            self.main.handle_suspect_action(True)
        elif key == Qt.Key_D:
            self.main.handle_suspect_action(False)
        elif key == Qt.Key_Space:
            self.main.handle_suspect_skip()
        elif key == Qt.Key_R:
            try:
                self.main.btn_blink.click()
            except Exception:
                self.main.toggle_blink()
        else:
            super().keyPressEvent(event)


class SuspectGlobalKeyFilter(QObject):
    """全局快捷键过滤器"""
    
    def __init__(self, main_window):
        super().__init__()
        self.main = main_window

    def eventFilter(self, obj, event):
        if not self.main._is_suspect_mode_active():
            return False

        et = event.type()
        if et not in (QEvent.ShortcutOverride, QEvent.KeyPress):
            return False

        key = event.key()
        if key not in (Qt.Key_S, Qt.Key_D, Qt.Key_Space, Qt.Key_R):
            return False

        if et == QEvent.ShortcutOverride:
            event.accept()
            return True

        if key == Qt.Key_S:
            self.main.handle_suspect_action(True)
            return True
        if key == Qt.Key_D:
            self.main.handle_suspect_action(False)
            return True
        if key == Qt.Key_Space:
            self.main.handle_suspect_skip()
            return True
        if key == Qt.Key_R:
            try:
                self.main.btn_blink.click()
            except Exception:
                self.main.toggle_blink()
            return True

        return False
