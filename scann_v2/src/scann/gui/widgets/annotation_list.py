"""标注列表面板

v2 FITS 全图模式下显示当前图像的所有标注框列表，
支持选中、删除、修改标签等操作。
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from scann.core.annotation_models import BBox, DETAIL_TYPE_DISPLAY, DetailType


class AnnotationListWidget(QWidget):
    """标注框列表

    Signals:
        bbox_selected(int): 选中标注框
        bbox_deleted(int): 删除标注框
        label_changed(int, str): 标签变更 (index, new_detail_type)
    """

    bbox_selected = pyqtSignal(int)
    bbox_delete_requested = pyqtSignal(int)
    label_changed = pyqtSignal(int, str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._table = QTableWidget()
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["#", "标签", "位置"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.currentCellChanged.connect(self._on_selection_changed)
        layout.addWidget(self._table)

    @property
    def table(self) -> QTableWidget:
        """暴露内部表格控件"""
        return self._table

    def set_bboxes(self, bboxes: list[BBox]) -> None:
        """设置标注框列表"""
        self._table.setRowCount(len(bboxes))
        for i, bbox in enumerate(bboxes):
            self._table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            label_text = bbox.detail_type or bbox.label
            try:
                dt = DetailType(label_text)
                label_text = DETAIL_TYPE_DISPLAY.get(dt, label_text)
            except (ValueError, KeyError):
                pass
            self._table.setItem(i, 1, QTableWidgetItem(label_text))
            self._table.setItem(i, 2, QTableWidgetItem(f"{bbox.x},{bbox.y}"))

    def clear(self) -> None:
        self._table.setRowCount(0)

    def _on_selection_changed(self, row: int, *_) -> None:
        if row >= 0:
            self.bbox_selected.emit(row)
