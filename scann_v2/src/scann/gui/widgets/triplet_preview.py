"""三联图预览面板

v1 模式下将 80×240 PNG 三联图拆分为 3 个 80×80 面板并排放大显示。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from PIL import Image


class TripletPreviewPanel(QWidget):
    """三联图放大预览 (3 × 80×80 并排)

    将 80×240 三联图拆分为:
    - 左: 差异图 (0:80)
    - 中: 新图 (80:160)
    - 右: 参考图 (160:240)
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        panel_layout = QHBoxLayout()
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(8)

        self._panels: list[QLabel] = []
        self._panel_titles = ["差异图", "新图", "参考图"]
        self._file_name = ""
        self._ai_tooltip = ""

        for title in self._panel_titles:
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumSize(80, 80)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            lbl.setStyleSheet("border: 1px solid #3C3C3C; background: #1E1E1E;")
            lbl.setToolTip(title)
            panel_layout.addWidget(lbl)
            self._panels.append(lbl)

        layout.addLayout(panel_layout)

        self._ai_hint_label = QLabel()
        self._ai_hint_label.setAlignment(Qt.AlignCenter)
        self._ai_hint_label.setWordWrap(True)
        self._ai_hint_label.setStyleSheet(self._badge_style("default"))
        self._ai_hint_label.hide()
        layout.addWidget(self._ai_hint_label)

    def set_image(self, image: Image.Image) -> None:
        """加载三联图并拆分显示"""
        w, h = image.size
        panel_w = w  # 80px
        panel_h = h // 3 if h > w else h  # 80px each

        # 转为 numpy 做拆分
        arr = np.array(image)
        if arr.ndim == 3:
            arr = arr[:, :, 0]  # 取第一通道

        # 判断排列方向 (80×240 → 水平三联 or 垂直三联)
        if h > w:
            # 垂直排列: 每个面板 80×80
            panel_h = h // 3
            panels = [arr[i * panel_h:(i + 1) * panel_h, :] for i in range(3)]
        else:
            # 水平排列: 每个面板 w/3 × h
            panel_w = w // 3
            panels = [arr[:, i * panel_w:(i + 1) * panel_w] for i in range(3)]

        for i, panel_data in enumerate(panels):
            if i < len(self._panels):
                self._set_panel_pixmap(self._panels[i], panel_data)

    def set_triplet_image(self, image) -> None:
        """加载三联图 (兼容别名)

        Args:
            image: PIL.Image 或 numpy 数组
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        self.set_image(image)

    def set_file_info(self, name: str) -> None:
        """显示文件名信息"""
        self._file_name = name
        self._refresh_tooltips()

    def set_ai_suggestion(self, suggestion: str, confidence: float) -> None:
        """显示 AI 建议和置信度"""
        normalized = (suggestion or "").strip().lower()
        suggestion_text = self._suggestion_text(normalized)
        tip = f"AI 建议: {suggestion_text} ({confidence:.1%})"
        self._ai_tooltip = tip
        self._ai_hint_label.setText(tip)
        self._ai_hint_label.setStyleSheet(self._badge_style(normalized))
        self._ai_hint_label.show()
        self._refresh_tooltips()

    def clear_ai_suggestion(self) -> None:
        """隐藏当前 AI 建议信息。"""
        self._ai_tooltip = ""
        self._ai_hint_label.clear()
        self._ai_hint_label.hide()
        self._refresh_tooltips()

    def clear(self) -> None:
        """清除所有面板"""
        for lbl in self._panels:
            lbl.clear()
        self._file_name = ""
        self.clear_ai_suggestion()

    def _set_panel_pixmap(self, label: QLabel, data: np.ndarray) -> None:
        """将 numpy 数组设为 QLabel 的 pixmap (自适应缩放)"""
        h, w = data.shape[:2]
        qimg = QImage(data.data.tobytes(), w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(scaled)

    def _refresh_tooltips(self) -> None:
        widget_tooltip_parts = []
        if self._file_name:
            widget_tooltip_parts.append(self._file_name)
        if self._ai_tooltip:
            widget_tooltip_parts.append(self._ai_tooltip)
        self.setToolTip("\n".join(widget_tooltip_parts))

        for idx, panel in enumerate(self._panels):
            parts = [self._panel_titles[idx]]
            if idx == 0 and self._file_name:
                parts[0] = f"{parts[0]} - {self._file_name}"
            if self._ai_tooltip:
                parts.append(self._ai_tooltip)
            panel.setToolTip("\n".join(parts))

    @staticmethod
    def _suggestion_text(suggestion: str) -> str:
        display_map = {
            "real": "A.真",
            "bogus": "B.假",
        }
        return display_map.get(suggestion, suggestion or "未知")

    @staticmethod
    def _badge_style(suggestion: str) -> str:
        color_map = {
            "real": "#1B5E20",
            "bogus": "#B71C1C",
            "default": "#37474F",
        }
        bg = color_map.get(suggestion, color_map["default"])
        return (
            "QLabel {"
            f"background: {bg};"
            " color: white;"
            " border-radius: 4px;"
            " padding: 4px 8px;"
            " font-size: 11px;"
            " font-weight: bold;"
            "}"
        )
