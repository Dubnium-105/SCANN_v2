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
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._panels: list[QLabel] = []
        titles = ["差异图", "新图", "参考图"]
        for title in titles:
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumSize(80, 80)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            lbl.setStyleSheet("border: 1px solid #3C3C3C; background: #1E1E1E;")
            lbl.setToolTip(title)
            layout.addWidget(lbl)
            self._panels.append(lbl)

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
        self.setToolTip(name)
        # 可在标题区域显示文件名
        if self._panels:
            self._panels[0].setToolTip(f"差异图 - {name}")

    def set_ai_suggestion(self, suggestion: str, confidence: float) -> None:
        """显示 AI 建议和置信度"""
        tip = f"AI 建议: {suggestion} ({confidence:.1%})"
        for panel in self._panels:
            # 在面板上叠加 AI 建议信息
            pass
        self.setToolTip(tip)

    def clear(self) -> None:
        """清除所有面板"""
        for lbl in self._panels:
            lbl.clear()

    def _set_panel_pixmap(self, label: QLabel, data: np.ndarray) -> None:
        """将 numpy 数组设为 QLabel 的 pixmap (自适应缩放)"""
        h, w = data.shape[:2]
        qimg = QImage(data.data.tobytes(), w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(scaled)
