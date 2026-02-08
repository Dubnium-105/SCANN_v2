"""实时直方图面板

需求:
- 直方图/屏幕拉伸功能
- 不改变图片原有实际像素，不保存，只临时调整显示亮度
- 黑点/白点滑块 (NoScrollSpinBox)
- 预设拉伸方式: 线性/对数/平方根/Asinh/自动
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import (
    QComboBox,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from scann.gui.widgets.no_scroll_spinbox import NoScrollDoubleSpinBox, NoScrollSpinBox


class StretchMode(Enum):
    """拉伸预设模式"""
    LINEAR = auto()
    LOG = auto()
    SQRT = auto()
    ASINH = auto()
    AUTO = auto()


class HistogramWidget(QWidget):
    """直方图绘制区"""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumHeight(80)
        self.setMaximumHeight(120)
        self._hist_data: Optional[np.ndarray] = None
        self._black_point: float = 0.0
        self._white_point: float = 1.0

    def set_histogram(self, hist: np.ndarray) -> None:
        """设置直方图数据 (256 bins)"""
        self._hist_data = hist
        self.update()

    def set_points(self, black: float, white: float) -> None:
        """设置黑白点 (0~1 归一化)"""
        self._black_point = black
        self._white_point = white
        self.update()

    def paintEvent(self, event) -> None:
        if self._hist_data is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        hist = self._hist_data.astype(float)
        max_val = hist.max() if hist.max() > 0 else 1.0

        # 绘制直方图
        pen = QPen(QColor("#4CAF50"), 1)
        painter.setPen(pen)
        bin_width = w / len(hist)
        for i, val in enumerate(hist):
            bar_h = int((val / max_val) * (h - 4))
            x = int(i * bin_width)
            painter.drawLine(x, h - 2, x, h - 2 - bar_h)

        # 绘制黑白点标记线
        bp_x = int(self._black_point * w)
        wp_x = int(self._white_point * w)

        painter.setPen(QPen(QColor("#F44336"), 2))  # 红色 = 黑点
        painter.drawLine(bp_x, 0, bp_x, h)

        painter.setPen(QPen(QColor("#2196F3"), 2))  # 蓝色 = 白点
        painter.drawLine(wp_x, 0, wp_x, h)

        painter.end()


class HistogramPanel(QDockWidget):
    """直方图拉伸面板 (可停靠)

    Signals:
        stretch_changed(float, float): 黑白点变化 (原始值)
        mode_changed(StretchMode): 拉伸模式变化
        reset_requested(): 重置请求
        apply_all_requested(): 应用到所有配对
    """

    stretch_changed = pyqtSignal(float, float)
    mode_changed = pyqtSignal(object)  # StretchMode
    reset_requested = pyqtSignal()
    apply_all_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__("直方图拉伸 (仅显示，不改变原始数据)", parent)
        self.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.RightDockWidgetArea)

        self._data_min: float = 0.0
        self._data_max: float = 65535.0

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # 直方图
        self.histogram_widget = HistogramWidget()
        layout.addWidget(self.histogram_widget)

        # 黑点
        bp_layout = QHBoxLayout()
        bp_layout.addWidget(QLabel("黑点:"))
        self.slider_black = QSlider(Qt.Horizontal)
        self.slider_black.setRange(0, 1000)
        self.slider_black.setValue(0)
        bp_layout.addWidget(self.slider_black, 1)
        self.spin_black = NoScrollSpinBox()
        self.spin_black.setRange(0, 65535)
        self.spin_black.setValue(0)
        self.spin_black.setFixedWidth(80)
        bp_layout.addWidget(self.spin_black)
        layout.addLayout(bp_layout)

        # 白点
        wp_layout = QHBoxLayout()
        wp_layout.addWidget(QLabel("白点:"))
        self.slider_white = QSlider(Qt.Horizontal)
        self.slider_white.setRange(0, 1000)
        self.slider_white.setValue(1000)
        wp_layout.addWidget(self.slider_white, 1)
        self.spin_white = NoScrollSpinBox()
        self.spin_white.setRange(0, 65535)
        self.spin_white.setValue(65535)
        self.spin_white.setFixedWidth(80)
        wp_layout.addWidget(self.spin_white)
        layout.addLayout(wp_layout)

        # 预设模式
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("预设:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["线性", "对数", "平方根", "Asinh", "自动拉伸"])
        mode_layout.addWidget(self.combo_mode, 1)
        layout.addLayout(mode_layout)

        # 按钮
        btn_layout = QHBoxLayout()
        self.btn_reset = QPushButton("重置")
        self.btn_apply_all = QPushButton("应用到所有配对")
        btn_layout.addWidget(self.btn_reset)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_apply_all)
        layout.addLayout(btn_layout)

        self.setWidget(content)

        # 信号连接
        self.slider_black.valueChanged.connect(self._on_black_slider)
        self.slider_white.valueChanged.connect(self._on_white_slider)
        self.spin_black.valueChanged.connect(self._on_black_spin)
        self.spin_white.valueChanged.connect(self._on_white_spin)
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        self.btn_reset.clicked.connect(self._on_reset)
        self.btn_apply_all.clicked.connect(self.apply_all_requested)

    def set_data_range(self, data_min: float, data_max: float) -> None:
        """设置数据值范围"""
        self._data_min = data_min
        self._data_max = data_max
        self.spin_black.setRange(int(data_min), int(data_max))
        self.spin_white.setRange(int(data_min), int(data_max))
        self.spin_black.setValue(int(data_min))
        self.spin_white.setValue(int(data_max))

    def set_image_data(self, data: np.ndarray) -> None:
        """根据图像数据更新直方图"""
        if data is None:
            return
        flat = data.flatten()
        hist, _ = np.histogram(flat, bins=256)
        self.histogram_widget.set_histogram(hist)
        self.set_data_range(float(flat.min()), float(flat.max()))

    @property
    def black_point(self) -> float:
        return float(self.spin_black.value())

    @property
    def white_point(self) -> float:
        return float(self.spin_white.value())

    def _on_black_slider(self, value: int) -> None:
        real = self._data_min + (value / 1000.0) * (self._data_max - self._data_min)
        self.spin_black.blockSignals(True)
        self.spin_black.setValue(int(real))
        self.spin_black.blockSignals(False)
        self.histogram_widget.set_points(value / 1000.0, self.slider_white.value() / 1000.0)
        self.stretch_changed.emit(real, self.white_point)

    def _on_white_slider(self, value: int) -> None:
        real = self._data_min + (value / 1000.0) * (self._data_max - self._data_min)
        self.spin_white.blockSignals(True)
        self.spin_white.setValue(int(real))
        self.spin_white.blockSignals(False)
        self.histogram_widget.set_points(self.slider_black.value() / 1000.0, value / 1000.0)
        self.stretch_changed.emit(self.black_point, real)

    def _on_black_spin(self, value: int) -> None:
        if self._data_max > self._data_min:
            norm = (value - self._data_min) / (self._data_max - self._data_min)
            self.slider_black.blockSignals(True)
            self.slider_black.setValue(int(norm * 1000))
            self.slider_black.blockSignals(False)
        self.stretch_changed.emit(float(value), self.white_point)

    def _on_white_spin(self, value: int) -> None:
        if self._data_max > self._data_min:
            norm = (value - self._data_min) / (self._data_max - self._data_min)
            self.slider_white.blockSignals(True)
            self.slider_white.setValue(int(norm * 1000))
            self.slider_white.blockSignals(False)
        self.stretch_changed.emit(self.black_point, float(value))

    def _on_mode_changed(self, index: int) -> None:
        modes = [StretchMode.LINEAR, StretchMode.LOG, StretchMode.SQRT,
                 StretchMode.ASINH, StretchMode.AUTO]
        if 0 <= index < len(modes):
            self.mode_changed.emit(modes[index])

    def _on_reset(self) -> None:
        self.spin_black.setValue(int(self._data_min))
        self.spin_white.setValue(int(self._data_max))
        self.combo_mode.setCurrentIndex(0)
        self.reset_requested.emit()
