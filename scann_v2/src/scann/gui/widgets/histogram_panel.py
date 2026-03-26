"""实时直方图拉伸面板。"""

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

from scann.gui.widgets.no_scroll_spinbox import NoScrollDoubleSpinBox


class StretchMode(Enum):
    LINEAR = auto()
    LOG = auto()
    SQRT = auto()
    ASINH = auto()
    AUTO = auto()


class HistogramWidget(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumHeight(80)
        self.setMaximumHeight(120)
        self._hist_data: Optional[np.ndarray] = None
        self._black_point: float = 0.0
        self._white_point: float = 1.0

    def set_histogram(self, hist: np.ndarray) -> None:
        self._hist_data = hist
        self.update()

    def set_points(self, black: float, white: float) -> None:
        self._black_point = float(np.clip(black, 0.0, 1.0))
        self._white_point = float(np.clip(white, 0.0, 1.0))
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        if self._hist_data is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()
        hist = self._hist_data.astype(float)
        max_val = hist.max() if hist.max() > 0 else 1.0

        painter.setPen(QPen(QColor("#4CAF50"), 1))
        bin_width = width / len(hist)
        for index, value in enumerate(hist):
            bar_height = int((value / max_val) * (height - 4))
            x_pos = int(index * bin_width)
            painter.drawLine(x_pos, height - 2, x_pos, height - 2 - bar_height)

        black_x = int(self._black_point * width)
        white_x = int(self._white_point * width)

        painter.setPen(QPen(QColor("#F44336"), 2))
        painter.drawLine(black_x, 0, black_x, height)

        painter.setPen(QPen(QColor("#2196F3"), 2))
        painter.drawLine(white_x, 0, white_x, height)
        painter.end()


class HistogramPanel(QDockWidget):
    stretch_changed = pyqtSignal(float, float)
    mode_changed = pyqtSignal(object)
    reset_requested = pyqtSignal()
    apply_match_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__("直方图拉伸 (仅显示，不修改原始数据)", parent)
        self.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.RightDockWidgetArea)

        self._data_min: float = 0.0
        self._data_max: float = 65535.0

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.histogram_widget = HistogramWidget()
        layout.addWidget(self.histogram_widget)

        black_layout = QHBoxLayout()
        black_layout.addWidget(QLabel("黑点:"))
        self.slider_black = QSlider(Qt.Horizontal)
        self.slider_black.setRange(0, 1000)
        self.slider_black.setValue(0)
        black_layout.addWidget(self.slider_black, 1)
        self.spin_black = NoScrollDoubleSpinBox()
        self.spin_black.setDecimals(6)
        self.spin_black.setRange(-1_000_000_000.0, 1_000_000_000.0)
        self.spin_black.setValue(0.0)
        self.spin_black.setFixedWidth(96)
        black_layout.addWidget(self.spin_black)
        layout.addLayout(black_layout)

        white_layout = QHBoxLayout()
        white_layout.addWidget(QLabel("白点:"))
        self.slider_white = QSlider(Qt.Horizontal)
        self.slider_white.setRange(0, 1000)
        self.slider_white.setValue(1000)
        white_layout.addWidget(self.slider_white, 1)
        self.spin_white = NoScrollDoubleSpinBox()
        self.spin_white.setDecimals(6)
        self.spin_white.setRange(-1_000_000_000.0, 1_000_000_000.0)
        self.spin_white.setValue(65535.0)
        self.spin_white.setFixedWidth(96)
        white_layout.addWidget(self.spin_white)
        layout.addLayout(white_layout)

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("预设:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["线性", "对数", "平方根", "Asinh", "自动拉伸"])
        mode_layout.addWidget(self.combo_mode, 1)
        layout.addLayout(mode_layout)

        button_layout = QHBoxLayout()
        self.btn_reset = QPushButton("重置")
        self.btn_apply_all = QPushButton("匹配另外两图 (M)")
        self.btn_apply_all.setToolTip("根据当前手动调整后的 Min/Max，同步该任务组另外两张图")
        button_layout.addWidget(self.btn_reset)
        button_layout.addStretch()
        button_layout.addWidget(self.btn_apply_all)
        layout.addLayout(button_layout)

        self.setWidget(content)

        self.slider_black.valueChanged.connect(self._on_black_slider)
        self.slider_white.valueChanged.connect(self._on_white_slider)
        self.spin_black.valueChanged.connect(self._on_black_spin)
        self.spin_white.valueChanged.connect(self._on_white_spin)
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        self.btn_reset.clicked.connect(self._on_reset)
        self.btn_apply_all.clicked.connect(self.apply_match_requested)

    @property
    def black_point(self) -> float:
        return float(self.spin_black.value())

    @property
    def white_point(self) -> float:
        return float(self.spin_white.value())

    def set_data_range(self, data_min: float, data_max: float) -> None:
        self._data_min = float(data_min)
        self._data_max = float(data_max)
        self.set_stretch_values(self._data_min, self._data_max)

    def set_image_data(
        self,
        data: np.ndarray,
        black_point: float | None = None,
        white_point: float | None = None,
    ) -> None:
        if data is None:
            return

        flat = np.asarray(data, dtype=np.float64)
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            return

        hist, _ = np.histogram(flat, bins=256)
        self.histogram_widget.set_histogram(hist)

        self._data_min = float(flat.min())
        self._data_max = float(flat.max())
        self.set_stretch_values(
            self._data_min if black_point is None else float(black_point),
            self._data_max if white_point is None else float(white_point),
        )

    def set_stretch_values(self, black_point: float, white_point: float) -> None:
        black_value = float(min(black_point, white_point))
        white_value = float(max(black_point, white_point))
        black_norm, white_norm = self._normalize_points(black_value, white_value)
        spin_min = min(self._data_min, black_value, white_value)
        spin_max = max(self._data_max, black_value, white_value)

        self.slider_black.blockSignals(True)
        self.slider_white.blockSignals(True)
        self.spin_black.blockSignals(True)
        self.spin_white.blockSignals(True)
        self.spin_black.setRange(spin_min, spin_max)
        self.spin_white.setRange(spin_min, spin_max)
        self.slider_black.setValue(int(np.clip(black_norm, 0.0, 1.0) * 1000))
        self.slider_white.setValue(int(np.clip(white_norm, 0.0, 1.0) * 1000))
        self.spin_black.setValue(black_value)
        self.spin_white.setValue(white_value)
        self.spin_black.blockSignals(False)
        self.spin_white.blockSignals(False)
        self.slider_black.blockSignals(False)
        self.slider_white.blockSignals(False)
        self.histogram_widget.set_points(black_norm, white_norm)

    def _normalize_points(self, black_point: float, white_point: float) -> tuple[float, float]:
        if self._data_max <= self._data_min:
            return 0.0, 1.0
        black_norm = (black_point - self._data_min) / (self._data_max - self._data_min)
        white_norm = (white_point - self._data_min) / (self._data_max - self._data_min)
        return float(np.clip(black_norm, 0.0, 1.0)), float(np.clip(white_norm, 0.0, 1.0))

    def _on_black_slider(self, value: int) -> None:
        real = self._data_min + (value / 1000.0) * (self._data_max - self._data_min)
        self.spin_black.blockSignals(True)
        self.spin_black.setValue(float(real))
        self.spin_black.blockSignals(False)
        self.histogram_widget.set_points(value / 1000.0, self.slider_white.value() / 1000.0)
        self.stretch_changed.emit(float(real), self.white_point)

    def _on_white_slider(self, value: int) -> None:
        real = self._data_min + (value / 1000.0) * (self._data_max - self._data_min)
        self.spin_white.blockSignals(True)
        self.spin_white.setValue(float(real))
        self.spin_white.blockSignals(False)
        self.histogram_widget.set_points(self.slider_black.value() / 1000.0, value / 1000.0)
        self.stretch_changed.emit(self.black_point, float(real))

    def _on_black_spin(self, value: float) -> None:
        black_norm, white_norm = self._normalize_points(float(value), self.white_point)
        self.slider_black.blockSignals(True)
        self.slider_black.setValue(int(black_norm * 1000))
        self.slider_black.blockSignals(False)
        self.histogram_widget.set_points(black_norm, white_norm)
        self.stretch_changed.emit(float(value), self.white_point)

    def _on_white_spin(self, value: float) -> None:
        black_norm, white_norm = self._normalize_points(self.black_point, float(value))
        self.slider_white.blockSignals(True)
        self.slider_white.setValue(int(white_norm * 1000))
        self.slider_white.blockSignals(False)
        self.histogram_widget.set_points(black_norm, white_norm)
        self.stretch_changed.emit(self.black_point, float(value))

    def _on_mode_changed(self, index: int) -> None:
        modes = [StretchMode.LINEAR, StretchMode.LOG, StretchMode.SQRT, StretchMode.ASINH, StretchMode.AUTO]
        if 0 <= index < len(modes):
            self.mode_changed.emit(modes[index])

    def _on_reset(self) -> None:
        self.set_stretch_values(self._data_min, self._data_max)
        self.combo_mode.setCurrentIndex(0)
        self.reset_requested.emit()
