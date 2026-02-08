"""闪烁速度控制滑块

带数值显示的闪烁速度控制 (50ms ~ 2000ms)。
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QSlider, QWidget

from scann.gui.widgets.no_scroll_spinbox import NoScrollSpinBox


class BlinkSpeedSlider(QWidget):
    """闪烁速度滑块 + 数值显示

    Signals:
        speed_changed(int): 速度改变 (毫秒)
    """

    speed_changed = pyqtSignal(int)

    MIN_SPEED = 50
    MAX_SPEED = 2000
    DEFAULT_SPEED = 500

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # 速度图标
        self.lbl_icon = QLabel("●")
        self.lbl_icon.setStyleSheet("color: #FFEB3B;")
        layout.addWidget(self.lbl_icon)

        # 滑块
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(self.MIN_SPEED, self.MAX_SPEED)
        self.slider.setValue(self.DEFAULT_SPEED)
        self.slider.setSingleStep(50)
        self.slider.setPageStep(100)
        self.slider.setFixedWidth(100)
        layout.addWidget(self.slider)

        # 数值 SpinBox (禁用滚轮)
        self.spin = NoScrollSpinBox()
        self.spin.setRange(self.MIN_SPEED, self.MAX_SPEED)
        self.spin.setValue(self.DEFAULT_SPEED)
        self.spin.setSuffix(" ms")
        self.spin.setSingleStep(50)
        self.spin.setFixedWidth(80)
        layout.addWidget(self.spin)

        # 信号连接
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.spin.valueChanged.connect(self._on_spin_changed)

    @property
    def speed_ms(self) -> int:
        return self.spin.value()

    @speed_ms.setter
    def speed_ms(self, value: int) -> None:
        value = max(self.MIN_SPEED, min(self.MAX_SPEED, value))
        self.spin.setValue(value)

    def _on_slider_changed(self, value: int) -> None:
        self.spin.blockSignals(True)
        self.spin.setValue(value)
        self.spin.blockSignals(False)
        self.speed_changed.emit(value)

    def _on_spin_changed(self, value: int) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(value)
        self.slider.blockSignals(False)
        self.speed_changed.emit(value)
