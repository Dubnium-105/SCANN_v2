"""半透明浮层状态标签

在图像查看器上显示 NEW/OLD/INV 等状态指示，
使用半透明背景，不遮挡图像主体。
"""

from __future__ import annotations

from PyQt5.QtCore import QPropertyAnimation, QTimer, Qt
from PyQt5.QtGui import QColor, QFont, QPainter
from PyQt5.QtWidgets import QLabel, QWidget


class OverlayLabel(QLabel):
    """半透明浮层状态标签

    用法:
        overlay = OverlayLabel("NEW", parent=image_viewer)
        overlay.set_color("#2196F3")
        overlay.move(10, 10)  # 左上角
    """

    # 预定义颜色方案
    COLORS = {
        "new": "#2196F3",   # 蓝色 = 新图
        "old": "#FF9800",   # 橙色 = 旧图
        "inv": "#9C27B0",   # 紫色 = 反色
        "blink": "#FFEB3B", # 黄色 = 闪烁中
    }

    def __init__(self, text: str = "", parent: QWidget | None = None):
        super().__init__(text, parent)
        self._bg_color = QColor(33, 150, 243, 180)  # 默认蓝色半透明
        self._text_color = QColor(255, 255, 255)
        self._visible_flag = True

        # 字体
        font = QFont("Arial", 14, QFont.Bold)
        self.setFont(font)
        self.setAlignment(Qt.AlignCenter)

        # 固定大小
        self.setFixedHeight(28)
        self.setMinimumWidth(50)

        # 闪烁动画定时器 (用于闪烁状态指示)
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._on_pulse)
        self._pulse_on = True

        self._update_style()

    def set_color(self, color: str) -> None:
        """设置背景颜色 (十六进制)"""
        c = QColor(color)
        c.setAlpha(180)
        self._bg_color = c
        self._update_style()

    def set_state(self, state: str) -> None:
        """根据预定义状态设置颜色

        Args:
            state: "new", "old", "inv", "blink"
        """
        color = self.COLORS.get(state.lower(), "#2196F3")
        self.set_color(color)

    def start_pulse(self, interval_ms: int = 500) -> None:
        """开始脉冲闪烁效果 (用于闪烁状态指示)"""
        self._pulse_timer.start(interval_ms)

    def stop_pulse(self) -> None:
        """停止脉冲"""
        self._pulse_timer.stop()
        self._pulse_on = True
        self.setVisible(self._visible_flag)

    def _on_pulse(self) -> None:
        """脉冲回调"""
        self._pulse_on = not self._pulse_on
        self.setVisible(self._pulse_on and self._visible_flag)

    def show_label(self) -> None:
        """显示标签"""
        self._visible_flag = True
        self.setVisible(True)

    def hide_label(self) -> None:
        """隐藏标签"""
        self._visible_flag = False
        self.setVisible(False)

    def set_file_name(self, name: str, match_found: bool = True) -> None:
        """在标签旁显示文件名及匹配状态

        Args:
            name: 文件名
            match_found: 是否找到对应的新/旧图
        """
        icon = "✓" if match_found else "✗"
        self.setText(f"{self.text().split(' |')[0]} | {icon} {name}")

    def _update_style(self) -> None:
        """更新样式表"""
        bg = self._bg_color
        self.setStyleSheet(
            f"QLabel {{"
            f"  background-color: rgba({bg.red()}, {bg.green()}, {bg.blue()}, {bg.alpha()});"
            f"  color: rgba({self._text_color.red()}, {self._text_color.green()}, "
            f"              {self._text_color.blue()}, {self._text_color.alpha()});"
            f"  border-radius: 4px;"
            f"  padding: 2px 8px;"
            f"}}"
        )
