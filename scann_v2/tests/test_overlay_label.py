"""OverlayLabel 浮层状态标签 单元测试

TDD 测试:
1. 初始化 → 默认属性
2. set_state → 切换预定义颜色
3. set_color → 自定义颜色
4. show/hide → 可见性切换
5. 脉冲动画 → 启动/停止
"""

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

from scann.gui.widgets.overlay_label import OverlayLabel


class TestOverlayLabelInit:
    """测试初始化"""

    def test_default_text(self, qapp):
        label = OverlayLabel("NEW")
        assert label.text() == "NEW"

    def test_empty_text(self, qapp):
        label = OverlayLabel()
        assert label.text() == ""

    def test_default_visible(self, qapp):
        label = OverlayLabel("TEST")
        assert label._visible_flag is True

    def test_minimum_height(self, qapp):
        label = OverlayLabel("TEST")
        assert label.minimumHeight() >= 0

    def test_alignment_center(self, qapp):
        label = OverlayLabel("TEST")
        assert label.alignment() == Qt.AlignCenter


class TestOverlayLabelStates:
    """测试预定义状态颜色"""

    @pytest.mark.parametrize("state,expected_hex", [
        ("new", "#2196F3"),
        ("old", "#FF9800"),
        ("inv", "#9C27B0"),
        ("blink", "#FFEB3B"),
    ])
    def test_set_state_updates_bg_color(self, qapp, state, expected_hex):
        label = OverlayLabel("X")
        label.set_state(state)
        expected = QColor(expected_hex)
        expected.setAlpha(180)
        assert label._bg_color.red() == expected.red()
        assert label._bg_color.green() == expected.green()
        assert label._bg_color.blue() == expected.blue()

    def test_set_state_case_insensitive(self, qapp):
        label = OverlayLabel("X")
        label.set_state("NEW")
        assert label._bg_color.red() == QColor("#2196F3").red()

    def test_set_state_unknown_fallback(self, qapp):
        label = OverlayLabel("X")
        label.set_state("nonexistent")
        # 回退到默认蓝色
        assert label._bg_color.red() == QColor("#2196F3").red()


class TestOverlayLabelCustomColor:
    """测试自定义颜色"""

    def test_set_color_hex(self, qapp):
        label = OverlayLabel("X")
        label.set_color("#FF0000")
        assert label._bg_color.red() == 255
        assert label._bg_color.green() == 0
        assert label._bg_color.blue() == 0

    def test_set_color_alpha_is_180(self, qapp):
        label = OverlayLabel("X")
        label.set_color("#00FF00")
        assert label._bg_color.alpha() == 180


class TestOverlayLabelVisibility:
    """测试可见性"""

    def test_hide_label(self, qapp):
        label = OverlayLabel("X")
        label.hide_label()
        assert label._visible_flag is False
        assert not label.isVisible()

    def test_show_label(self, qapp):
        label = OverlayLabel("X")
        label.hide_label()
        label.show_label()
        assert label._visible_flag is True

    def test_hide_then_show(self, qapp):
        label = OverlayLabel("X")
        label.hide_label()
        assert label._visible_flag is False
        label.show_label()
        assert label._visible_flag is True


class TestOverlayLabelPulse:
    """测试脉冲动画"""

    def test_start_pulse_creates_timer(self, qapp):
        label = OverlayLabel("X")
        label.start_pulse(200)
        assert label._pulse_timer.isActive()

    def test_stop_pulse_stops_timer(self, qapp):
        label = OverlayLabel("X")
        label.start_pulse(200)
        label.stop_pulse()
        assert not label._pulse_timer.isActive()

    def test_stop_pulse_resets_visible(self, qapp):
        label = OverlayLabel("X")
        label.show_label()
        label.start_pulse(200)
        label.stop_pulse()
        assert label._pulse_on is True

    def test_pulse_toggles_on_state(self, qapp):
        label = OverlayLabel("X")
        label._pulse_on = True
        label._on_pulse()
        assert label._pulse_on is False
        label._on_pulse()
        assert label._pulse_on is True
