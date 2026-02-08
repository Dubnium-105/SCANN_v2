"""BlinkSpeedSlider 闪烁速度滑块 单元测试

TDD 测试:
1. 初始化 → 默认速度 500ms
2. 范围限制 → 50~2000ms
3. slider ↔ spin 同步
4. speed_changed 信号
5. speed_ms 属性 getter/setter
"""

import pytest

from scann.gui.widgets.blink_speed_slider import BlinkSpeedSlider


@pytest.fixture
def slider(qapp):
    """创建 BlinkSpeedSlider 实例"""
    return BlinkSpeedSlider()


class TestBlinkSpeedInit:
    """测试初始化"""

    def test_default_speed(self, slider):
        assert slider.speed_ms == 500

    def test_slider_default(self, slider):
        assert slider.slider.value() == 500

    def test_spin_default(self, slider):
        assert slider.spin.value() == 500

    def test_min_speed(self, slider):
        assert slider.MIN_SPEED == 50

    def test_max_speed(self, slider):
        assert slider.MAX_SPEED == 2000


class TestSpeedRange:
    """测试速度范围限制"""

    def test_slider_range(self, slider):
        assert slider.slider.minimum() == 50
        assert slider.slider.maximum() == 2000

    def test_spin_range(self, slider):
        assert slider.spin.minimum() == 50
        assert slider.spin.maximum() == 2000

    def test_speed_ms_setter_clamps_low(self, slider):
        slider.speed_ms = 10
        assert slider.speed_ms == 50

    def test_speed_ms_setter_clamps_high(self, slider):
        slider.speed_ms = 5000
        assert slider.speed_ms == 2000


class TestSliderSpinSync:
    """测试 slider ↔ spin 同步"""

    def test_slider_changes_spin(self, slider):
        slider.slider.setValue(800)
        assert slider.spin.value() == 800

    def test_spin_changes_slider(self, slider):
        slider.spin.setValue(1200)
        assert slider.slider.value() == 1200


class TestSpeedChangedSignal:
    """测试 speed_changed 信号"""

    def test_slider_emits_signal(self, slider):
        received = []
        slider.speed_changed.connect(lambda v: received.append(v))
        slider.slider.setValue(700)
        assert 700 in received

    def test_spin_emits_signal(self, slider):
        received = []
        slider.speed_changed.connect(lambda v: received.append(v))
        slider.spin.setValue(1000)
        assert 1000 in received

    def test_no_duplicate_signal(self, slider):
        """slider 变更 spin 时不应发出重复信号"""
        received = []
        slider.speed_changed.connect(lambda v: received.append(v))
        slider.slider.setValue(600)
        # 只应收到 1 次 (slider 发出, spin blockSignals)
        assert received.count(600) == 1


class TestSpeedMsProperty:
    """测试 speed_ms 属性"""

    def test_getter(self, slider):
        slider.spin.setValue(900)
        assert slider.speed_ms == 900

    def test_setter(self, slider):
        slider.speed_ms = 1500
        assert slider.spin.value() == 1500
        assert slider.speed_ms == 1500

    def test_setter_syncs_slider(self, slider):
        slider.speed_ms = 300
        # setter 设置 spin → spin 信号 → 更新 slider
        assert slider.spin.value() == 300
