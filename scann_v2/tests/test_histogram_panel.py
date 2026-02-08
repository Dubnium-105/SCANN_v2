"""HistogramPanel 直方图拉伸面板 单元测试

TDD 测试:
1. 初始化 → 默认数据范围
2. set_data_range → 更新 spin 范围
3. set_image_data → 计算直方图 + 更新范围
4. black/white point 属性
5. slider ↔ spin 同步
6. stretch_changed 信号
7. mode_changed 信号
8. 重置
"""

import pytest
import numpy as np
from unittest.mock import Mock

from scann.gui.widgets.histogram_panel import HistogramPanel, HistogramWidget, StretchMode


@pytest.fixture
def panel(qapp):
    """创建 HistogramPanel 实例"""
    return HistogramPanel()


class TestHistogramPanelInit:
    """测试初始化"""

    def test_default_title(self, panel):
        assert "直方图" in panel.windowTitle()

    def test_default_data_range(self, panel):
        assert panel._data_min == 0.0
        assert panel._data_max == 65535.0

    def test_default_black_point(self, panel):
        assert panel.black_point == 0.0

    def test_default_white_point(self, panel):
        assert panel.white_point == 65535.0

    def test_initial_mode_linear(self, panel):
        assert panel.combo_mode.currentIndex() == 0
        assert panel.combo_mode.currentText() == "线性"


class TestSetDataRange:
    """测试数据范围设置"""

    def test_set_range_updates_spins(self, panel):
        panel.set_data_range(100.0, 50000.0)
        assert panel.spin_black.minimum() == 100
        assert panel.spin_white.maximum() == 50000

    def test_set_range_resets_values(self, panel):
        panel.set_data_range(500.0, 30000.0)
        assert panel.spin_black.value() == 500
        assert panel.spin_white.value() == 30000

    def test_set_range_stores_internally(self, panel):
        panel.set_data_range(200.0, 60000.0)
        assert panel._data_min == 200.0
        assert panel._data_max == 60000.0


class TestSetImageData:
    """测试图像数据设置"""

    def test_16bit_data(self, panel):
        data = np.array([[100, 200], [300, 400]], dtype=np.uint16)
        panel.set_image_data(data)
        assert panel._data_min == 100.0
        assert panel._data_max == 400.0

    def test_float_data(self, panel):
        data = np.array([[0.1, 0.5], [0.8, 0.9]], dtype=np.float32)
        panel.set_image_data(data)
        assert abs(panel._data_min - 0.1) < 0.01
        assert abs(panel._data_max - 0.9) < 0.01

    def test_histogram_widget_updated(self, panel):
        data = np.random.randint(0, 65535, (64, 64), dtype=np.uint16)
        panel.set_image_data(data)
        assert panel.histogram_widget._hist_data is not None
        assert len(panel.histogram_widget._hist_data) == 256

    def test_none_data_no_crash(self, panel):
        panel.set_image_data(None)  # 不应崩溃


class TestBlackWhitePoints:
    """测试黑白点属性"""

    def test_black_point_property(self, panel):
        panel.spin_black.setValue(1000)
        assert panel.black_point == 1000.0

    def test_white_point_property(self, panel):
        panel.spin_white.setValue(50000)
        assert panel.white_point == 50000.0


class TestStretchSignal:
    """测试拉伸参数变化信号"""

    def test_black_spin_emits_stretch_changed(self, panel):
        received = []
        panel.stretch_changed.connect(lambda b, w: received.append((b, w)))
        panel.spin_black.setValue(500)
        assert len(received) >= 1
        assert received[-1][0] == 500.0

    def test_white_spin_emits_stretch_changed(self, panel):
        received = []
        panel.stretch_changed.connect(lambda b, w: received.append((b, w)))
        panel.spin_white.setValue(40000)
        assert len(received) >= 1
        assert received[-1][1] == 40000.0


class TestModeChanged:
    """测试模式切换"""

    def test_mode_changed_signal(self, panel):
        received = []
        panel.mode_changed.connect(lambda m: received.append(m))
        panel.combo_mode.setCurrentIndex(1)  # 对数
        assert len(received) == 1
        assert received[0] == StretchMode.LOG

    def test_mode_sqrt(self, panel):
        received = []
        panel.mode_changed.connect(lambda m: received.append(m))
        panel.combo_mode.setCurrentIndex(2)
        assert received[-1] == StretchMode.SQRT

    def test_mode_auto(self, panel):
        received = []
        panel.mode_changed.connect(lambda m: received.append(m))
        panel.combo_mode.setCurrentIndex(4)
        assert received[-1] == StretchMode.AUTO


class TestReset:
    """测试重置"""

    def test_reset_restores_defaults(self, panel):
        panel.set_data_range(100.0, 50000.0)
        panel.spin_black.setValue(5000)
        panel.spin_white.setValue(30000)
        panel.combo_mode.setCurrentIndex(2)

        panel._on_reset()

        assert panel.spin_black.value() == 100
        assert panel.spin_white.value() == 50000
        assert panel.combo_mode.currentIndex() == 0

    def test_reset_emits_signal(self, panel):
        received = []
        panel.reset_requested.connect(lambda: received.append(True))
        panel._on_reset()
        assert len(received) == 1


class TestHistogramWidget:
    """测试直方图绘制 Widget"""

    def test_init(self, qapp):
        w = HistogramWidget()
        assert w._hist_data is None
        assert w._black_point == 0.0
        assert w._white_point == 1.0

    def test_set_histogram(self, qapp):
        w = HistogramWidget()
        hist = np.random.randint(0, 1000, 256)
        w.set_histogram(hist)
        assert w._hist_data is not None
        assert len(w._hist_data) == 256

    def test_set_points(self, qapp):
        w = HistogramWidget()
        w.set_points(0.2, 0.8)
        assert w._black_point == 0.2
        assert w._white_point == 0.8
