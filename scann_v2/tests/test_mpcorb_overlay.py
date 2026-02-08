"""MpcorbOverlay MPCORB 已知小行星叠加层 单元测试

TDD 测试:
1. 初始化 → 空列表
2. set_positions → 添加标记
3. clear → 清除所有标记
4. set_visible → 可见性
5. toggle → 切换可见性
6. 坐标转换异常 → 跳过该位置
"""

import pytest
from unittest.mock import Mock, MagicMock

from PyQt5.QtWidgets import QGraphicsScene

from scann.core.models import SkyPosition
from scann.gui.widgets.mpcorb_overlay import MpcorbOverlay


@pytest.fixture
def scene(qapp):
    """创建 QGraphicsScene"""
    return QGraphicsScene()


@pytest.fixture
def overlay(scene):
    """创建 MpcorbOverlay 实例"""
    return MpcorbOverlay(scene)


@pytest.fixture
def sample_positions():
    """样本天球坐标"""
    return [
        SkyPosition(ra=180.0, dec=45.0, mag=15.2, name="2024 AB1"),
        SkyPosition(ra=181.0, dec=44.0, mag=17.5, name="2024 CD2"),
        SkyPosition(ra=182.0, dec=43.0, name="2024 EF3"),  # 无星等
    ]


def mock_wcs_to_pixel(ra, dec):
    """模拟 WCS→像素坐标转换"""
    return (ra * 10, dec * 10)


class TestMpcorbOverlayInit:
    """测试初始化"""

    def test_empty_items(self, overlay):
        assert len(overlay._items) == 0

    def test_default_visible(self, overlay):
        assert overlay.is_visible is True


class TestSetPositions:
    """测试设置位置"""

    def test_adds_items(self, overlay, sample_positions):
        overlay.set_positions(sample_positions, mock_wcs_to_pixel)
        # 每个位置 = 1个圆 + 1个标签 = 2个item
        assert len(overlay._items) == 6

    def test_items_on_scene(self, overlay, scene, sample_positions):
        initial_count = len(scene.items())
        overlay.set_positions(sample_positions, mock_wcs_to_pixel)
        # 应该有 6 个新 item
        assert len(scene.items()) == initial_count + 6

    def test_replaces_previous(self, overlay, sample_positions):
        overlay.set_positions(sample_positions, mock_wcs_to_pixel)
        assert len(overlay._items) == 6
        # 重新设置
        overlay.set_positions(sample_positions[:1], mock_wcs_to_pixel)
        assert len(overlay._items) == 2

    def test_conversion_error_skips(self, overlay):
        def bad_wcs(ra, dec):
            raise ValueError("WCS not available")

        positions = [SkyPosition(ra=180.0, dec=45.0, name="test")]
        overlay.set_positions(positions, bad_wcs)
        assert len(overlay._items) == 0

    def test_partial_conversion_error(self, overlay):
        call_count = [0]

        def flaky_wcs(ra, dec):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("fail")
            return (ra * 10, dec * 10)

        positions = [
            SkyPosition(ra=1.0, dec=1.0, name="ok1"),
            SkyPosition(ra=2.0, dec=2.0, name="fail"),
            SkyPosition(ra=3.0, dec=3.0, name="ok2"),
        ]
        overlay.set_positions(positions, flaky_wcs)
        # 只有 2 个成功 → 4 个 item
        assert len(overlay._items) == 4

    def test_mag_in_label(self, overlay, sample_positions):
        overlay.set_positions(sample_positions, mock_wcs_to_pixel)
        # 第一个位置有 mag=15.2, 标签应包含 "(15.2)"
        # items 排列: [ellipse1, text1, ellipse2, text2, ...]
        text_item = overlay._items[1]  # 第一个文本
        assert "15.2" in text_item.toPlainText()

    def test_no_mag_in_label(self, overlay, sample_positions):
        overlay.set_positions(sample_positions, mock_wcs_to_pixel)
        text_item = overlay._items[5]  # 第三个文本 (无 mag)
        assert "(" not in text_item.toPlainText()


class TestClear:
    """测试清除"""

    def test_clear_removes_items(self, overlay, sample_positions):
        overlay.set_positions(sample_positions, mock_wcs_to_pixel)
        overlay.clear()
        assert len(overlay._items) == 0

    def test_clear_empty_no_crash(self, overlay):
        overlay.clear()  # 不应崩溃


class TestVisibility:
    """测试可见性"""

    def test_set_visible_false(self, overlay, sample_positions):
        overlay.set_positions(sample_positions, mock_wcs_to_pixel)
        overlay.set_visible(False)
        assert overlay.is_visible is False
        for item in overlay._items:
            assert not item.isVisible()

    def test_set_visible_true(self, overlay, sample_positions):
        overlay.set_positions(sample_positions, mock_wcs_to_pixel)
        overlay.set_visible(False)
        overlay.set_visible(True)
        assert overlay.is_visible is True
        for item in overlay._items:
            assert item.isVisible()

    def test_toggle(self, overlay):
        assert overlay.toggle() is False
        assert overlay.is_visible is False
        assert overlay.toggle() is True
        assert overlay.is_visible is True

    def test_new_items_respect_visibility(self, overlay, sample_positions):
        overlay.set_visible(False)
        overlay.set_positions(sample_positions, mock_wcs_to_pixel)
        for item in overlay._items:
            assert not item.isVisible()
