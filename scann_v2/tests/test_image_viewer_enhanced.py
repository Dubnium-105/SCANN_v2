"""FitsImageViewer 增强功能测试

TDD 测试:
1. center_on_point → centerOn + 可选缩放
2. fit_in_view → fitInView
3. draw_markers → 标记绘制 (选中十字线, 判决图标, 编号)
4. mouse_moved 信号 (鼠标追踪)
5. zoom_changed 信号 (滚轮缩放)
6. 缩放限制 (ZOOM_MIN / ZOOM_MAX)
7. scene_ref 属性
8. 键盘 F 键适配
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtWidgets import QGraphicsScene

from scann.core.models import Candidate, TargetVerdict
from scann.gui.image_viewer import FitsImageViewer


@pytest.fixture
def viewer(qapp):
    """创建 FitsImageViewer 实例"""
    v = FitsImageViewer()
    return v


class TestFitsImageViewerInit:
    """测试初始化"""

    def test_scene_exists(self, viewer):
        assert viewer.scene() is not None

    def test_scene_ref_property(self, viewer):
        assert viewer.scene_ref is viewer._scene

    def test_initial_zoom(self, viewer):
        assert viewer._zoom_level == 1.0

    def test_marker_items_empty(self, viewer):
        assert viewer._marker_items == []

    def test_mouse_tracking_enabled(self, viewer):
        assert viewer.hasMouseTracking()


class TestSetImageData:
    """测试图像加载"""

    def test_load_float32(self, viewer):
        data = np.random.rand(100, 100).astype(np.float32)
        viewer.set_image_data(data)
        assert not viewer._pixmap_item.pixmap().isNull()

    def test_load_uint16(self, viewer):
        data = (np.random.rand(50, 50) * 65535).astype(np.uint16)
        viewer.set_image_data(data)
        assert not viewer._pixmap_item.pixmap().isNull()

    def test_load_uint8(self, viewer):
        data = (np.random.rand(40, 40) * 255).astype(np.uint8)
        viewer.set_image_data(data)
        assert not viewer._pixmap_item.pixmap().isNull()

    def test_load_inverted(self, viewer):
        data = np.zeros((30, 30), np.float32)
        viewer.set_image_data(data, inverted=True)
        assert not viewer._pixmap_item.pixmap().isNull()

    def test_load_none_no_crash(self, viewer):
        viewer.set_image_data(None)  # 不应崩溃

    def test_load_rgb(self, viewer):
        data = (np.random.rand(30, 30, 3) * 255).astype(np.uint8)
        viewer.set_image_data(data)
        assert not viewer._pixmap_item.pixmap().isNull()


class TestCenterOnPoint:
    """测试 center_on_point"""

    def test_center_basic(self, viewer):
        # 加载图像以确保场景有内容
        data = np.zeros((200, 200), np.float32)
        viewer.set_image_data(data)
        viewer.center_on_point(100, 100)
        # 不应崩溃

    def test_center_with_zoom(self, viewer):
        data = np.zeros((200, 200), np.float32)
        viewer.set_image_data(data)
        viewer.center_on_point(50, 50, zoom_to=200)
        # 缩放应为 2.0 (200%)
        assert abs(viewer._zoom_level - 2.0) < 0.1

    def test_center_zoom_clamped_min(self, viewer):
        data = np.zeros((200, 200), np.float32)
        viewer.set_image_data(data)
        viewer.center_on_point(50, 50, zoom_to=1)  # 1% < ZOOM_MIN*100=5%
        assert viewer._zoom_level >= viewer.ZOOM_MIN

    def test_center_zoom_clamped_max(self, viewer):
        data = np.zeros((200, 200), np.float32)
        viewer.set_image_data(data)
        viewer.center_on_point(50, 50, zoom_to=5000)  # 5000% > ZOOM_MAX*100=2000%
        assert viewer._zoom_level <= viewer.ZOOM_MAX


class TestFitInView:
    """测试 fit_in_view"""

    def test_fit_in_view_nocrash(self, viewer):
        data = np.zeros((100, 100), np.float32)
        viewer.set_image_data(data)
        viewer.fit_in_view()
        # 不应崩溃

    def test_fit_in_view_empty_scene(self, viewer):
        viewer.fit_in_view()  # 空场景不应崩溃


class TestDrawMarkers:
    """测试候选标记绘制"""

    def test_draw_basic_markers(self, viewer):
        candidates = [
            Candidate(x=50, y=50, ai_score=0.9),
            Candidate(x=100, y=100, ai_score=0.3),
        ]
        viewer.draw_markers(candidates, selected_idx=0)
        assert len(viewer._marker_items) > 0

    def test_draw_clears_old_markers(self, viewer):
        c = [Candidate(x=50, y=50)]
        viewer.draw_markers(c)
        first_count = len(viewer._marker_items)
        viewer.draw_markers(c)
        # 重新绘制后标记数应相同 (旧的被清除了)
        assert len(viewer._marker_items) == first_count

    def test_hide_all(self, viewer):
        c = [Candidate(x=50, y=50)]
        viewer.draw_markers(c, hide_all=True)
        assert len(viewer._marker_items) == 0

    def test_selected_has_crosshair(self, viewer):
        c = [Candidate(x=50, y=50)]
        viewer.draw_markers(c, selected_idx=0)
        # 选中项: 1 ellipse + 2 crosshair lines + 1 text = 4 items
        assert len(viewer._marker_items) >= 4

    def test_verdict_real_shows_checkmark(self, viewer):
        c = [Candidate(x=50, y=50, verdict=TargetVerdict.REAL)]
        viewer.draw_markers(c, selected_idx=-1)
        # 应该有: 1 ellipse + 1 verdict text + 1 number text = 3
        assert len(viewer._marker_items) >= 3

    def test_verdict_bogus_shows_x(self, viewer):
        c = [Candidate(x=50, y=50, verdict=TargetVerdict.BOGUS)]
        viewer.draw_markers(c, selected_idx=-1)
        assert len(viewer._marker_items) >= 3

    def test_manual_candidate_color(self, viewer):
        c = [Candidate(x=50, y=50, is_manual=True)]
        viewer.draw_markers(c)
        assert len(viewer._marker_items) > 0  # 紫色标记

    def test_known_candidate_color(self, viewer):
        c = [Candidate(x=50, y=50, is_known=True)]
        viewer.draw_markers(c)
        assert len(viewer._marker_items) > 0  # 灰色标记


class TestZoomLimits:
    """测试缩放限制"""

    def test_zoom_min(self, viewer):
        assert viewer.ZOOM_MIN == 0.05

    def test_zoom_max(self, viewer):
        assert viewer.ZOOM_MAX == 20.0

    def test_zoom_factor(self, viewer):
        assert viewer.ZOOM_FACTOR == 1.25


class TestSignals:
    """测试信号"""

    def test_zoom_changed_signal_defined(self, viewer):
        received = []
        viewer.zoom_changed.connect(lambda v: received.append(v))
        # 手动调用 _emit_zoom 触发
        viewer._emit_zoom()
        assert len(received) == 1

    def test_point_clicked_signal_defined(self, viewer):
        # 验证信号存在 (能连接)
        received = []
        viewer.point_clicked.connect(lambda x, y: received.append((x, y)))
        viewer.point_clicked.emit(10, 20)
        assert received == [(10, 20)]

    def test_right_click_signal_defined(self, viewer):
        received = []
        viewer.right_click.connect(lambda x, y: received.append((x, y)))
        viewer.right_click.emit(30, 40)
        assert received == [(30, 40)]

    def test_mouse_moved_signal_defined(self, viewer):
        received = []
        viewer.mouse_moved.connect(lambda x, y: received.append((x, y)))
        viewer.mouse_moved.emit(50, 60)
        assert received == [(50, 60)]
