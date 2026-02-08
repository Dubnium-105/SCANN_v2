"""主窗口功能测试 (重写版)

覆盖新 MainWindow API:
- 菜单栏 (7 个菜单)
- 侧边栏 (CollapsibleSidebar)
- 浮层标签 (overlay_state, overlay_inv, overlay_blink)
- 控制栏 (按钮 + 滑块)
- 状态栏 (CoordinateLabel × 2)
- 图像切换 / 闪烁 / 反色
- 候选标记 / 导航
- 公共 API: set_image_data, set_candidates
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from scann.core.models import Candidate, TargetVerdict
from scann.services.blink_service import BlinkState


# ═══════════════════════════════════════════════
#  辅助: 创建 Mock 化的 MainWindow 实例
# ═══════════════════════════════════════════════

def _make_mock_window():
    """创建一个跳过 __init__ 的 MainWindow, 手动挂载 Mock 属性"""
    from scann.gui.main_window import MainWindow

    with patch("scann.gui.main_window.QMainWindow.__init__"):
        w = MainWindow.__new__(MainWindow)

    # 图像查看器
    w.image_viewer = Mock()

    # 闪烁服务
    w.blink_service = Mock()
    w.blink_service.is_inverted = False
    w.blink_service.is_running = False
    w.blink_service.speed_ms = 500
    w.blink_service.current_state = BlinkState.NEW

    # 定时器
    w.blink_timer = Mock()

    # 浮层标签
    w.overlay_state = Mock()
    w.overlay_inv = Mock()
    w.overlay_blink = Mock()

    # 控制栏按钮
    w.btn_show_new = Mock()
    w.btn_show_old = Mock()
    w.btn_blink = Mock()
    w.btn_invert = Mock()
    w.btn_mark_real = Mock()
    w.btn_mark_bogus = Mock()
    w.btn_next_candidate = Mock()

    # 侧边栏
    w.sidebar = Mock()

    # 候选表格
    w.suspect_table = Mock()

    # 闪烁速度
    w.blink_speed = Mock()

    # 直方图面板
    w.histogram_panel = Mock()

    # 文件列表
    w.file_list = Mock()

    # 状态栏
    w.status_image_type = Mock()
    w.status_pixel_coord = Mock()
    w.status_wcs_coord = Mock()
    w.status_zoom = Mock()

    # 动作
    w.act_show_markers = Mock()
    w.act_show_markers.isChecked.return_value = True

    # statusBar mock
    w.statusBar = Mock(return_value=Mock())

    # 数据
    w._candidates = []
    w._current_candidate_idx = -1
    w._new_image_data = None
    w._old_image_data = None

    return w


# ═══════════════════════════════════════════════
#  图像切换
# ═══════════════════════════════════════════════


class TestShowImage:
    """测试 _show_image 统一入口"""

    def test_show_new(self):
        w = _make_mock_window()
        w._new_image_data = np.zeros((64, 64), np.float32)
        w._show_image("new")
        w.image_viewer.set_image_data.assert_called_once()
        w.overlay_state.setText.assert_called_with("NEW")
        w.overlay_state.set_state.assert_called_with("new")

    def test_show_old(self):
        w = _make_mock_window()
        w._old_image_data = np.zeros((64, 64), np.float32)
        w._show_image("old")
        w.image_viewer.set_image_data.assert_called_once()
        w.overlay_state.setText.assert_called_with("OLD")
        w.overlay_state.set_state.assert_called_with("old")

    def test_show_new_none_data(self):
        w = _make_mock_window()
        w._new_image_data = None
        w._show_image("new")
        w.image_viewer.set_image_data.assert_not_called()
        w.overlay_state.setText.assert_called()

    def test_show_inverted(self):
        w = _make_mock_window()
        w.blink_service.is_inverted = True
        w._new_image_data = np.zeros((32, 32), np.float32)
        w._show_image("new")
        w.image_viewer.set_image_data.assert_called_once_with(
            w._new_image_data, inverted=True
        )


class TestOnShowNewOld:
    """测试 _on_show_new / _on_show_old"""

    def test_on_show_new_sets_buttons(self):
        w = _make_mock_window()
        w._new_image_data = np.zeros((32, 32), np.float32)
        w._on_show_new()
        w.btn_show_new.setChecked.assert_called_with(True)
        w.btn_show_old.setChecked.assert_called_with(False)

    def test_on_show_old_sets_buttons(self):
        w = _make_mock_window()
        w._old_image_data = np.zeros((32, 32), np.float32)
        w._on_show_old()
        w.btn_show_new.setChecked.assert_called_with(False)
        w.btn_show_old.setChecked.assert_called_with(True)


# ═══════════════════════════════════════════════
#  闪烁模式
# ═══════════════════════════════════════════════


class TestBlinkMode:
    """测试闪烁模式"""

    def test_toggle_starts_timer(self):
        w = _make_mock_window()
        w.blink_service.toggle.return_value = True
        w.blink_service.speed_ms = 400
        w._on_blink_toggle()
        w.blink_timer.setInterval.assert_called_with(400)
        w.blink_timer.start.assert_called_once()
        w.btn_blink.setChecked.assert_called_with(True)
        w.overlay_blink.show_label.assert_called_once()
        w.overlay_blink.start_pulse.assert_called_once()

    def test_toggle_stops_timer(self):
        w = _make_mock_window()
        w.blink_service.toggle.return_value = False
        w._on_blink_toggle()
        w.blink_timer.stop.assert_called_once()
        w.btn_blink.setChecked.assert_called_with(False)
        w.overlay_blink.stop_pulse.assert_called_once()
        w.overlay_blink.hide_label.assert_called_once()

    def test_blink_tick_new(self):
        w = _make_mock_window()
        w._new_image_data = np.zeros((32, 32), np.float32)
        w.blink_service.tick.return_value = BlinkState.NEW
        w._on_blink_tick()
        w.overlay_state.set_state.assert_called_with("new")

    def test_blink_tick_old(self):
        w = _make_mock_window()
        w._old_image_data = np.zeros((32, 32), np.float32)
        w.blink_service.tick.return_value = BlinkState.OLD
        w._on_blink_tick()
        w.overlay_state.set_state.assert_called_with("old")

    def test_blink_speed_changed(self):
        w = _make_mock_window()
        w.blink_service.is_running = True
        w._on_blink_speed_changed(300)
        assert w.blink_service.speed_ms == 300
        w.blink_timer.setInterval.assert_called_with(300)

    def test_blink_speed_changed_not_running(self):
        w = _make_mock_window()
        w.blink_service.is_running = False
        w._on_blink_speed_changed(300)
        w.blink_timer.setInterval.assert_not_called()


# ═══════════════════════════════════════════════
#  反色
# ═══════════════════════════════════════════════


class TestInvertDisplay:
    """测试反色切换"""

    def test_invert_shows_overlay(self):
        w = _make_mock_window()
        w._new_image_data = np.zeros((32, 32), np.float32)
        w.blink_service.toggle_invert.return_value = True
        w.blink_service.current_state = BlinkState.NEW
        w._on_invert_toggle()
        w.overlay_inv.show_label.assert_called_once()
        w.btn_invert.setChecked.assert_called_with(True)

    def test_uninvert_hides_overlay(self):
        w = _make_mock_window()
        w._new_image_data = np.zeros((32, 32), np.float32)
        w.blink_service.toggle_invert.return_value = False
        w.blink_service.current_state = BlinkState.NEW
        w._on_invert_toggle()
        w.overlay_inv.hide_label.assert_called_once()
        w.btn_invert.setChecked.assert_called_with(False)

    def test_invert_refreshes_display(self):
        w = _make_mock_window()
        w.blink_service.toggle_invert.return_value = True
        w.blink_service.current_state = BlinkState.OLD
        w._old_image_data = np.zeros((32, 32), np.float32)
        w._on_invert_toggle()
        w.image_viewer.set_image_data.assert_called_once()


# ═══════════════════════════════════════════════
#  候选标记
# ═══════════════════════════════════════════════


class TestCandidateMarking:
    """测试候选标记 (真/假)"""

    def test_mark_real(self):
        w = _make_mock_window()
        cand = Candidate(x=100, y=200)
        w._candidates = [cand]
        w._current_candidate_idx = 0
        w._on_mark_real()
        assert cand.verdict == TargetVerdict.REAL
        w.suspect_table.update_candidate.assert_called_with(0)

    def test_mark_bogus(self):
        w = _make_mock_window()
        cand = Candidate(x=100, y=200)
        w._candidates = [cand]
        w._current_candidate_idx = 0
        w._on_mark_bogus()
        assert cand.verdict == TargetVerdict.BOGUS
        w.suspect_table.update_candidate.assert_called_with(0)

    def test_mark_empty_list_no_crash(self):
        w = _make_mock_window()
        w._candidates = []
        w._current_candidate_idx = -1
        w._on_mark_real()  # 不应崩溃
        w.suspect_table.update_candidate.assert_not_called()

    def test_mark_out_of_range_no_crash(self):
        w = _make_mock_window()
        w._candidates = [Candidate(x=1, y=1)]
        w._current_candidate_idx = 5  # 越界
        w._on_mark_bogus()
        w.suspect_table.update_candidate.assert_not_called()

    def test_mark_shows_status_message(self):
        w = _make_mock_window()
        w._candidates = [Candidate(x=10, y=20)]
        w._current_candidate_idx = 0
        w._on_mark_real()
        w.statusBar().showMessage.assert_called_once()
        msg = w.statusBar().showMessage.call_args[0][0]
        assert "真目标" in msg


# ═══════════════════════════════════════════════
#  候选导航
# ═══════════════════════════════════════════════


class TestCandidateNavigation:
    """测试候选导航"""

    def test_next_candidate_cycles(self):
        w = _make_mock_window()
        w._candidates = [Candidate(x=1, y=1), Candidate(x=2, y=2), Candidate(x=3, y=3)]
        w._current_candidate_idx = 0

        w._on_next_candidate()
        assert w._current_candidate_idx == 1
        w._on_next_candidate()
        assert w._current_candidate_idx == 2
        w._on_next_candidate()
        assert w._current_candidate_idx == 0  # 循环

    def test_next_candidate_empty_list(self):
        w = _make_mock_window()
        w._candidates = []
        w._on_next_candidate()  # 不应崩溃

    def test_focus_candidate_centers_view(self):
        w = _make_mock_window()
        cand = Candidate(x=150, y=250)
        w._candidates = [cand]
        w._focus_candidate(0)
        w.image_viewer.center_on_point.assert_called_with(150, 250)

    def test_candidate_selected_from_table(self):
        w = _make_mock_window()
        cand = Candidate(x=50, y=60)
        w._candidates = [cand]
        w._on_candidate_selected(0)
        assert w._current_candidate_idx == 0
        w.image_viewer.center_on_point.assert_called()

    def test_candidate_double_clicked_zooms(self):
        w = _make_mock_window()
        cand = Candidate(x=50, y=60)
        w._candidates = [cand]
        w._on_candidate_double_clicked(0)
        w.image_viewer.center_on_point.assert_called_with(50, 60, zoom_to=200)


# ═══════════════════════════════════════════════
#  配对导航
# ═══════════════════════════════════════════════


class TestPairNavigation:
    """测试配对导航 (← →)"""

    def test_prev_pair(self):
        w = _make_mock_window()
        w.file_list.currentRow.return_value = 2
        w._on_prev_pair()
        w.file_list.setCurrentRow.assert_called_with(1)

    def test_prev_pair_at_zero(self):
        w = _make_mock_window()
        w.file_list.currentRow.return_value = 0
        w._on_prev_pair()
        w.file_list.setCurrentRow.assert_not_called()

    def test_next_pair(self):
        w = _make_mock_window()
        w.file_list.currentRow.return_value = 1
        w.file_list.count.return_value = 5
        w._on_next_pair()
        w.file_list.setCurrentRow.assert_called_with(2)

    def test_next_pair_at_last(self):
        w = _make_mock_window()
        w.file_list.currentRow.return_value = 4
        w.file_list.count.return_value = 5
        w._on_next_pair()
        w.file_list.setCurrentRow.assert_not_called()


# ═══════════════════════════════════════════════
#  公共 API
# ═══════════════════════════════════════════════


class TestPublicAPI:
    """测试公共 API"""

    def test_set_image_data(self):
        w = _make_mock_window()
        new_data = np.zeros((64, 64), np.float32)
        old_data = np.ones((64, 64), np.float32)
        w.set_image_data(new_data, old_data)
        assert w._new_image_data is new_data
        assert w._old_image_data is old_data
        w.image_viewer.set_image_data.assert_called()

    def test_set_image_data_updates_histogram(self):
        w = _make_mock_window()
        data = np.zeros((32, 32), np.float32)
        w.set_image_data(data, None)
        w.histogram_panel.set_image_data.assert_called_with(data)

    def test_set_candidates(self):
        w = _make_mock_window()
        cands = [Candidate(x=10, y=20), Candidate(x=30, y=40)]
        w.set_candidates(cands)
        assert w._candidates is cands
        assert w._current_candidate_idx == 0
        w.suspect_table.set_candidates.assert_called_with(cands)

    def test_set_candidates_empty(self):
        w = _make_mock_window()
        w.set_candidates([])
        assert w._current_candidate_idx == -1


# ═══════════════════════════════════════════════
#  直方图面板切换
# ═══════════════════════════════════════════════


class TestHistogramToggle:
    """测试直方图面板切换"""

    def test_toggle_histogram_shows(self):
        w = _make_mock_window()
        w.histogram_panel.isVisible.return_value = False
        w._on_toggle_histogram()
        w.histogram_panel.setVisible.assert_called_with(True)

    def test_toggle_histogram_hides(self):
        w = _make_mock_window()
        w.histogram_panel.isVisible.return_value = True
        w._on_toggle_histogram()
        w.histogram_panel.setVisible.assert_called_with(False)
