"""ImageSessionController 单元测试。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from scann.core.models import FitsHeader, SkyPosition
from scann.gui.controllers import ImageSessionController
from scann.services.blink_service import BlinkState


def _make_window():
    window = Mock()
    window._config = SimpleNamespace(blink_speed_ms=500)
    window._new_image_data = None
    window._old_image_data = None
    window._new_fits_header = None
    window.image_viewer = Mock()
    window.histogram_panel = Mock()
    window.histogram_panel.black_point = 0.0
    window.histogram_panel.white_point = 1.0
    window.overlay_state = Mock()
    window.overlay_inv = Mock()
    window.status_image_type = Mock()
    window.status_pixel_coord = Mock()
    window.status_wcs_coord = Mock()
    window.status_zoom = Mock()
    window.btn_show_new = Mock()
    window.btn_show_old = Mock()
    window.btn_blink = Mock()
    window.btn_invert = Mock()
    window.blink_timer = Mock()
    window.blink_service = Mock()
    window.blink_service.current_state = BlinkState.NEW
    window.blink_service.is_inverted = False
    window.blink_service.is_running = False
    window.blink_service.speed_ms = 500
    return window


class TestImageSessionController:
    def test_show_image_updates_viewer_and_status(self):
        window = _make_window()
        controller = ImageSessionController(window)
        window._new_image_data = np.ones((16, 16), dtype=np.float32)

        with patch(
            "scann.gui.controllers.image_session_controller.histogram_stretch",
            return_value=np.zeros((16, 16), dtype=np.float32),
        ) as mock_stretch:
            controller.show_image("new")

        mock_stretch.assert_called_once()
        window.histogram_panel.set_image_data.assert_called_once_with(window._new_image_data)
        window.image_viewer.set_image_data.assert_called_once()
        window.overlay_state.setText.assert_called_once_with("NEW")
        window.overlay_state.set_state.assert_called_once_with("new")
        window.status_image_type.setText.assert_called_once_with("当前: NEW")

    def test_show_image_with_missing_data_updates_overlay_only(self):
        window = _make_window()
        controller = ImageSessionController(window)

        controller.show_image("old")

        window.overlay_state.setText.assert_called_once_with("无OLD")
        window.image_viewer.set_image_data.assert_not_called()

    def test_toggle_blink_starts_and_stops_timer(self):
        window = _make_window()
        controller = ImageSessionController(window)
        window.blink_service.toggle.side_effect = [True, False]

        controller.toggle_blink()
        controller.toggle_blink()

        window.btn_blink.setChecked.assert_any_call(True)
        window.btn_blink.setChecked.assert_any_call(False)
        window.blink_timer.setInterval.assert_called_once_with(500)
        window.blink_timer.start.assert_called_once_with()
        window.blink_timer.stop.assert_called_once_with()

    def test_stretch_changed_uses_current_image_state(self):
        window = _make_window()
        controller = ImageSessionController(window)
        window._old_image_data = np.ones((8, 8), dtype=np.float32)
        window.blink_service.current_state = BlinkState.OLD
        window.blink_service.is_inverted = True

        with patch(
            "scann.gui.controllers.image_session_controller.histogram_stretch",
            return_value=np.full((8, 8), 2.0, dtype=np.float32),
        ) as mock_stretch:
            controller.stretch_changed(10.0, 20.0)

        mock_stretch.assert_called_once_with(
            window._old_image_data,
            black_point=10.0,
            white_point=20.0,
        )
        window.image_viewer.set_image_data.assert_called_once()
        args, kwargs = window.image_viewer.set_image_data.call_args
        np.testing.assert_array_equal(args[0], np.full((8, 8), 2.0, dtype=np.float32))
        assert kwargs == {"inverted": True}

    def test_mouse_moved_updates_pixel_and_wcs(self):
        window = _make_window()
        controller = ImageSessionController(window)
        window._new_fits_header = FitsHeader(raw={"CTYPE1": "RA---TAN"})

        with patch(
            "scann.gui.controllers.image_session_controller.pixel_to_wcs",
            return_value=SkyPosition(ra=180.5, dec=45.3),
        ) as mock_pixel_to_wcs:
            controller.mouse_moved(64, 32)

        mock_pixel_to_wcs.assert_called_once_with(64, 32, window._new_fits_header)
        window.status_pixel_coord.set_pixel_coordinates.assert_called_once_with(64, 32)
        window.status_wcs_coord.set_wcs_coordinates.assert_called_once()

    def test_set_image_data_updates_window_state_and_refreshes(self):
        window = _make_window()
        controller = ImageSessionController(window)
        new_data = np.ones((12, 12), dtype=np.float32)
        old_data = np.zeros((12, 12), dtype=np.float32)

        with patch(
            "scann.gui.controllers.image_session_controller.histogram_stretch",
            return_value=new_data,
        ):
            controller.set_image_data(new_data, old_data)

        assert window._new_image_data is new_data
        assert window._old_image_data is old_data
        window.blink_service.set_state.assert_called_once_with(BlinkState.NEW)
        assert window.histogram_panel.set_image_data.call_count >= 1
