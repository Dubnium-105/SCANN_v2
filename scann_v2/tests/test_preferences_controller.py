"""PreferencesController 与 ConfigService 的回归测试。"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from PyQt5.QtWidgets import QMessageBox

from scann.core.models import AppConfig
from scann.gui.controllers.preferences_controller import PreferencesController
from scann.services.config_service import ConfigService


def _make_window() -> SimpleNamespace:
    config = AppConfig()
    config.show_markers = True
    config.show_mpcorb = False
    config.show_known_objects = True
    config.histogram_visible = False
    config.sidebar_width = 240
    config.sidebar_collapsed = False
    config.auto_collapse_sidebar = True
    config.confirm_before_close = True

    window = SimpleNamespace()
    window._config = config
    window._new_folder = "new-dir"
    window._old_folder = "old-dir"
    window._logger = Mock()
    window._show_message = Mock()
    window.blink_service = SimpleNamespace(speed_ms=500)
    window.model_controller = Mock()
    window.histogram_panel = Mock()
    window.histogram_panel.black_point = 0.2
    window.histogram_panel.white_point = 0.9
    window.histogram_panel.combo_mode.currentIndex.return_value = 2
    window.histogram_panel.isVisible.return_value = True
    window.blink_speed = SimpleNamespace(speed_ms=0)
    window.act_show_markers = Mock()
    window.act_show_markers.isChecked.return_value = False
    window.act_show_mpcorb = Mock()
    window.act_show_mpcorb.isChecked.return_value = True
    window.act_show_known = Mock()
    window.act_show_known.isChecked.return_value = False
    window.sidebar = Mock()
    window.sidebar.is_collapsed = True
    window.sidebar.preferred_width = 320
    window.width = Mock(return_value=1440)
    window.height = Mock(return_value=960)
    window.main_splitter = Mock()
    window.overlay_state = Mock()
    window.overlay_inv = Mock()
    window.overlay_blink = Mock()
    window.image_viewer = Mock()
    window.image_viewer.width.return_value = 800
    window.image_viewer.height.return_value = 600
    return window


class TestConfigService:
    def test_load_config_delegates_to_core_loader(self) -> None:
        service = ConfigService()
        expected = AppConfig()

        with patch("scann.services.config_service.core_load_config", return_value=expected) as loader:
            result = service.load_config("demo.json")

        loader.assert_called_once_with("demo.json")
        assert result is expected

    def test_save_config_delegates_to_core_saver(self, tmp_path) -> None:
        service = ConfigService()
        config = AppConfig()
        output = tmp_path / "config.json"

        with patch("scann.services.config_service.core_save_config", return_value=output) as saver:
            result = service.save_config(config, output)

        saver.assert_called_once_with(config, output)
        assert result == output


class TestPreferencesController:
    @patch("scann.gui.dialogs.settings_dialog.SettingsDialog")
    def test_open_preferences_saves_and_syncs_runtime(self, mock_dialog_cls) -> None:
        window = _make_window()
        config_service = Mock(spec=ConfigService)
        controller = PreferencesController(window, config_service)

        window._config.ai_confidence = 0.17
        window._config.batch_size = 128

        dialog = Mock()
        dialog.exec_.return_value = True
        mock_dialog_cls.return_value = dialog

        controller.open_preferences()

        config_service.save_config.assert_called_once_with(window._config)
        assert window.blink_service.speed_ms == window._config.blink_speed_ms
        window.model_controller.apply_runtime_config.assert_called_once_with()
        window._show_message.assert_called_once_with("设置已保存")

    def test_save_runtime_state_writes_window_state_back_to_config(self) -> None:
        window = _make_window()
        controller = PreferencesController(window, Mock(spec=ConfigService))

        controller.save_runtime_state()

        assert window._config.new_folder == "new-dir"
        assert window._config.old_folder == "old-dir"
        assert window._config.blink_speed_ms == 500
        assert window._config.stretch_black_point == pytest.approx(0.2)
        assert window._config.stretch_white_point == pytest.approx(0.9)
        assert window._config.stretch_mode == "平方根"
        assert window._config.show_markers is False
        assert window._config.show_mpcorb is True
        assert window._config.show_known_objects is False
        assert window._config.histogram_visible is True
        assert window._config.sidebar_collapsed is True
        assert window._config.sidebar_width == 320
        assert window._config.window_width == 1440
        assert window._config.window_height == 960

    def test_restore_ui_state_applies_config_back_to_window(self) -> None:
        window = _make_window()
        window._config.blink_speed_ms = 750
        window._config.show_markers = True
        window._config.show_mpcorb = False
        window._config.show_known_objects = True
        window._config.histogram_visible = True
        window._config.stretch_mode = "Asinh"
        window._config.sidebar_width = 280
        window._config.sidebar_collapsed = True
        controller = PreferencesController(window, Mock(spec=ConfigService))

        controller.restore_ui_state()

        assert window.blink_speed.speed_ms == 750
        assert window.blink_service.speed_ms == 750
        window.act_show_markers.setChecked.assert_called_once_with(True)
        window.act_show_mpcorb.setChecked.assert_called_once_with(False)
        window.act_show_known.setChecked.assert_called_once_with(True)
        window.histogram_panel.setVisible.assert_called_once_with(True)
        window.histogram_panel.combo_mode.setCurrentIndex.assert_called_once_with(3)
        window.sidebar.set_preferred_width.assert_called_once_with(280)
        window.main_splitter.setSizes.assert_called_once()
        window.sidebar.collapse.assert_called_once_with()

    @patch("scann.gui.controllers.preferences_controller.QMessageBox.question")
    def test_handle_close_event_can_cancel_close(self, mock_question) -> None:
        window = _make_window()
        event = Mock()
        config_service = Mock(spec=ConfigService)
        controller = PreferencesController(window, config_service)
        mock_question.return_value = QMessageBox.No

        should_close = controller.handle_close_event(event)

        assert should_close is False
        event.ignore.assert_called_once_with()
        config_service.save_config.assert_not_called()

    @patch("scann.gui.controllers.preferences_controller.QMessageBox.question")
    def test_handle_close_event_saves_config_when_confirmed(self, mock_question) -> None:
        window = _make_window()
        event = Mock()
        config_service = Mock(spec=ConfigService)
        controller = PreferencesController(window, config_service)
        mock_question.return_value = QMessageBox.Yes

        should_close = controller.handle_close_event(event)

        assert should_close is True
        config_service.save_config.assert_called_once_with(window._config)
        window._logger.info.assert_called_once_with("配置已自动保存")

    def test_handle_resize_event_updates_sidebar_and_overlay_positions(self) -> None:
        window = _make_window()
        controller = PreferencesController(window, Mock(spec=ConfigService))

        controller.handle_resize_event()

        window.sidebar.auto_collapse_check.assert_called_once_with(1440)
        window.overlay_state.move.assert_called_once_with(10, 10)
        window.overlay_inv.move.assert_called_once_with(740, 564)
        window.overlay_blink.move.assert_called_once_with(700, 564)