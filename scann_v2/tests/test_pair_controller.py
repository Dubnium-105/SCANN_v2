"""PairController 单元测试。"""

from unittest.mock import Mock

from scann.gui.controllers import PairController
from scann.services.pair_service import PairService


def _make_controller():
    window = Mock()
    service = PairService()
    controller = PairController(window, service)
    return controller, window, service


class TestPairController:
    def test_holds_pair_service_for_future_migration(self):
        controller, _window, service = _make_controller()

        assert controller.pair_service is service

    def test_open_folder_actions_delegate_to_existing_window_impl(self):
        controller, window, _service = _make_controller()

        controller.open_new_folder()
        controller.open_old_folder()

        window._open_new_folder_impl.assert_called_once_with()
        window._open_old_folder_impl.assert_called_once_with()

    def test_pair_navigation_actions_delegate_to_existing_window_impl(self):
        controller, window, _service = _make_controller()

        controller.prev_pair()
        controller.next_pair()
        controller.select_pair(3)

        window._prev_pair_impl.assert_called_once_with()
        window._next_pair_impl.assert_called_once_with()
        window._select_pair_impl.assert_called_once_with(3)

    def test_recent_folder_actions_delegate_to_existing_window_impl(self):
        controller, window, _service = _make_controller()

        controller.add_recent_folder("/data/new")
        controller.update_recent_menu()
        controller.open_recent_folder("/data/new")

        window._add_recent_folder_impl.assert_called_once_with("/data/new")
        window._update_recent_menu_impl.assert_called_once_with()
        window._open_recent_folder_impl.assert_called_once_with("/data/new")