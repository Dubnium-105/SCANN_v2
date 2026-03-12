from types import SimpleNamespace
from unittest.mock import Mock, patch

from scann.core.models import Candidate, FitsHeader
from scann.gui.controllers import DetectionController, QueryController
from scann.gui.presenters import StatusPresenter
from scann.services.query_service import QueryResponse, QueryResult


class _Signal:
    def __init__(self):
        self.callback = None

    def connect(self, callback):
        self.callback = callback


class _Action:
    def __init__(self, label):
        self.label = label
        self.triggered = _Signal()


class _Menu:
    instances = []

    def __init__(self, *_args, **_kwargs):
        self.actions = []
        self.exec_calls = []
        _Menu.instances.append(self)

    def addAction(self, label):
        action = _Action(label)
        self.actions.append(action)
        return action

    def addSeparator(self):
        return None

    def exec_(self, position):
        self.exec_calls.append(position)


def _make_mock_window():
    import scann.gui.main_window as main_window

    from scann.gui.main_window import MainWindow
    from scann.gui.controllers import PairController
    from scann.services.pair_service import PairService

    with patch("scann.gui.main_window.QMainWindow.__init__"):
        window = MainWindow.__new__(MainWindow)

    window.image_viewer = Mock()
    window.image_viewer.mapFromScene.side_effect = lambda x, y: (x, y)
    window.image_viewer.mapToGlobal.side_effect = lambda point: point
    window.statusBar = Mock(return_value=Mock())
    window._logger = Mock()
    window.status_presenter = StatusPresenter(window.statusBar(), window._logger)
    window._new_fits_header = None
    window._candidates = []
    window._current_candidate_idx = -1
    window._config = SimpleNamespace()
    window.pair_service = PairService(
        scan_folder_fn=main_window.scan_fits_folder,
        match_pairs_fn=main_window.match_new_old_pairs,
        read_fits_fn=main_window.read_fits,
    )
    window.pair_controller = PairController(window, window.pair_service)
    window.detection_controller = DetectionController(window)
    window.query_controller = QueryController(window)
    return window


class TestMainWindowQueryFlow:
    @patch("scann.gui.controllers.query_controller.QMenu", _Menu)
    def test_right_click_menu_wires_query_and_context_actions(self):
        window = _make_mock_window()
        window._do_query = Mock()
        window._on_context_mpc_report = Mock()
        window._on_context_add_candidate = Mock()
        window._on_copy_wcs_coordinates = Mock()
        window.query_controller.do_query = window._do_query
        window.query_controller.context_mpc_report = window._on_context_mpc_report
        window.query_controller.context_add_candidate = window._on_context_add_candidate
        window.query_controller.copy_wcs_coordinates = window._on_copy_wcs_coordinates

        _Menu.instances.clear()
        window._on_image_right_click(42, 84)

        menu = _Menu.instances[-1]
        assert [action.label for action in menu.actions] == [
            "🔍 查询 VSX",
            "🔍 查询 MPC",
            "🔍 查询 SIMBAD",
            "🔍 查询 TNS",
            "🛰️ 查询人造卫星",
            "📝 生成 MPC 80列报告",
            "➕ 手动添加候选体",
            "📋 复制像素坐标",
            "📋 复制天球坐标",
        ]

        action_map = {action.label: action for action in menu.actions}
        action_map["🔍 查询 VSX"].triggered.callback(False)
        action_map["🔍 查询 MPC"].triggered.callback(False)
        action_map["📝 生成 MPC 80列报告"].triggered.callback(False)
        action_map["➕ 手动添加候选体"].triggered.callback(False)
        action_map["📋 复制天球坐标"].triggered.callback(False)

        window._do_query.assert_any_call("vsx", 42, 84)
        window._do_query.assert_any_call("mpc", 42, 84)
        window._on_context_mpc_report.assert_called_once_with(42, 84)
        window._on_context_add_candidate.assert_called_once_with(42, 84)
        window._on_copy_wcs_coordinates.assert_called_once_with(42, 84)
        assert menu.exec_calls == [(42.0, 84.0)]

    def test_menu_query_uses_selected_candidate_coordinates(self):
        window = _make_mock_window()
        window._candidates = [Candidate(x=123, y=456)]
        window._current_candidate_idx = 0
        window._do_query = Mock()
        window.query_controller.do_query = window._do_query

        window._on_menu_query("simbad")

        window._do_query.assert_called_once_with("simbad", 123, 456)

    def test_menu_query_without_selected_candidate_shows_hint(self):
        window = _make_mock_window()

        window._on_menu_query("vsx")

        status_bar = window.statusBar.return_value
        status_bar.showMessage.assert_called_once()
        assert "请先选中一个候选体" in status_bar.showMessage.call_args.args[0]

    @patch("scann.gui.controllers.query_controller.QueryResultPopup")
    @patch("scann.gui.controllers.query_controller.QueryService")
    @patch("scann.gui.controllers.query_controller.pixel_to_wcs")
    def test_do_query_with_wcs_invokes_service_and_formats_popup(
        self,
        mock_pixel_to_wcs,
        mock_service_cls,
        mock_popup_cls,
    ):
        window = _make_mock_window()
        window._new_fits_header = FitsHeader(raw={"CTYPE1": "RA---TAN"})

        sky = Mock()
        sky.ra = 180.5
        sky.dec = -20.25
        mock_pixel_to_wcs.return_value = sky

        service = Mock()
        service.execute_query.return_value = QueryResponse(
            results=[
                QueryResult(
                    source="TNS",
                    name="AT2025abc",
                    object_type="Supernova",
                    distance_arcsec=1.5,
                )
            ]
        )
        mock_service_cls.return_value = service

        popup = Mock()
        mock_popup_cls.return_value = popup

        window._do_query("tns", 50, 60)

        service.execute_query.assert_called_once_with(
            "tns",
            180.5,
            -20.25,
            obs_datetime=None,
        )
        popup.set_success.assert_called_once_with(count=1)
        popup.set_content.assert_called_once()
        content_args = popup.set_content.call_args.args
        content_kwargs = popup.set_content.call_args.kwargs
        assert "AT2025abc" in content_args[0]
        assert content_kwargs["coords"] == "12 02 00.00 -20 15 00.00"
        popup.show.assert_called_once()