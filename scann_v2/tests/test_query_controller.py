"""QueryController 行为测试。"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

from scann.core.models import Candidate, FitsHeader, SkyPosition
from scann.gui.controllers import QueryController
from scann.services.query_service import QueryResult


def _make_window() -> Mock:
    window = Mock()
    window.image_viewer = Mock()
    window.image_viewer.mapFromScene.side_effect = lambda x, y: (x, y)
    window.image_viewer.mapToGlobal.side_effect = lambda point: point
    window.status_pixel_coord = Mock()
    window.candidate_presenter = Mock()
    window.detection_controller = Mock()
    window._show_message = Mock()
    window._update_markers = Mock()
    window._candidates = []
    window._current_candidate_idx = -1
    window._new_fits_header = None
    return window


def test_image_clicked_updates_status_coordinates() -> None:
    window = _make_window()
    controller = QueryController(window)

    controller.image_clicked(12, 34)

    window.status_pixel_coord.set_pixel_coordinates.assert_called_once_with(12, 34)


def test_context_add_candidate_updates_candidate_list_and_view() -> None:
    window = _make_window()
    controller = QueryController(window)

    controller.context_add_candidate(20, 40)

    assert len(window._candidates) == 1
    assert window._candidates[0].is_manual is True
    assert window._current_candidate_idx == 0
    window.candidate_presenter.set_candidates.assert_called_once_with(window._candidates)
    window._update_markers.assert_called_once_with()


@patch("scann.gui.controllers.query_controller.QueryResultPopup")
@patch("scann.gui.controllers.query_controller.QueryService")
@patch("scann.gui.controllers.query_controller.pixel_to_wcs")
def test_do_query_satellite_uses_query_service(
    mock_pixel_to_wcs,
    mock_service_cls,
    mock_popup_cls,
) -> None:
    window = _make_window()
    window._new_fits_header = SimpleNamespace(
        raw={},
        observation_datetime="2026-03-10T20:15:00",
    )

    mock_pixel_to_wcs.return_value = SkyPosition(ra=180.0, dec=-12.5)
    service = Mock()
    service.check_satellite.return_value = [
        QueryResult(
            source="Satellite",
            name="ISS",
            object_type="satellite",
            distance_arcsec=3.2,
        )
    ]
    mock_service_cls.return_value = service
    popup = Mock()
    mock_popup_cls.return_value = popup

    controller = QueryController(window)
    controller.do_query("satellite", 10, 20)

    service.check_satellite.assert_called_once_with(
        180.0,
        -12.5,
        obs_datetime="2026-03-10T20:15:00",
    )
    popup.set_success.assert_called_once_with(count=1)
    popup.show.assert_called_once_with()


def test_menu_query_uses_current_candidate_coordinates() -> None:
    window = _make_window()
    window._candidates = [Candidate(x=15, y=25)]
    window._current_candidate_idx = 0
    controller = QueryController(window)
    controller.do_query = Mock()

    controller.menu_query("simbad")

    controller.do_query.assert_called_once_with("simbad", 15, 25)


def test_context_mpc_report_focuses_nearest_candidate_before_opening_dialog() -> None:
    window = _make_window()
    window._candidates = [Candidate(x=10, y=10), Candidate(x=42, y=44)]
    controller = QueryController(window)
    controller.mpc_report = Mock()

    controller.context_mpc_report(40, 40)

    assert window._current_candidate_idx == 1
    window.detection_controller.focus_candidate.assert_called_once_with(1)
    controller.mpc_report.assert_called_once_with()


@patch("scann.gui.controllers.query_controller.QApplication")
@patch("scann.gui.controllers.query_controller.pixel_to_wcs")
def test_copy_wcs_coordinates_copies_formatted_text(mock_pixel_to_wcs, mock_qapp) -> None:
    window = _make_window()
    window._new_fits_header = FitsHeader(raw={})
    mock_pixel_to_wcs.return_value = SkyPosition(ra=180.5, dec=45.3)
    clipboard = Mock()
    mock_qapp.clipboard.return_value = clipboard
    controller = QueryController(window)

    controller.copy_wcs_coordinates(64, 64)

    clipboard.setText.assert_called_once()
    copied_text = clipboard.setText.call_args.args[0]
    assert copied_text.startswith("12 ")
    assert "+45 " in copied_text