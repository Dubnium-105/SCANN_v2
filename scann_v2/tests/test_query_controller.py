"""QueryController 行为测试。"""

import shutil
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pathlib import Path

import pytest

from scann.core.models import Candidate, FitsHeader, SkyPosition
from scann.gui.controllers import QueryController
from scann.services.query_service import QueryResponse, QueryResult
from scann.services.exclusion_service import ExclusionService
from scann.services.siril_astrometry import ResolvedSkyCoordinate


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
    window.detection_controller = Mock()
    window.detection_controller.get_exclusion_service.return_value = Mock()
    window.detection_controller.resolve_current_new_image_path.return_value = "/tmp/example.fit"
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
    service.execute_query.return_value = QueryResponse(
        results=[
            QueryResult(
                source="Satellite",
                name="ISS",
                object_type="satellite",
                distance_arcsec=3.2,
            )
        ]
    )
    mock_service_cls.return_value = service
    popup = Mock()
    mock_popup_cls.return_value = popup

    controller = QueryController(window)
    controller._start_async_query = Mock()
    controller.do_query("satellite", 10, 20)

    controller._start_async_query.assert_called_once_with(
        service,
        "satellite",
        180.0,
        -12.5,
        "12 00 00.00 -12 30 00.00",
    )
    popup.show.assert_not_called()


def test_handle_async_query_finished_displays_popup() -> None:
    window = _make_window()
    controller = QueryController(window)
    controller._show_query_results = Mock()

    response = QueryResponse(
        results=[
            QueryResult(
                source="Satellite",
                name="ISS",
                object_type="satellite",
                distance_arcsec=3.2,
            )
        ]
    )

    controller._handle_async_query_finished(
        "satellite",
        "12 00 00.00 -12 30 00.00",
        response,
    )

    controller._show_query_results.assert_called_once_with(
        "satellite",
        "12 00 00.00 -12 30 00.00",
        response,
    )


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


@patch("scann.gui.controllers.query_controller.QApplication")
@patch("scann.gui.controllers.query_controller.pixel_to_wcs")
def test_copy_wcs_coordinates_prefers_resolved_coordinate(mock_pixel_to_wcs, mock_qapp) -> None:
    window = _make_window()
    window._new_fits_header = FitsHeader(raw={})
    resolved = ResolvedSkyCoordinate.from_hms_dms(
        ra_hms="10h49m28.34s",
        dec_dms="+34°43'01.27\"",
    )
    exclusion_service = Mock()
    exclusion_service.get_candidate_sky_coordinate.return_value = resolved
    window.detection_controller.get_exclusion_service.return_value = exclusion_service

    clipboard = Mock()
    mock_qapp.clipboard.return_value = clipboard
    controller = QueryController(window)

    controller.copy_wcs_coordinates(64, 64)

    clipboard.setText.assert_called_once_with("10 49 28.34 +34 43 01.27")
    mock_pixel_to_wcs.assert_not_called()


def test_populate_candidate_coordinates_sets_wcs_text() -> None:
    window = _make_window()
    window._new_fits_header = FitsHeader(raw={})
    resolved = ResolvedSkyCoordinate.from_hms_dms(
        ra_hms="10h49m28.34s",
        dec_dms="+34°43'01.27\"",
    )
    exclusion_service = Mock()
    exclusion_service.get_candidate_sky_coordinate.return_value = resolved
    window.detection_controller.get_exclusion_service.return_value = exclusion_service
    candidates = [Candidate(x=10, y=20)]
    controller = QueryController(window)

    controller.populate_candidate_coordinates(candidates)

    assert candidates[0].wcs_text == "10 49 28.34 +34 43 01.27"


def test_do_query_uses_consistent_coordinate_text_in_message_and_popup() -> None:
    window = _make_window()
    window._new_fits_header = FitsHeader(raw={})
    resolved = ResolvedSkyCoordinate.from_hms_dms(
        ra_hms="10h49m28.34s",
        dec_dms="+34°43'01.27\"",
    )
    exclusion_service = Mock()
    exclusion_service.get_candidate_sky_coordinate.return_value = resolved
    window.detection_controller.get_exclusion_service.return_value = exclusion_service
    controller = QueryController(window)
    controller._execute_query = Mock(return_value=[])
    controller._show_query_results = Mock()

    controller.do_query("mpc", 64, 64)

    window._show_message.assert_called_once_with(
        "正在查询 mpc (10 49 28.34 +34 43 01.27)...",
        5000,
    )
    controller._show_query_results.assert_called_once_with(
        "mpc",
        "10 49 28.34 +34 43 01.27",
        [],
    )


@patch("scann.gui.controllers.query_controller.QueryResultPopup")
def test_show_query_results_displays_error_popup_and_warning(mock_popup_cls) -> None:
    window = _make_window()
    controller = QueryController(window)
    popup = Mock()
    mock_popup_cls.return_value = popup

    controller._show_query_results(
        "vsx",
        "10 49 28.34 +34 43 01.27",
        QueryResponse(error="VSX 请求失败: timeout"),
    )

    window._show_message.assert_called_once_with(
        "查询失败: VSX 请求失败: timeout",
        5000,
        level="WARNING",
    )
    popup.set_error.assert_called_once_with("VSX 请求失败: timeout")


def test_populate_candidate_coordinates_real_sample_preserves_fractional_seconds() -> None:
    from astropy.io import fits

    repo_root = Path(__file__).resolve().parents[2]
    sample_path = repo_root / "dataset" / "new" / "NGC 3381__aligned_crop.fts"
    if not sample_path.exists():
        pytest.skip("真实样本不存在")

    if shutil.which("siril-cli.exe") is None and shutil.which("siril.exe") is None:
        pytest.skip("Siril CLI 不可用")

    window = _make_window()
    window._new_fits_header = FitsHeader(raw=dict(fits.getheader(sample_path)))
    window.detection_controller.get_exclusion_service.return_value = ExclusionService()
    window.detection_controller.resolve_current_new_image_path.return_value = str(sample_path)
    candidates = [Candidate(x=1315, y=616)]
    controller = QueryController(window)

    controller.populate_candidate_coordinates(candidates)

    assert candidates[0].wcs_text == "10 49 28.07 +34 43 00.84"