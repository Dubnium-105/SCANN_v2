from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from scann.core.models import FitsHeader, FitsImage
from scann.services.blink_service import BlinkState


def _make_mock_window():
    import scann.gui.main_window as main_window

    from scann.gui.main_window import MainWindow
    from scann.gui.controllers import PairController
    from scann.gui.presenters import CandidatePresenter, StatusPresenter
    from scann.services.pair_service import PairService

    with patch("scann.gui.main_window.QMainWindow.__init__"):
        window = MainWindow.__new__(MainWindow)

    window.image_viewer = Mock()
    window.blink_service = Mock()
    window.blink_service.is_inverted = False
    window.blink_service.current_state = BlinkState.NEW
    window.blink_service.set_state = Mock()
    window.blink_timer = Mock()
    window.overlay_state = Mock()
    window.overlay_inv = Mock()
    window.overlay_blink = Mock()
    window.btn_show_new = Mock()
    window.btn_show_old = Mock()
    window.btn_blink = Mock()
    window.btn_invert = Mock()
    window.btn_mark_real = Mock()
    window.btn_mark_bogus = Mock()
    window.btn_next_candidate = Mock()
    window.histogram_panel = Mock()
    window.histogram_panel.black_point = 0.0
    window.histogram_panel.white_point = 1.0
    window.file_list = Mock()
    window.suspect_table = Mock()
    window.status_image_type = Mock()
    window.status_pixel_coord = Mock()
    window.status_wcs_coord = Mock()
    window.status_zoom = Mock()
    window.statusBar = Mock(return_value=Mock())
    window.act_show_markers = Mock()
    window.act_show_markers.isChecked.return_value = True
    window._logger = Mock()
    window.status_presenter = StatusPresenter(window.statusBar(), window._logger)
    window.candidate_presenter = CandidatePresenter(window.suspect_table, window.image_viewer)
    window._config = SimpleNamespace(
        blink_speed_ms=500,
        new_folder="",
        old_folder="",
        recent_folders=[],
        max_recent_count=5,
    )
    window._new_folder = ""
    window._old_folder = ""
    window._image_pairs = []
    window._current_pair_idx = -1
    window._new_image_data = None
    window._old_image_data = None
    window._new_fits_header = None
    window._old_fits_header = None
    window._candidates = []
    window._current_candidate_idx = -1
    window._candidates_cache = {}
    window.menu_recent = Mock()
    window.pair_service = PairService(
        scan_folder_fn=main_window.scan_fits_folder,
        match_pairs_fn=main_window.match_new_old_pairs,
        read_fits_fn=main_window.read_fits,
    )
    window.pair_controller = PairController(window, window.pair_service)
    return window


class TestMainWindowPairFlow:
    @patch("scann.gui.main_window.read_fits")
    @patch("scann.gui.main_window.scan_fits_folder")
    @patch("scann.gui.main_window.QFileDialog.getExistingDirectory")
    def test_open_new_folder_resets_state_and_loads_first_image(
        self,
        mock_dialog,
        mock_scan,
        mock_read,
    ):
        from scann.data.file_manager import FitsFileInfo

        window = _make_mock_window()
        window._image_pairs = [Mock()]
        window._current_pair_idx = 3
        window._candidates_cache = {1: [Mock()]}

        mock_dialog.return_value = "/data/new"
        mock_scan.return_value = [
            FitsFileInfo(
                path=Path("/data/new/img_001.fits"),
                stem="img_001",
                size_bytes=1024,
                modified_time=0.0,
            ),
            FitsFileInfo(
                path=Path("/data/new/img_002.fits"),
                stem="img_002",
                size_bytes=1024,
                modified_time=0.0,
            ),
        ]
        mock_read.return_value = FitsImage(
            data=np.ones((16, 16), dtype=np.float32),
            header=FitsHeader(raw={"OBJECT": "Field"}),
            path=Path("/data/new/img_001.fits"),
        )

        window._on_open_new_folder()

        assert window._new_folder == "/data/new"
        assert window._image_pairs == []
        assert window._current_pair_idx == -1
        assert window._candidates_cache == {}
        window.file_list.clear.assert_called_once()
        assert window.file_list.addItem.call_count == 2
        mock_read.assert_called_once_with(Path("/data/new/img_001.fits"))
        window.image_viewer.set_image_data.assert_called_once()

    @patch("scann.gui.main_window.match_new_old_pairs")
    @patch("scann.gui.main_window.scan_fits_folder")
    @patch("scann.gui.main_window.QFileDialog.getExistingDirectory")
    def test_open_old_folder_matches_pairs_and_auto_loads_first_pair(
        self,
        mock_dialog,
        mock_scan,
        mock_match,
    ):
        from scann.data.file_manager import FitsImagePair

        window = _make_mock_window()
        window._new_folder = "/data/new"
        window._candidates_cache = {0: [Mock()]}
        window.pair_controller.load_pair = Mock()

        mock_dialog.return_value = "/data/old"
        mock_scan.return_value = []
        mock_match.return_value = (
            [
                FitsImagePair(
                    name="img_001",
                    new_path=Path("/data/new/img_001.fits"),
                    old_path=Path("/data/old/img_001.fits"),
                )
            ],
            ["img_002"],
            ["img_003"],
        )

        window._on_open_old_folder()

        assert window._old_folder == "/data/old"
        assert len(window._image_pairs) == 1
        assert window._candidates_cache == {}
        mock_match.assert_called_once_with("/data/new", "/data/old")
        window.file_list.clear.assert_called_once()
        added_labels = [call.args[0] for call in window.file_list.addItem.call_args_list]
        assert added_labels == ["✅ img_001", "🆕 img_002 (仅新图)", "📁 img_003 (仅旧图)"]
        window.pair_controller.load_pair.assert_called_once_with(0)

    @patch("scann.gui.main_window.read_fits")
    def test_load_pair_prefers_aligned_artifacts_and_restores_cached_candidates(
        self,
        mock_read,
        tmp_path,
    ):
        from scann.data.file_manager import FitsImagePair

        window = _make_mock_window()
        pair = FitsImagePair(
            name="img_001",
            new_path=tmp_path / "new" / "img_001.fits",
            old_path=tmp_path / "old" / "img_001.fits",
        )
        pair.new_path.parent.mkdir(parents=True, exist_ok=True)
        pair.old_path.parent.mkdir(parents=True, exist_ok=True)
        window._image_pairs = [pair]
        window._candidates_cache = {0: [Mock(name="cached_candidate")]}
        window.set_candidates = Mock()

        new_aligned_path, old_aligned_path, new_marker_path, old_marker_path = window._aligned_artifact_paths(pair)
        new_aligned_path.write_text("aligned", encoding="utf-8")
        old_aligned_path.write_text("aligned", encoding="utf-8")
        new_marker_path.write_text("aligned", encoding="utf-8")
        old_marker_path.write_text("aligned", encoding="utf-8")

        header = FitsHeader(raw={})

        def _fake_read(path):
            path = Path(path)
            if path == new_aligned_path:
                return FitsImage(
                    data=np.ones((24, 24), dtype=np.float32),
                    header=header,
                    path=path,
                )
            if path == old_aligned_path:
                data = np.ones((24, 24), dtype=np.float32)
                data[:, 0] = 0.0
                data[0, :] = 0.0
                return FitsImage(data=data, header=header, path=path)
            raise AssertionError(f"unexpected path: {path}")

        mock_read.side_effect = _fake_read

        window._load_pair(0)

        assert window._current_pair_idx == 0
        assert window._current_pair_using_aligned is True
        read_paths = [Path(call.args[0]) for call in mock_read.call_args_list]
        assert read_paths == [new_aligned_path, old_aligned_path]
        window.set_candidates.assert_called_once_with(window._candidates_cache[0])
        assert window.histogram_panel.set_image_data.call_count >= 1
        np.testing.assert_array_equal(
            window.histogram_panel.set_image_data.call_args_list[-1].args[0],
            window._new_image_data,
        )

    def test_pair_navigation_updates_file_list_selection(self):
        window = _make_mock_window()
        window.file_list.currentRow.return_value = 1
        window.file_list.count.return_value = 4

        window._on_next_pair()
        window.file_list.setCurrentRow.assert_called_once_with(2)

        window.file_list.setCurrentRow.reset_mock()
        window.file_list.currentRow.return_value = 2

        window._on_prev_pair()
        window.file_list.setCurrentRow.assert_called_once_with(1)