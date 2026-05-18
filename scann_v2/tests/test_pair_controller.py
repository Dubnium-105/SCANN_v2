"""PairController 单元测试。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from scann.core.models import AlignResult, FitsHeader, FitsImage, ImagePair
from scann.data.file_manager import FitsImagePair
from scann.gui.controllers import PairController
from scann.services.dataset_preprocess_service import DatasetPreprocessService
from scann.services.pair_service import PairService


def _make_window():
    window = Mock()
    window._config = SimpleNamespace(
        new_folder="",
        old_folder="",
        recent_folders=[],
        max_recent_count=3,
    )
    window.file_list = Mock()
    window.file_list.count.return_value = 0
    window.histogram_panel = Mock()
    window.menu_recent = Mock()
    window._logger = Mock()
    window._show_message = Mock()
    window._update_markers = Mock()
    window.candidate_presenter = Mock()
    window.set_candidates = Mock()
    window.set_image_data = Mock()
    window._new_folder = ""
    window._old_folder = ""
    window._dataset_root = ""
    window._image_pairs = []
    window._current_pair_idx = -1
    window._current_pair_using_aligned = False
    window._new_image_data = None
    window._new_marked_image_data = None
    window._old_image_data = None
    window._new_fits_header = None
    window._old_fits_header = None
    window._candidates = []
    window._current_candidate_idx = -1
    window._candidates_cache = {}
    return window


def _make_controller(service: PairService | Mock | None = None):
    window = _make_window()
    service = service or Mock(spec=PairService)
    if isinstance(service, Mock):
        service.resolve_marked_image_path.return_value = None
    preprocess = Mock(spec=DatasetPreprocessService)
    controller = PairController(window, service, preprocess)
    return controller, window, service, preprocess


class TestPairController:
    def test_holds_pair_service(self):
        controller, _window, service, _preprocess = _make_controller()

        assert controller.pair_service is service

    def test_pair_navigation_updates_file_list_selection(self):
        controller, window, _service, _preprocess = _make_controller()
        window.file_list.currentRow.return_value = 1
        window.file_list.count.return_value = 4

        controller.next_pair()
        window.file_list.setCurrentRow.assert_called_once_with(2)

        window.file_list.setCurrentRow.reset_mock()
        window.file_list.currentRow.return_value = 2

        controller.prev_pair()
        window.file_list.setCurrentRow.assert_called_once_with(1)

    @patch("scann.gui.controllers.pair_controller.QFileDialog.getExistingDirectory")
    def test_open_dataset_cancelled(self, mock_dialog):
        controller, window, _service, preprocess = _make_controller(Mock(spec=PairService))
        mock_dialog.return_value = ""

        controller.open_dataset()

        assert getattr(window, "_dataset_root", "") == ""
        preprocess.prepare_dataset.assert_not_called()
        window.file_list.clear.assert_not_called()

    @pytest.mark.skip(reason="replaced by unified dataset entry")
    def test_open_new_folder_scans_and_loads_first_image(self, mock_dialog):
        service = Mock(spec=PairService)
        controller, window, _service, preprocess = _make_controller(service)
        mock_dialog.return_value = "/data/new"
        service.scan_new_folder.return_value = [
            SimpleNamespace(path=Path("/data/new/img_001.fits"), stem="img_001"),
            SimpleNamespace(path=Path("/data/new/img_002.fits"), stem="img_002"),
        ]
        service.read_image.return_value = FitsImage(
            data=np.ones((16, 16), dtype=np.float32),
            header=FitsHeader(raw={"OBJECT": "Field"}),
            path=Path("/data/new/img_001.fits"),
        )
        window._image_pairs = [Mock()]
        window._current_pair_idx = 7
        window._candidates_cache = {3: [Mock()]}

        controller.open_new_folder()

        assert window._new_folder == "/data/new"
        assert window._config.new_folder == "/data/new"
        assert window._image_pairs == []
        assert window._current_pair_idx == -1
        assert window._candidates_cache == {}
        service.scan_new_folder.assert_called_once_with("/data/new")
        service.read_image.assert_called_once_with(Path("/data/new/img_001.fits"))
        window.file_list.clear.assert_called_once_with()
        assert window.file_list.addItem.call_count == 2
        window.set_image_data.assert_called_once_with(
            service.read_image.return_value.data,
            None,
            None,
        )

    @patch("scann.gui.controllers.pair_controller.QFileDialog.getExistingDirectory")
    @pytest.mark.skip(reason="replaced by unified dataset entry")
    def test_open_old_folder_matches_pairs_and_loads_first_pair(self, mock_dialog):
        service = Mock(spec=PairService)
        controller, window, _service, preprocess = _make_controller(service)
        mock_dialog.return_value = "/data/old"
        window._new_folder = "/data/new"
        window._candidates_cache = {0: [Mock()]}
        pair = FitsImagePair(
            name="img_001",
            new_path=Path("/data/new/img_001.fits"),
            old_path=Path("/data/old/img_001.fits"),
        )
        service.scan_old_folder.return_value = []
        service.match_pairs.return_value = ([pair], ["img_002"], ["img_003"])
        controller.load_pair = Mock()

        controller.open_old_folder()

        assert window._old_folder == "/data/old"
        assert window._config.old_folder == "/data/old"
        assert window._image_pairs == [pair]
        assert window._candidates_cache == {}
        service.match_pairs.assert_called_once_with("/data/new", "/data/old")
        added_labels = [call.args[0] for call in window.file_list.addItem.call_args_list]
        assert added_labels == ["✅ img_001", "🆕 img_002 (仅新图)", "📁 img_003 (仅旧图)"]
        controller.load_pair.assert_called_once_with(0)

    def test_load_pair_uses_service_and_restores_cached_candidates(self):
        service = Mock(spec=PairService)
        controller, window, _service, _preprocess = _make_controller(service)
        pair = FitsImagePair(
            name="img_001",
            new_path=Path("/data/new/img_001.fits"),
            old_path=Path("/data/old/img_001.fits"),
        )
        window._image_pairs = [pair]
        window._candidates_cache = {0: [Mock(name="cached_candidate")]}
        service.load_pair.return_value = ImagePair(
            name="img_001",
            new_image=FitsImage(
                data=np.ones((24, 24), dtype=np.float32),
                header=FitsHeader(raw={}),
                path=pair.new_path,
            ),
            old_image=FitsImage(
                data=np.ones((24, 24), dtype=np.float32),
                header=FitsHeader(raw={}),
                path=pair.old_path,
            ),
            aligned=True,
        )
        service.calc_nonzero_valid_bounds.return_value = (1, 23, 2, 22)

        controller.load_pair(0)

        assert window._current_pair_idx == 0
        assert window._current_pair_using_aligned is True
        window._update_markers.assert_called_once_with()
        window.candidate_presenter.set_candidates.assert_called_once_with([])
        window.set_image_data.assert_called_once_with(
            window._new_image_data,
            window._old_image_data,
            None,
        )
        window.set_candidates.assert_called_once_with(window._candidates_cache[0])
        assert window._new_image_data.shape == (20, 22)
        assert window._old_image_data.shape == (20, 22)

    @patch("scann.gui.controllers.pair_controller.align")
    def test_load_pair_reuses_alignment_crop_for_raw_marked_image(self, mock_align):
        service = Mock(spec=PairService)
        controller, window, _service, _preprocess = _make_controller(service)
        pair = FitsImagePair(
            name="img_001",
            new_path=Path("/data/new/img_001.fits"),
            old_path=Path("/data/old/img_001.fits"),
        )
        aligned_new_path = Path("/data/new/img_001__aligned_crop.fts")
        aligned_old_path = Path("/data/old/img_001__aligned_crop.fts")
        raw_marked_path = Path("/data/new_marked/img_001.fits")
        window._image_pairs = [pair]

        service.load_pair.return_value = ImagePair(
            name="img_001",
            new_image=FitsImage(
                data=np.ones((6, 6), dtype=np.float32),
                header=FitsHeader(raw={}),
                path=aligned_new_path,
            ),
            old_image=FitsImage(
                data=np.ones((6, 6), dtype=np.float32) * 2,
                header=FitsHeader(raw={}),
                path=aligned_old_path,
            ),
            aligned=True,
        )

        def _resolve_marked(path: str | Path):
            if Path(path) == aligned_new_path:
                return None
            if Path(path) == pair.new_path:
                return raw_marked_path
            return None

        raw_new = np.arange(20 * 20, dtype=np.float32).reshape(20, 20)
        raw_marked = np.rot90(raw_new + 100.0, 2)
        aligned_marked = raw_new + 100.0
        service.resolve_marked_image_path.side_effect = _resolve_marked
        service.read_image.side_effect = [
            FitsImage(data=raw_marked, header=FitsHeader(raw={}), path=raw_marked_path),
            FitsImage(data=raw_new, header=FitsHeader(raw={}), path=pair.new_path),
        ]
        service.resolve_alignment_crop_bounds.return_value = (3, 9, 4, 10)
        service.calc_nonzero_valid_bounds.return_value = None
        mock_align.return_value = AlignResult(
            aligned_old=aligned_marked,
            dx=0.0,
            dy=0.0,
            success=True,
        )

        controller.load_pair(0)

        mock_align.assert_called_once()
        np.testing.assert_array_equal(mock_align.call_args.args[1], raw_marked)
        np.testing.assert_array_equal(
            window._new_marked_image_data,
            aligned_marked[4:10, 3:9],
        )

    def test_recent_folder_actions_update_config_and_menu(self):
        controller, window, _service, _preprocess = _make_controller()
        action = Mock()
        window.menu_recent.addAction.side_effect = [action]

        controller.add_recent_folder("/data/new")

        assert window._config.recent_folders == ["/data/new"]
        window.menu_recent.clear.assert_called_once_with()
        window.menu_recent.addAction.assert_called_once_with("/data/new")
        action.triggered.connect.assert_called_once()
