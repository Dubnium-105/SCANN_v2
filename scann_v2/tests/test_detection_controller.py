"""DetectionController 行为测试。"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from scann.core.models import Candidate, FitsHeader, FitsImage, TargetVerdict
from scann.gui.controllers import DetectionController


def _make_window() -> Mock:
    window = Mock()
    window._candidates = []
    window._current_candidate_idx = -1
    window._current_pair_idx = -1
    window._candidates_cache = {}
    window._new_image_data = None
    window._old_image_data = None
    window._new_fits_header = None
    window._current_pair_using_aligned = False
    window._image_pairs = []
    window._new_folder = ""
    window._batch_dialog = None
    window._inference_engine = None
    window._config = SimpleNamespace(
        thresh=1.5,
        min_area=3,
        max_area=99,
        sharpness=0.2,
        max_sharpness=2.0,
        contrast=0.1,
        edge_margin=4,
        dynamic_thresh=True,
        kill_flat=False,
        kill_dipole=True,
        aspect_ratio_max=3.5,
        extent_max=0.8,
        topk=25,
        slice_size=128,
        mpcorb_path="",
        observatory=None,
    )
    window.suspect_table = Mock()
    window.image_viewer = Mock()
    window.status_pixel_coord = Mock()
    window.progress_bar = Mock()
    window.btn_align = Mock()
    window.act_align = Mock()
    window._logger = Mock()
    window._show_message = Mock()
    window._update_markers = Mock()
    window.set_candidates = Mock(side_effect=lambda candidates: setattr(window, "_candidates", candidates))
    window._aligned_artifact_paths = Mock()
    window._pair_has_aligned_artifacts = Mock(return_value=False)
    window._calc_overlap_crop_bounds = Mock(return_value=(0, 4, 0, 4))
    window._resolve_pair_image_paths = Mock(return_value=(Path("/tmp/new.fits"), Path("/tmp/old.fits"), False))
    window._load_pair = Mock()
    return window


def test_mark_real_updates_candidate_and_view() -> None:
    window = _make_window()
    controller = DetectionController(window)
    candidate = Candidate(x=10, y=20)
    window._candidates = [candidate]
    window._current_candidate_idx = 0

    controller.mark_real()

    assert candidate.verdict == TargetVerdict.REAL
    window.suspect_table.update_candidate.assert_called_once_with(0)
    window._update_markers.assert_called_once_with()


def test_candidate_navigation_updates_focus() -> None:
    window = _make_window()
    controller = DetectionController(window)
    window._candidates = [Candidate(x=1, y=2), Candidate(x=3, y=4)]

    controller.next_candidate()

    assert window._current_candidate_idx == 0
    window.image_viewer.center_on_point.assert_called_once_with(1, 2)
    window.status_pixel_coord.set_pixel_coordinates.assert_called_once_with(1, 2)


def test_candidate_double_click_zooms() -> None:
    window = _make_window()
    controller = DetectionController(window)
    window._candidates = [Candidate(x=5, y=6)]

    controller.candidate_double_clicked(0)

    window.image_viewer.center_on_point.assert_called_once_with(5, 6, zoom_to=200)


def test_build_detection_params_reads_window_config() -> None:
    window = _make_window()
    controller = DetectionController(window)

    result = controller.build_detection_params()

    assert result.thresh == 1.5
    assert result.min_area == 3
    assert result.topk == 25


@patch("scann.gui.controllers.detection_controller.DetectionPipeline")
def test_batch_detect_updates_candidates_and_cache(mock_pipeline_cls) -> None:
    from scann.services.detection_service import PipelineResult

    window = _make_window()
    controller = DetectionController(window)
    window._new_image_data = np.zeros((8, 8), dtype=np.float32)
    window._old_image_data = np.ones((8, 8), dtype=np.float32)
    window._current_pair_idx = 2

    candidates = [Candidate(x=10, y=20, ai_score=0.9)]
    mock_pipeline = Mock()
    mock_pipeline.process_pair.return_value = PipelineResult(
        pair_name="pair",
        candidates=candidates,
    )
    mock_pipeline_cls.return_value = mock_pipeline

    controller.batch_detect()

    window.set_candidates.assert_called_once_with(candidates)
    assert window._candidates_cache[2] == candidates


@patch("scann.gui.controllers.detection_controller.ExclusionService")
@patch("scann.gui.controllers.detection_controller.DetectionPipeline")
def test_batch_detect_passes_header_and_image_path_to_pipeline(
    mock_pipeline_cls,
    mock_exclusion_service_cls,
) -> None:
    from scann.services.detection_service import PipelineResult

    window = _make_window()
    controller = DetectionController(window)
    window._new_image_data = np.zeros((8, 8), dtype=np.float32)
    window._old_image_data = np.ones((8, 8), dtype=np.float32)
    window._new_fits_header = FitsHeader(raw={"RA": 180.0, "DEC": 0.0})
    window._current_pair_idx = 0
    window._image_pairs = [SimpleNamespace(new_path=Path("/tmp/new.fits"), old_path=Path("/tmp/old.fits"))]
    window._config.mpcorb_path = "/tmp/MPCORB.DAT"

    mock_exclusion_service = Mock()
    mock_exclusion_service_cls.return_value = mock_exclusion_service

    mock_pipeline = Mock()
    mock_pipeline.process_pair.return_value = PipelineResult(pair_name="pair", candidates=[])
    mock_pipeline_cls.return_value = mock_pipeline

    controller.batch_detect()

    assert mock_pipeline_cls.call_args.kwargs["exclusion_service"] is mock_exclusion_service
    assert mock_pipeline.process_pair.call_args.kwargs["header"] is window._new_fits_header
    assert Path(mock_pipeline.process_pair.call_args.kwargs["image_path"]) == Path("/tmp/new.fits")
    mock_exclusion_service.load_mpcorb.assert_called_once_with()


@patch("scann.gui.controllers.detection_controller.scan_fits_folder")
@patch("scann.gui.controllers.detection_controller.read_fits")
@patch("scann.gui.controllers.detection_controller.write_fits")
def test_run_batch_process_applies_denoise(
    mock_write,
    mock_read,
    mock_scan,
) -> None:
    window = _make_window()
    controller = DetectionController(window)
    mock_scan.return_value = [Path("/tmp/test.fits")]
    mock_read.return_value = FitsImage(
        data=np.ones((4, 4), dtype=np.float32),
        header=FitsHeader(raw={}),
        path=Path("/tmp/test.fits"),
    )

    with patch("scann.gui.controllers.detection_controller.denoise") as mock_denoise:
        mock_denoise.return_value = np.ones((4, 4), dtype=np.float32)
        controller.run_batch_process(
            {
                "input_dir": "/tmp",
                "output_dir": "/tmp/out",
                "denoise": True,
                "denoise_method": "中值滤波",
                "kernel_size": 3,
                "flat_field": False,
            }
        )

    mock_denoise.assert_called_once()
    mock_write.assert_called_once()


@patch("scann.gui.controllers.detection_controller.align")
@patch("scann.gui.controllers.detection_controller.read_fits")
@patch("scann.gui.controllers.detection_controller.write_fits")
def test_batch_align_processes_pairs(
    mock_write,
    mock_read,
    mock_align,
    tmp_path,
) -> None:
    from scann.data.file_manager import FitsImagePair
    from scann.core.models import AlignResult

    window = _make_window()
    controller = DetectionController(window)
    pair = FitsImagePair(
        name="img_001",
        new_path=tmp_path / "new" / "img_001.fits",
        old_path=tmp_path / "old" / "img_001.fits",
    )
    pair.new_path.parent.mkdir(parents=True, exist_ok=True)
    pair.old_path.parent.mkdir(parents=True, exist_ok=True)
    window._image_pairs = [pair]
    window._aligned_artifact_paths.return_value = (
        tmp_path / "new" / "img_001__aligned_crop.fts",
        tmp_path / "old" / "img_001__aligned_crop.fts",
        tmp_path / "new" / "img_001__aligned.marker",
        tmp_path / "old" / "img_001__aligned.marker",
    )

    new_data = np.arange(16, dtype=np.float32).reshape(4, 4)
    old_data = np.flip(new_data, axis=1).copy()
    mock_read.side_effect = [
        FitsImage(data=new_data, header=FitsHeader(raw={}), path=pair.new_path),
        FitsImage(data=old_data, header=FitsHeader(raw={}), path=pair.old_path),
    ]
    mock_align.return_value = AlignResult(
        aligned_old=old_data,
        dx=1.0,
        dy=2.0,
        success=True,
    )

    controller.batch_align()

    mock_align.assert_called_once()
    assert mock_write.call_count == 2