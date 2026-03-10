"""PairService 单元测试。"""

from pathlib import Path

import pytest

import numpy as np

from scann.core.models import FitsHeader, FitsImage, ImagePair
from scann.data.file_manager import FitsImagePair
from scann.services.pair_service import PairService


class TestPairService:
    """验证 PairService 只封装现有扫描、配对和读取能力。"""

    def test_scan_new_folder(self, sample_fits_folder):
        service = PairService()

        files = service.scan_new_folder(sample_fits_folder)

        assert len(files) >= 6
        assert all(file_info.path.parent == sample_fits_folder for file_info in files)

    def test_scan_old_folder(self, sample_fits_folder):
        service = PairService()

        files = service.scan_old_folder(sample_fits_folder)

        assert len(files) >= 6
        assert all(file_info.path.parent == sample_fits_folder for file_info in files)

    def test_match_pairs(self, fits_file_pair):
        service = PairService()
        new_path, old_path = fits_file_pair

        pairs, only_new, only_old = service.match_pairs(new_path.parent, old_path.parent)

        assert len(pairs) == 1
        assert pairs[0].name == "field_001"
        assert pairs[0].new_path == new_path
        assert pairs[0].old_path == old_path
        assert only_new == []
        assert only_old == []

    def test_resolve_pair_image_paths_returns_original_paths(self):
        service = PairService()
        pair = FitsImagePair(
            name="field_001",
            new_path=Path("/data/new/field_001.fits"),
            old_path=Path("/data/old/field_001.fits"),
        )

        new_path, old_path, using_aligned = service.resolve_pair_image_paths(pair)

        assert new_path == pair.new_path
        assert old_path == pair.old_path
        assert using_aligned is False

    def test_pair_has_aligned_artifacts_when_all_outputs_exist(self, tmp_path):
        service = PairService()
        pair = FitsImagePair(
            name="field_001",
            new_path=tmp_path / "new" / "field_001.fits",
            old_path=tmp_path / "old" / "field_001.fits",
        )
        pair.new_path.parent.mkdir(parents=True, exist_ok=True)
        pair.old_path.parent.mkdir(parents=True, exist_ok=True)

        new_aligned_path, old_aligned_path, new_marker_path, old_marker_path = (
            service.aligned_artifact_paths(pair)
        )
        for path in (new_aligned_path, old_aligned_path, new_marker_path, old_marker_path):
            path.write_text("ok", encoding="utf-8")

        assert service.pair_has_aligned_artifacts(pair) is True

    def test_calc_nonzero_valid_bounds_trims_black_border(self):
        service = PairService()
        image = np.ones((20, 20), dtype=np.float32)
        image[:, :3] = 0.0
        image[:2, :] = 0.0

        bounds = service.calc_nonzero_valid_bounds(image)

        assert bounds is not None
        x0, x1, y0, y1 = bounds
        assert x0 >= 3
        assert y0 >= 2
        assert x1 <= 20
        assert y1 <= 20

    def test_load_pair_reads_both_images(self, fits_file_pair):
        service = PairService()
        new_path, old_path = fits_file_pair
        pair = FitsImagePair(name="field_001", new_path=new_path, old_path=old_path)

        image_pair = service.load_pair(pair)

        assert isinstance(image_pair, ImagePair)
        assert image_pair.name == "field_001"
        assert image_pair.new_image.path == new_path
        assert image_pair.old_image.path == old_path
        assert image_pair.aligned is False

    def test_load_pair_marks_aligned_when_aligned_outputs_exist(self, fits_file_pair):
        new_path, old_path = fits_file_pair
        pair = FitsImagePair(name="field_001", new_path=new_path, old_path=old_path)
        service = PairService(
            read_fits_fn=lambda path: FitsImage(
                data=np.zeros((8, 8), dtype=np.float32),
                header=FitsHeader(raw={}),
                path=Path(path),
            )
        )
        new_aligned_path, old_aligned_path, new_marker_path, old_marker_path = (
            service.aligned_artifact_paths(pair)
        )
        new_aligned_path.write_text("aligned", encoding="utf-8")
        old_aligned_path.write_text("aligned", encoding="utf-8")
        new_marker_path.write_text("aligned", encoding="utf-8")
        old_marker_path.write_text("aligned", encoding="utf-8")

        image_pair = service.load_pair(pair)

        assert image_pair.aligned is True
        assert image_pair.new_image.path == new_aligned_path
        assert image_pair.old_image.path == old_aligned_path

    def test_load_pair_propagates_read_errors(self):
        service = PairService()
        pair = FitsImagePair(
            name="missing",
            new_path=Path("missing_new.fits"),
            old_path=Path("missing_old.fits"),
        )

        with pytest.raises(FileNotFoundError):
            service.load_pair(pair)