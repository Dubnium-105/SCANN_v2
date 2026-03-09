"""PairService 单元测试。"""

from pathlib import Path

import pytest

from scann.core.models import ImagePair
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

        new_path, old_path = service.resolve_pair_image_paths(pair)

        assert new_path == pair.new_path
        assert old_path == pair.old_path

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

    def test_load_pair_propagates_read_errors(self):
        service = PairService()
        pair = FitsImagePair(
            name="missing",
            new_path=Path("missing_new.fits"),
            old_path=Path("missing_old.fits"),
        )

        with pytest.raises(FileNotFoundError):
            service.load_pair(pair)