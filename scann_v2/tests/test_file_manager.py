"""文件管理模块单元测试"""

import pytest


class TestFileManager:
    """测试 FITS 文件扫描与配对"""

    def test_scan_fits_folder(self, sample_fits_folder):
        from scann.data.file_manager import scan_fits_folder

        files = scan_fits_folder(str(sample_fits_folder))
        assert len(files) >= 5  # 至少 5 个 .fits 文件
        # 也应该找到 .fit 文件
        assert len(files) >= 6

    def test_scan_empty_folder(self, tmp_dir):
        from scann.data.file_manager import scan_fits_folder

        empty = tmp_dir / "empty"
        empty.mkdir()
        files = scan_fits_folder(str(empty))
        assert files == []

    def test_match_pairs(self, fits_file_pair):
        from scann.data.file_manager import match_new_old_pairs

        new_path, old_path = fits_file_pair
        pairs, only_new, only_old = match_new_old_pairs(
            str(new_path.parent), str(old_path.parent)
        )
        assert len(pairs) >= 1
        assert pairs[0].new_path.name == pairs[0].old_path.name

    def test_match_pairs_no_overlap(self, tmp_dir, synth_fits_data_16bit):
        """不匹配的文件名不应配对"""
        try:
            from astropy.io import fits
        except ImportError:
            pytest.skip("astropy not installed")
        from scann.data.file_manager import match_new_old_pairs

        new_dir = tmp_dir / "new2"
        old_dir = tmp_dir / "old2"
        new_dir.mkdir()
        old_dir.mkdir()

        hdr = fits.Header()
        fits.writeto(str(new_dir / "aaa.fits"), synth_fits_data_16bit, header=hdr)
        fits.writeto(str(old_dir / "bbb.fits"), synth_fits_data_16bit, header=hdr)

        pairs, _, _ = match_new_old_pairs(str(new_dir), str(old_dir))
        assert len(pairs) == 0

    def test_fits_file_info_fields(self, sample_fits_folder):
        from scann.data.file_manager import scan_fits_folder

        files = scan_fits_folder(str(sample_fits_folder))
        for f in files:
            assert hasattr(f, "path")
            assert hasattr(f, "filename")  # property derived from path
            assert hasattr(f, "stem")
            assert f.path.exists()
            assert f.filename == f.path.name
