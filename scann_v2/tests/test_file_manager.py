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

    def test_match_pairs_with_fw_prefix(self, tmp_dir, synth_fits_data_16bit):
        """测试前缀兼容机制：FW_ 前缀应正确匹配"""
        try:
            from astropy.io import fits
        except ImportError:
            pytest.skip("astropy not installed")
        from scann.data.file_manager import match_new_old_pairs

        new_dir = tmp_dir / "new_prefix"
        old_dir = tmp_dir / "old_prefix"
        new_dir.mkdir()
        old_dir.mkdir()

        hdr = fits.Header()
        # 新图：正常文件名
        fits.writeto(str(new_dir / "img_001.fits"), synth_fits_data_16bit, header=hdr)
        fits.writeto(str(new_dir / "img_002.fits"), synth_fits_data_16bit, header=hdr)

        # 旧图：带 FW_ 前缀
        fits.writeto(str(old_dir / "FW_img_001.fits"), synth_fits_data_16bit, header=hdr)
        fits.writeto(str(old_dir / "fw_img_002.fits"), synth_fits_data_16bit, header=hdr)

        pairs, only_new, only_old = match_new_old_pairs(str(new_dir), str(old_dir))

        assert len(pairs) == 2, f"应配对2对，实际配对: {len(pairs)}"
        assert len(only_new) == 0, f"不应有仅新图: {only_new}"
        assert len(only_old) == 0, f"不应有仅旧图: {only_old}"

        # 验证配对正确
        pair_names = [p.name for p in pairs]
        assert "img_001" in pair_names
        assert "img_002" in pair_names

    def test_match_pairs_exact_match_first(self, tmp_dir, synth_fits_data_16bit):
        """测试精确匹配优先：应优先精确匹配，再去前缀匹配"""
        try:
            from astropy.io import fits
        except ImportError:
            pytest.skip("astropy not installed")
        from scann.data.file_manager import match_new_old_pairs

        new_dir = tmp_dir / "new_priority"
        old_dir = tmp_dir / "old_priority"
        new_dir.mkdir()
        old_dir.mkdir()

        hdr = fits.Header()
        # 新图：正常文件名
        fits.writeto(str(new_dir / "img_001.fits"), synth_fits_data_16bit, header=hdr)

        # 旧图：同时存在精确匹配和前缀匹配
        # 精确匹配优先
        fits.writeto(str(old_dir / "img_001.fits"), synth_fits_data_16bit, header=hdr)
        fits.writeto(str(old_dir / "FW_img_001.fits"), synth_fits_data_16bit, header=hdr)

        pairs, only_new, only_old = match_new_old_pairs(str(new_dir), str(old_dir))

        assert len(pairs) == 1, f"应配对1对，实际配对: {len(pairs)}"
        assert len(only_new) == 0
        assert len(only_old) == 1, "FW_img_001.fits 应作为仅旧图"

        # 验证配对使用精确匹配
        assert pairs[0].new_path.name == "img_001.fits"
        assert pairs[0].old_path.name == "img_001.fits"

    def test_match_pairs_case_insensitive_prefix(self, tmp_dir, synth_fits_data_16bit):
        """测试前缀大小写不敏感：FW_、fw_、Fw_ 都应被识别"""
        try:
            from astropy.io import fits
        except ImportError:
            pytest.skip("astropy not installed")
        from scann.data.file_manager import match_new_old_pairs

        new_dir = tmp_dir / "new_case"
        old_dir = tmp_dir / "old_case"
        new_dir.mkdir()
        old_dir.mkdir()

        hdr = fits.Header()
        # 新图
        fits.writeto(str(new_dir / "test_001.fits"), synth_fits_data_16bit, header=hdr)
        fits.writeto(str(new_dir / "test_002.fits"), synth_fits_data_16bit, header=hdr)
        fits.writeto(str(new_dir / "test_003.fits"), synth_fits_data_16bit, header=hdr)

        # 旧图：不同大小写的前缀
        fits.writeto(str(old_dir / "FW_test_001.fits"), synth_fits_data_16bit, header=hdr)
        fits.writeto(str(old_dir / "fw_test_002.fits"), synth_fits_data_16bit, header=hdr)
        fits.writeto(str(old_dir / "Fw_test_003.fits"), synth_fits_data_16bit, header=hdr)

        pairs, only_new, only_old = match_new_old_pairs(str(new_dir), str(old_dir))

        assert len(pairs) == 3, f"应配对3对，实际配对: {len(pairs)}"
        assert len(only_new) == 0
        assert len(only_old) == 0

    def test_scan_fits_folder_ignores_aligned_crop_artifacts(self, tmp_dir, synth_fits_data_16bit):
        """扫描列表不应显式包含 __aligned_crop 产物文件"""
        try:
            from astropy.io import fits
        except ImportError:
            pytest.skip("astropy not installed")
        from scann.data.file_manager import scan_fits_folder

        folder = tmp_dir / "scan_ignore_aligned"
        folder.mkdir()
        hdr = fits.Header()

        fits.writeto(str(folder / "IC 196.fts"), synth_fits_data_16bit, header=hdr)
        fits.writeto(str(folder / "IC 196__aligned_crop.fts"), synth_fits_data_16bit, header=hdr)

        files = scan_fits_folder(str(folder))
        names = [f.path.name for f in files]
        assert "IC 196.fts" in names
        assert "IC 196__aligned_crop.fts" not in names

    def test_match_pairs_ignores_aligned_crop_artifacts(self, tmp_dir, synth_fits_data_16bit):
        """配对列表与仅新/仅旧列表均不应显式包含 __aligned_crop 产物文件"""
        try:
            from astropy.io import fits
        except ImportError:
            pytest.skip("astropy not installed")
        from scann.data.file_manager import match_new_old_pairs

        new_dir = tmp_dir / "new_ignore_aligned"
        old_dir = tmp_dir / "old_ignore_aligned"
        new_dir.mkdir()
        old_dir.mkdir()
        hdr = fits.Header()

        fits.writeto(str(new_dir / "IC 196.fts"), synth_fits_data_16bit, header=hdr)
        fits.writeto(str(old_dir / "FW_IC 196.fit"), synth_fits_data_16bit, header=hdr)

        # 历史对齐产物（应被忽略）
        fits.writeto(str(new_dir / "IC 196__aligned_crop.fts"), synth_fits_data_16bit, header=hdr)
        fits.writeto(str(old_dir / "FW_IC 196__aligned_crop.fit"), synth_fits_data_16bit, header=hdr)

        pairs, only_new, only_old = match_new_old_pairs(str(new_dir), str(old_dir))

        assert len(pairs) == 1
        assert pairs[0].name == "IC 196"
        assert only_new == []
        assert only_old == []
