"""FITS I/O 模块单元测试"""

import numpy as np
import pytest

from scann.core.models import BitDepth


class TestReadFits:
    """测试 FITS 文件读取"""

    def test_read_fits_returns_fitsimage(self, fits_file_pair):
        from scann.core.fits_io import read_fits

        new_path, _ = fits_file_pair
        result = read_fits(str(new_path))
        assert result is not None
        assert result.data is not None
        assert result.data.ndim == 2
        assert result.header is not None

    def test_read_fits_preserves_dtype(self, fits_file_pair):
        from scann.core.fits_io import read_fits

        new_path, _ = fits_file_pair
        result = read_fits(str(new_path))
        # 16bit FITS -> uint16 or int16
        assert result.data.dtype in (np.uint16, np.int16, np.float32)

    def test_read_fits_header_fields(self, fits_file_pair):
        from scann.core.fits_io import read_fits

        new_path, _ = fits_file_pair
        result = read_fits(str(new_path))
        assert result.header is not None

    def test_read_fits_nonexistent_raises(self, tmp_dir):
        from scann.core.fits_io import read_fits

        with pytest.raises((FileNotFoundError, OSError)):
            read_fits(str(tmp_dir / "nonexistent.fits"))

    def test_read_header_only(self, fits_file_pair):
        from scann.core.fits_io import read_header

        new_path, _ = fits_file_pair
        header = read_header(str(new_path))
        assert header is not None


class TestWriteFits:
    """测试 FITS 文件写入"""

    def test_write_fits_16bit(self, tmp_dir, synth_fits_data_16bit):
        from scann.core.fits_io import read_fits, write_fits

        out = tmp_dir / "output_16.fits"
        write_fits(str(out), synth_fits_data_16bit, bit_depth=BitDepth.INT16)
        assert out.exists()

        # 读回验证
        result = read_fits(str(out))
        assert result.data.dtype in (np.uint16, np.int16)

    def test_write_fits_32bit(self, tmp_dir, synth_fits_data_32bit):
        from scann.core.fits_io import read_fits, write_fits

        out = tmp_dir / "output_32.fits"
        write_fits(str(out), synth_fits_data_32bit, bit_depth=BitDepth.INT32)
        assert out.exists()

        result = read_fits(str(out))
        assert result.data.dtype in (np.int32, np.uint32, np.float32)

    def test_write_fits_preserves_header(self, fits_file_pair, tmp_dir):
        """需求: 保存时绝不修改 FITS header"""
        from scann.core.fits_io import read_fits, write_fits

        new_path, _ = fits_file_pair
        original = read_fits(str(new_path))

        out = tmp_dir / "output_preserved.fits"
        write_fits(str(out), original.data, header=original.header, bit_depth=BitDepth.INT16)

        saved = read_fits(str(out))
        assert saved.header.raw["OBJECT"] == "TestField"
        assert saved.header.raw["TELESCOP"] == "TestScope"

    def test_write_fits_integer_only(self, tmp_dir, synth_float_image):
        """需求: 只保存整数格式 (16/32bit可选)"""
        from scann.core.fits_io import read_fits, write_fits

        out = tmp_dir / "output_int.fits"
        write_fits(str(out), synth_float_image, bit_depth=BitDepth.INT16)

        result = read_fits(str(out))
        # 必须是整数类型
        assert np.issubdtype(result.data.dtype, np.integer)
