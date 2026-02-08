"""天文坐标模块单元测试"""

import pytest


class TestAstrometry:
    """测试坐标转换"""

    def test_format_ra_hms(self):
        from scann.core.astrometry import format_ra_hms

        # 180 度 = 12h 00m 00.00s
        result = format_ra_hms(180.0)
        assert "12" in result
        assert "00" in result

    def test_format_dec_dms(self):
        from scann.core.astrometry import format_dec_dms

        # +45.5 度
        result = format_dec_dms(45.5)
        assert "45" in result
        assert "+" in result or result[0].isdigit()

    def test_format_ra_zero(self):
        from scann.core.astrometry import format_ra_hms

        result = format_ra_hms(0.0)
        assert "00" in result

    def test_format_dec_negative(self):
        from scann.core.astrometry import format_dec_dms

        result = format_dec_dms(-30.25)
        assert "-" in result
        assert "30" in result

    def test_format_ra_edge_360(self):
        from scann.core.astrometry import format_ra_hms

        # 360度应等于 0h
        result = format_ra_hms(360.0)
        # 取模后应该为 0h
        assert "00" in result or "24" in result
