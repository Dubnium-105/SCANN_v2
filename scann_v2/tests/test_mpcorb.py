"""MPCORB 模块单元测试"""

import pytest


class TestMpcorbParser:
    """测试 MPCORB 文件解析"""

    SAMPLE_MPCORB_LINE = (
        "00001    3.34  0.12 K249A  25.07130   73.41651   80.26070   10.58687"
        "  0.0785209  0.21407094   2.7670940  0 MPO722043  6751  92 1801-2024"
        " 0.60 M-v 30h MPCLINUX   0000      (1) Ceres              20240917"
    )

    def test_parse_mpcorb_line(self):
        from scann.core.mpcorb import _parse_mpcorb_line

        orbit = _parse_mpcorb_line(self.SAMPLE_MPCORB_LINE)
        assert orbit is not None
        assert orbit.designation == "00001"

    def test_filter_by_magnitude(self):
        from scann.core.mpcorb import AsteroidOrbit, filter_by_magnitude

        orbits = [
            AsteroidOrbit(designation="00001", abs_magnitude=3.34, epoch=0.0,
                          mean_anomaly=25.07, arg_perihelion=73.42,
                          ascending_node=80.26, inclination=10.59,
                          eccentricity=0.0785,
                          semi_major_axis=2.767, slope_param=0.12),
            AsteroidOrbit(designation="99999", abs_magnitude=22.0, epoch=0.0,
                          mean_anomaly=0.0, arg_perihelion=0.0,
                          ascending_node=0.0, inclination=0.0,
                          eccentricity=0.0,
                          semi_major_axis=0.0, slope_param=0.15),
        ]
        bright = filter_by_magnitude(orbits, limit_mag=10.0)
        assert len(bright) == 1
        assert bright[0].designation == "00001"

    def test_empty_line_returns_none(self):
        from scann.core.mpcorb import _parse_mpcorb_line

        result = _parse_mpcorb_line("")
        assert result is None

    def test_short_line_returns_none(self):
        from scann.core.mpcorb import _parse_mpcorb_line

        result = _parse_mpcorb_line("too short")
        assert result is None

    def test_unpack_epoch_k249a(self):
        """测试解包packed epoch格式 K249A"""
        from scann.core.mpcorb import _unpack_packed_epoch

        # K249A = K(2000s) + 24(年份) + 9A(月日分数)
        # K = 2000s century prefix
        # 24 = 2024
        # 9A = 9.x (需要解析为月份/日)
        jd = _unpack_packed_epoch("K249A")
        # 2024年9月某日，JD应该在2460000附近
        assert isinstance(jd, float)
        assert jd > 2460000  # 2024年9月的JD大约是2460580

    def test_unpack_epoch_j8300(self):
        """测试解包packed epoch格式 J8300 (旧格式示例)"""
        from scann.core.mpcorb import _unpack_packed_epoch

        # J = 1900s century prefix
        # 83 = 1983
        # 00 = 年初
        jd = _unpack_packed_epoch("J8300")
        assert isinstance(jd, float)
        assert jd > 2440000  # 1983年初的JD大约是2445300

    def test_epoch_is_set_in_parsed_orbit(self):
        """测试解析后epoch被正确设置"""
        from scann.core.mpcorb import _parse_mpcorb_line

        orbit = _parse_mpcorb_line(self.SAMPLE_MPCORB_LINE)
        assert orbit is not None
        assert orbit.epoch > 0.0  # epoch应该被正确解析，不再是0.0
