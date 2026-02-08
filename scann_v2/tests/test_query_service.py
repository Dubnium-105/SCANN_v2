"""Query Service 测试

使用测试驱动开发 (TDD) 实现：
1. 计算查询结果与目标位置的距离
"""

import pytest

from scann.services.query_service import QueryService, QueryResult


class TestQueryServiceDistance:
    """测试距离计算功能"""

    def test_calculate_distance_same_position(self):
        """测试：相同位置距离应为0"""
        # 准备
        result = QueryResult(
            source="VSX",
            name="V1234 Cen",
            object_type="Delta Cephei",
            distance_arcsec=0.0,
        )

        # VSX API 返回的数据格式（示例）
        vsx_item = {
            "Name": "V1234 Cen",
            "RA": 13.2500,  # 度
            "Dec": -45.5000,  # 度
            "Type": "Delta Cephei",
        }
        target_ra = 13.2500
        target_dec = -45.5000

        # 执行
        distance = QueryService._calculate_distance(
            vsx_item["RA"], vsx_item["Dec"], target_ra, target_dec
        )

        # 断言
        assert distance == 0.0

    def test_calculate_distance_known_separation(self):
        """测试：已知距离的位置"""
        # 测试赤经上差1度的情况（在赤纬0度附近，1度 RA = 3600角秒）
        # 在赤纬0度，RA 1度的弧长 = 3600角秒
        vsx_item = {
            "Name": "Test Star",
            "RA": 180.0,  # 度
            "Dec": 0.0,  # 度
        }
        target_ra = 181.0
        target_dec = 0.0

        distance = QueryService._calculate_distance(
            vsx_item["RA"], vsx_item["Dec"], target_ra, target_dec
        )

        # 在赤纬0度，1度的赤经差约等于3600角秒
        assert abs(distance - 3600.0) < 1.0

    def test_calculate_distance_with_dec_offset(self):
        """测试：包含赤纬偏移的"""
        # 纯赤纬偏移
        vsx_item = {
            "Name": "Test Star",
            "RA": 180.0,
            "Dec": 0.0,
        }
        target_ra = 180.0
        target_dec = 1.0

        distance = QueryService._calculate_distance(
            vsx_item["RA"], vsx_item["Dec"], target_ra, target_dec
        )

        # 1度赤纬 = 3600角秒
        assert abs(distance - 3600.0) < 1.0

    def test_calculate_distance_diagonal(self):
        """测试：对角线距离"""
        # 1度RA + 1度Dec 的对角线距离
        # 在赤纬0度附近，近似为 sqrt(3600^2 + 3600^2) = 5091角秒
        vsx_item = {
            "Name": "Test Star",
            "RA": 180.0,
            "Dec": 0.0,
        }
        target_ra = 181.0
        target_dec = 1.0

        distance = QueryService._calculate_distance(
            vsx_item["RA"], vsx_item["Dec"], target_ra, target_dec
        )

        # 理论距离约为 5091 角秒
        expected = 3600.0 * 2**0.5
        assert abs(distance - expected) < 5.0

    def test_calculate_distance_high_declination(self):
        """测试：高赤纬位置的RA压缩"""
        # 在赤纬60度，1度RA的弧长会减小
        vsx_item = {
            "Name": "Test Star",
            "RA": 0.0,
            "Dec": 60.0,
        }
        target_ra = 1.0
        target_dec = 60.0

        distance = QueryService._calculate_distance(
            vsx_item["RA"], vsx_item["Dec"], target_ra, target_dec
        )

        # 在赤纬60度，cos(60)=0.5，所以1度RA = 1800角秒
        assert abs(distance - 1800.0) < 10.0

    def test_query_vsx_includes_distance(self):
        """测试：VSX查询结果应包含距离"""
        # 这个测试需要mock或实际API响应
        # 先实现距离计算功能
        # 集成测试可以后续添加
        pass
