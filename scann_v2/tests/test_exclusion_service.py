"""Exclusion Service 测试

使用测试驱动开发 (TDD) 实现：
1. 坐标匹配逻辑：将候选体与已知小行星匹配
"""

import pytest
from unittest.mock import Mock, patch

from scann.core.models import Candidate, CandidateFeatures, FitsHeader, ObservatoryConfig, SkyPosition
from scann.core.mpcorb import AsteroidOrbit
from scann.services.exclusion_service import ExclusionService


class TestExclusionServiceMatching:
    """测试坐标匹配功能"""

    def test_candidate_marked_as_known_when_matches_asteroid(self):
        """测试：匹配的候选体应标记为已知"""
        # 准备
        service = ExclusionService(
            observatory=ObservatoryConfig(),
            limit_magnitude=20.0
        )

        # 模拟一个小行星在特定位置
        mock_asteroid = Mock(spec=AsteroidOrbit)
        mock_asteroid.designation = "2024 ABC"
        mock_asteroid.ra = 180.0  # 度
        mock_asteroid.dec = 0.0   # 度
        mock_asteroid.mag = 18.5
        service._asteroids = [mock_asteroid]

        # 准备候选体（与已知小行星位置相同）
        candidates = [
            Candidate(x=100, y=100, features=CandidateFeatures()),
        ]

        # 准备 FITS 头（包含 WCS 信息）
        # 假设 (100, 100) 像素对应 (180.0, 0.0) 的天球坐标
        header = FitsHeader(raw={
            "CRVAL1": 180.0,  # 参考像素的 RA
            "CRVAL2": 0.0,    # 参考像素的 Dec
            "CRPIX1": 100.0,  # 参考像素 X
            "CRPIX2": 100.0,  # 参考像素 Y
            "CDELT1": 1.0/3600.0,  # 每像素 1 角秒 (度)
            "CDELT2": 1.0/3600.0,
            "RA": 180.0,
            "DEC": 0.0,
        })

        # 执行
        with patch.object(service, '_pixel_to_sky', return_value=SkyPosition(ra=180.0, dec=0.0)):
            result = service.check_candidates(candidates, header)

        # 断言：候选体应被标记为已知
        assert len(result) == 1
        assert result[0].is_known == True
        assert result[0].known_id == "2024 ABC"

    def test_candidate_not_marked_when_no_match(self):
        """测试：不匹配的候选体不应标记"""
        # 准备
        service = ExclusionService()

        # 小行星在 (180, 0)
        mock_asteroid = Mock(spec=AsteroidOrbit)
        mock_asteroid.designation = "2024 ABC"
        mock_asteroid.ra = 180.0
        mock_asteroid.dec = 0.0
        mock_asteroid.mag = 18.5
        service._asteroids = [mock_asteroid]

        # 候选体在 (181, 1) - 远离小行星
        candidates = [
            Candidate(x=200, y=200, features=CandidateFeatures()),
        ]

        header = FitsHeader(raw={})

        # 执行
        with patch.object(service, '_pixel_to_sky', return_value=SkyPosition(ra=181.0, dec=1.0)):
            result = service.check_candidates(candidates, header)

        # 断言
        assert len(result) == 1
        assert result[0].is_known == False
        assert result[0].known_id == ""

    def test_pixel_to_sky_conversion(self):
        """测试：像素坐标转天球坐标"""
        service = ExclusionService()

        # 简单 WCS：参考像素(100,100)对应(180,0)，每像素1角秒
        header = FitsHeader(raw={
            "CRVAL1": 180.0,
            "CRVAL2": 0.0,
            "CRPIX1": 100.0,
            "CRPIX2": 100.0,
            "CDELT1": 1.0/3600.0,
            "CDELT2": 1.0/3600.0,
        })

        # 执行
        position = service._pixel_to_sky(header, 100, 100)

        # 断言
        assert position.ra == pytest.approx(180.0)
        assert position.dec == pytest.approx(0.0)

    def test_pixel_to_sky_with_offset(self):
        """测试：偏移的像素坐标"""
        service = ExclusionService()

        header = FitsHeader(raw={
            "CRVAL1": 180.0,
            "CRVAL2": 0.0,
            "CRPIX1": 100.0,
            "CRPIX2": 100.0,
            "CDELT1": 1.0/3600.0,
            "CDELT2": 1.0/3600.0,
        })

        # 像素(101, 101) 应该是 (180 + 1角秒, 0 + 1角秒)
        position = service._pixel_to_sky(header, 101, 101)

        # 1角秒 = 1/3600 度
        expected_ra = 180.0 + 1.0/3600.0
        expected_dec = 0.0 + 1.0/3600.0

        assert position.ra == pytest.approx(expected_ra, abs=1e-6)
        assert position.dec == pytest.approx(expected_dec, abs=1e-6)

    def test_matching_within_radius(self):
        """测试：在匹配半径内的应标记"""
        service = ExclusionService()

        # 小行星位置
        mock_asteroid = Mock(spec=AsteroidOrbit)
        mock_asteroid.designation = "2024 XYZ"
        mock_asteroid.ra = 180.0
        mock_asteroid.dec = 0.0
        mock_asteroid.mag = 18.0
        service._asteroids = [mock_asteroid]

        # 候选体在 3 角秒处（默认匹配半径 5 角秒）
        candidates = [Candidate(x=100, y=100, features=CandidateFeatures())]

        header = FitsHeader(raw={})

        # 候选体天球坐标偏离小行星 3 角秒
        with patch.object(service, '_pixel_to_sky',
                         return_value=SkyPosition(ra=180.0 + 3.0/3600.0, dec=0.0)):
            result = service.check_candidates(candidates, header)

        # 应该匹配
        assert result[0].is_known == True

    def test_no_matching_outside_radius(self):
        """测试：超出匹配半径的不应标记"""
        service = ExclusionService()

        mock_asteroid = Mock(spec=AsteroidOrbit)
        mock_asteroid.designation = "2024 XYZ"
        mock_asteroid.ra = 180.0
        mock_asteroid.dec = 0.0
        mock_asteroid.mag = 18.0
        service._asteroids = [mock_asteroid]

        candidates = [Candidate(x=100, y=100, features=CandidateFeatures())]

        header = FitsHeader(raw={})

        # 候选体天球坐标偏离小行星 10 角秒（超出默认5角秒半径）
        with patch.object(service, '_pixel_to_sky',
                         return_value=SkyPosition(ra=180.0 + 10.0/3600.0, dec=0.0)):
            result = service.check_candidates(candidates, header)

        # 不应该匹配
        assert result[0].is_known == False
