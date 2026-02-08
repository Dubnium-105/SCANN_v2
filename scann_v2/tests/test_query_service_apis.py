"""Query Service External APIs 测试

使用测试驱动开发 (TDD) 实现：
1. MPC 小行星/彗星查询
2. SIMBAD 天体查询
3. TNS 暂现源查询
4. 卫星 TLE 检查
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from scann.services.query_service import QueryService, QueryResult


class TestMPCQuery:
    """测试 MPC 小行星/彗星查询"""

    def test_query_mpc_basic(self):
        """测试：基本 MPC 查询"""
        service = QueryService()

        # Mock HTTP 响应
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [
                {
                    "name": "Ceres",
                    "number": "1",
                    "ra": "10:30:00",
                    "dec": "+15:30:00",
                    "v": "9.0",
                }
            ]
        }
        mock_response.status_code = 200

        with patch("requests.get", return_value=mock_response):
            results = service.query_mpc(ra_deg=157.5, dec_deg=15.5)

        # 应该返回一个结果
        assert len(results) == 1
        assert results[0].source == "MPC"
        assert results[0].name == "1 Ceres"
        assert results[0].object_type == "asteroid"
        assert results[0].magnitude == 9.0

    def test_query_mpc_empty_results(self):
        """测试：无结果的 MPC 查询"""
        service = QueryService()

        mock_response = Mock()
        mock_response.json.return_value = {"results": []}
        mock_response.status_code = 200

        with patch("requests.get", return_value=mock_response):
            results = service.query_mpc(ra_deg=0.0, dec_deg=0.0)

        # 应该返回空列表
        assert results == []

    def test_query_mpc_with_radius(self):
        """测试：带搜索半径的 MPC 查询"""
        service = QueryService()

        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [
                {
                    "name": "Vesta",
                    "number": "4",
                    "ra": "12:00:00",
                    "dec": "+20:00:00",
                    "v": "8.0",
                }
            ]
        }
        mock_response.status_code = 200

        with patch("requests.get", return_value=mock_response) as mock_get:
            results = service.query_mpc(ra_deg=180.0, dec_deg=20.0, radius_arcsec=600.0)

        # 应该调用正确的 URL
        assert mock_get.called
        assert len(results) == 1
        assert results[0].name == "4 Vesta"

    def test_query_mpc_network_error(self):
        """测试：网络错误处理"""
        service = QueryService()

        with patch("requests.get", side_effect=Exception("Network error")):
            results = service.query_mpc(ra_deg=0.0, dec_deg=0.0)

        # 应该返回空列表而不是抛出异常
        assert results == []

    def test_query_mpc_distance_calculation(self):
        """测试：距离计算"""
        service = QueryService()

        # 测试数据：目标位置和查询位置相同
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [
                {
                    "name": "Test Asteroid",
                    "number": "99999",
                    "ra": "10:30:00",  # 157.5 度
                    "dec": "+15:30:00",  # 15.5 度
                    "v": "12.0",
                }
            ]
        }
        mock_response.status_code = 200

        with patch("requests.get", return_value=mock_response):
            results = service.query_mpc(ra_deg=157.5, dec_deg=15.5)

        # 距离应该接近 0
        assert len(results) == 1
        assert results[0].distance_arcsec < 1.0


class TestSIMBADQuery:
    """测试 SIMBAD 天体查询"""

    @pytest.mark.skipif(
        True,  # 需要 astroquery，暂时跳过
        reason="需要 astroquery 包"
    )
    def test_query_simbad_basic(self):
        """测试：基本 SIMBAD 查询"""
        service = QueryService()

        results = service.query_simbad(ra_deg=0.0, dec_deg=0.0)

        # SIMBAD 查询应该返回结果列表
        assert isinstance(results, list)

    @pytest.mark.skipif(
        True,
        reason="需要 astroquery 包"
    )
    def test_query_simbad_empty(self):
        """测试：空区域的 SIMBAD 查询"""
        service = QueryService()

        results = service.query_simbad(ra_deg=100.0, dec_deg=-90.0)

        # 应该返回空列表
        assert results == []


class TestTNSQuery:
    """测试 TNS 暂现源查询"""

    def test_query_tns_basic(self):
        """测试：基本 TNS 查询"""
        service = QueryService()

        # Mock HTTP 响应
        mock_response = Mock()
        mock_response.json.return_value = {
            "object": {
                "name": "AT2020abc",
                "objtype": "12",  # Supernova
                "ra": "12:30:00",
                "dec": "+45:00:00",
                "mag": "15.0",
            }
        }
        mock_response.status_code = 200

        with patch("requests.post", return_value=mock_response):
            results = service.query_tns(ra_deg=187.5, dec_deg=45.0)

        # 应该返回一个结果
        assert len(results) == 1
        assert results[0].source == "TNS"
        assert results[0].name == "AT2020abc"

    def test_query_tns_empty_results(self):
        """测试：无结果的 TNS 查询"""
        service = QueryService()

        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_response.status_code = 200

        with patch("requests.post", return_value=mock_response):
            results = service.query_tns(ra_deg=0.0, dec_deg=0.0)

        # 应该返回空列表
        assert results == []

    def test_query_tns_network_error(self):
        """测试：网络错误处理"""
        service = QueryService()

        with patch("requests.post", side_effect=Exception("Network error")):
            results = service.query_tns(ra_deg=0.0, dec_deg=0.0)

        # 应该返回空列表
        assert results == []


class TestSatelliteCheck:
    """测试卫星检查"""

    def test_check_satellite_basic(self):
        """测试：基本卫星检查"""
        service = QueryService()

        # Mock TLE 数据
        mock_response = Mock()
        mock_response.text = "1 25544U 98067A   20001.00000000  .00000000  00000-0  00000-0 0  9999\n2 25544  51.6416 247.4627 0004576 359.2713 200.8514 15.49135398 12345"
        mock_response.status_code = 200

        with patch("requests.get", return_value=mock_response):
            results = service.check_satellite(
                ra_deg=10.0,
                dec_deg=20.0,
                obs_datetime=datetime(2020, 1, 1, 12, 0, 0)
            )

        # 应该返回结果列表
        assert isinstance(results, list)

    def test_check_satellite_no_data(self):
        """测试：无卫星数据时的检查"""
        service = QueryService()

        mock_response = Mock()
        mock_response.text = ""
        mock_response.status_code = 200

        with patch("requests.get", return_value=mock_response):
            results = service.check_satellite(ra_deg=0.0, dec_deg=0.0)

        # 应该返回空列表
        assert results == []

    def test_check_satellite_network_error(self):
        """测试：网络错误处理"""
        service = QueryService()

        with patch("requests.get", side_effect=Exception("Network error")):
            results = service.check_satellite(ra_deg=0.0, dec_deg=0.0)

        # 应该返回空列表
        assert results == []

    def test_check_satellite_distance_filtering(self):
        """测试：距离过滤"""
        service = QueryService()

        mock_response = Mock()
        mock_response.text = "1 25544U 98067A   20001.00000000  .00000000  00000-0  00000-0 0  9999\n2 25544  51.6416 247.4627 0004576 359.2713 200.8514 15.49135398 12345"
        mock_response.status_code = 200

        with patch("requests.get", return_value=mock_response):
            results = service.check_satellite(
                ra_deg=0.0,
                dec_deg=0.0,
                obs_datetime=datetime(2020, 1, 1, 12, 0, 0)
            )

        # 所有结果都应该有距离信息
        for result in results:
            assert result.distance_arcsec >= 0.0
