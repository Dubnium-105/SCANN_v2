"""Query Service External APIs 测试

使用测试驱动开发 (TDD) 实现：
1. MPC 小行星/彗星查询
2. SIMBAD 天体查询
3. TNS 暂现源查询
4. 卫星 TLE 检查
"""

import pytest
import requests
from unittest.mock import Mock, patch
from datetime import datetime

from scann.services.query_service import QueryService, QueryResult


def _mpchecker_html(*lines: str) -> str:
    content = "\n".join(lines)
    return f"<html><body><pre>{content}</pre></body></html>"


class TestMPCQuery:
    """测试 MPC 小行星/彗星查询"""

    def test_query_mpc_basic(self):
        """测试：基本 MPC 查询"""
        service = QueryService()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = _mpchecker_html(
            " Object designation         R.A.      Decl.     V       Offsets     Motion/hr   Orbit  Further observations?",
            "                           h  m  s      \u00b0  '  \"        R.A.   Decl.  R.A.  Decl.        Comment",
            "            1 Ceres       10 30 00.0 +15 30 00   9.0   0.0E   0.0N     0+     0+    6o  None needed at this time.",
        )

        with patch("requests.post", return_value=mock_response):
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
        mock_response.status_code = 200
        mock_response.text = (
            "<html><body>No known minor planets, brighter than <i>V</i> = 24.0, "
            "were found in the 5.0-arcminute region around R.A. = 10 48 41.87, "
            "Decl. = +34 27 49.5 (J2000.0) on 2026 03 11.00 UT.</body></html>"
        )

        with patch("requests.post", return_value=mock_response):
            results = service.query_mpc(ra_deg=0.0, dec_deg=0.0)

        # 应该返回空列表
        assert results == []

    def test_query_mpc_with_radius(self):
        """测试：带搜索半径的 MPC 查询"""
        service = QueryService()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = _mpchecker_html(
            " Object designation         R.A.      Decl.     V       Offsets     Motion/hr   Orbit  Further observations?",
            "                           h  m  s      \u00b0  '  \"        R.A.   Decl.  R.A.  Decl.        Comment",
            "            4 Vesta       12 00 00.0 +20 00 00   8.0   0.0E   0.0N     0+     0+    6o  None needed at this time.",
        )

        with patch("requests.post", return_value=mock_response) as mock_post:
            results = service.query_mpc(ra_deg=180.0, dec_deg=20.0, radius_arcsec=600.0)

        assert mock_post.called
        assert len(results) == 1
        assert results[0].name == "4 Vesta"

    def test_query_mpc_network_error(self):
        """测试：网络错误处理"""
        service = QueryService()

        with patch("requests.post", side_effect=requests.RequestException("Network error")):
            results = service.query_mpc(ra_deg=0.0, dec_deg=0.0)

        assert results == []
        assert "Network error" in results.error

    def test_query_mpc_invalid_response_reports_mpchecker_error(self):
        service = QueryService()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "<html><title>Error from WebCS Script</title></html>"

        with patch("requests.post", return_value=mock_response):
            results = service.query_mpc(ra_deg=0.0, dec_deg=0.0)

        assert results == []
        assert "MPChecker" in results.error

    def test_query_mpc_distance_calculation(self):
        """测试：距离计算"""
        service = QueryService()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = _mpchecker_html(
            " Object designation         R.A.      Decl.     V       Offsets     Motion/hr   Orbit  Further observations?",
            "                           h  m  s      \u00b0  '  \"        R.A.   Decl.  R.A.  Decl.        Comment",
            "      99999 Test Asteroid  10 30 00.0 +15 30 00  12.0   0.0E   0.0N     0+     0+    6o  None needed at this time.",
        )

        with patch("requests.post", return_value=mock_response):
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

        with patch("requests.post", side_effect=requests.RequestException("Network error")):
            results = service.query_tns(ra_deg=0.0, dec_deg=0.0)

        assert results == []
        assert "旧公开端点不可用" in results.error or "Network error" in results.error

    def test_query_tns_401_reports_authentication_requirement(self):
        service = QueryService()

        legacy_error = requests.RequestException("legacy unavailable")
        auth_response = Mock()
        auth_response.status_code = 401

        with patch("requests.post", side_effect=[legacy_error, auth_response]):
            results = service.query_tns(ra_deg=0.0, dec_deg=0.0)

        assert results == []
        assert "需要认证" in results.error


class TestSatelliteCheck:
    """测试卫星检查"""

    def test_check_satellite_basic(self):
        """测试：基本卫星检查"""
        service = QueryService()

        mock_response = Mock()
        mock_response.text = "<html><body><pre>\n1 observations found\n0 objects\n0 objects after removing slow ones\n</pre></body></html>"
        mock_response.status_code = 200

        with patch("requests.post", return_value=mock_response):
            results = service.check_satellite(
                ra_deg=10.0,
                dec_deg=20.0,
                obs_datetime=datetime(2020, 1, 1, 12, 0, 0)
            )

        assert results.error == ""
        assert list(results) == []

    def test_check_satellite_no_data(self):
        """测试：无卫星数据时的检查"""
        service = QueryService()

        mock_response = Mock()
        mock_response.text = ""
        mock_response.status_code = 200

        with patch("requests.post", return_value=mock_response):
            results = service.check_satellite(ra_deg=0.0, dec_deg=0.0)

        # 应该返回空列表
        assert results == []

    def test_check_satellite_network_error(self):
        """测试：网络错误处理"""
        service = QueryService()

        with patch("requests.post", side_effect=Exception("Network error")):
            results = service.check_satellite(ra_deg=0.0, dec_deg=0.0)

        assert results == []
        assert "Network error" in results.error

    def test_check_satellite_distance_filtering(self):
        """测试：距离过滤"""
        service = QueryService()

        mock_response = Mock()
        mock_response.text = "<html><body><pre>\n1 observations found\n2 objects\n2 objects after removing slow ones\n     TLE01    C2026 01 18.58704 02 28 24.667+27 54 16.53         18.52V      500\n     10925U = 1978-055A   e=0.68; P=717.9 min; i=64.0: SAT-A\n             no observed motion (single obs)  dist= 19425.4 km; offset= 0.1000 deg\n             motion 31.5952\"/sec at PA 158.1 (computed)\n\n     TLE01    C2026 01 18.58704 02 28 24.667+27 54 16.53         18.52V      500\n     48262U = 1980-063F   e=0.72; P=702.0 min; i=63.6: SAT-B\n             no observed motion (single obs)  dist= 17546.5 km; offset= 0.3000 deg\n             motion 47.2848\"/sec at PA  31.1 (computed)\n</pre></body></html>"
        mock_response.status_code = 200

        with patch("requests.post", return_value=mock_response):
            results = service.check_satellite(
                ra_deg=0.0,
                dec_deg=0.0,
                obs_datetime=datetime(2026, 1, 18, 14, 5, 20),
                radius_arcsec=900.0,
            )

        assert [result.name for result in results] == ["SAT-A"]
