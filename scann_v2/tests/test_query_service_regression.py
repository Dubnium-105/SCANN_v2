from datetime import datetime
from unittest.mock import Mock, patch

from scann.services.query_service import QueryResult, QueryService


def _mpchecker_html(*lines: str) -> str:
    content = "\n".join(lines)
    return f"<html><body><pre>{content}</pre></body></html>"


class TestQueryServiceRegression:
    def test_query_vsx_returns_normalized_query_result_objects(self):
        service = QueryService(timeout=3)
        response = Mock()
        response.json.return_value = {
            "VSXObjects": {
                "VSXObject": [
                    {
                        "Name": "V1234 Sgr",
                        "Type": "EA",
                        "RA": "12:00:00",
                        "Dec": "+30:00:00",
                    }
                ]
            }
        }
        response.raise_for_status.return_value = None

        with patch("requests.get", return_value=response):
            results = service.query_vsx(180.0, 30.0)

        assert len(results) == 1
        result = results[0]
        assert isinstance(result, QueryResult)
        assert result.source == "VSX"
        assert result.name == "V1234 Sgr"
        assert result.object_type == "EA"
        assert result.distance_arcsec == 0.0
        assert result.raw_data["Type"] == "EA"

    def test_query_vsx_supports_current_api_field_names(self):
        service = QueryService(timeout=3)
        response = Mock()
        response.json.return_value = {
            "VSXObjects": {
                "VSXObject": [
                    {
                        "Name": "HAT-182-0006221",
                        "RA2000": "10.50000",
                        "Declination2000": "-20.25000",
                        "VariabilityType": "EW",
                        "Category": "Variable",
                    }
                ]
            }
        }
        response.raise_for_status.return_value = None

        with patch("requests.get", return_value=response):
            results = service.query_vsx(10.5, -20.25)

        assert len(results) == 1
        result = results[0]
        assert result.name == "HAT-182-0006221"
        assert result.object_type == "EW"
        assert result.distance_arcsec == 0.0
        assert result.raw_data["Category"] == "Variable"

    def test_query_mpc_returns_key_fields_with_formatted_name_and_url(self):
        service = QueryService(timeout=3)
        response = Mock()
        response.status_code = 200
        response.text = _mpchecker_html(
            " Object designation         R.A.      Decl.     V       Offsets     Motion/hr   Orbit  Further observations?",
            "                           h  m  s      \u00b0  '  \"        R.A.   Decl.  R.A.  Decl.        Comment",
            "            1 Ceres       10 30 00.0 +15 30 00   9.0   0.0E   0.0N     0+     0+    6o  None needed at this time.",
        )

        with patch("requests.post", return_value=response):
            results = service.query_mpc(157.5, 15.5)

        assert len(results) == 1
        result = results[0]
        assert result.source == "MPC"
        assert result.name == "1 Ceres"
        assert result.object_type == "asteroid"
        assert result.magnitude == 9.0
        assert "show_object?object_id=1 Ceres" in result.url
        assert "None needed" in result.raw_data["comment"]

    def test_query_tns_returns_transient_result_with_url(self):
        service = QueryService(timeout=3)
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "object": {
                "name": "AT2026xyz",
                "objtype": "12",
                "ra": "12:30:00",
                "dec": "+45:00:00",
                "discovery_data": {"mag": "17.2"},
            }
        }

        with patch("requests.post", return_value=response):
            results = service.query_tns(187.5, 45.0)

        assert len(results) == 1
        result = results[0]
        assert result.source == "TNS"
        assert result.name == "AT2026xyz"
        assert result.object_type == "Supernova"
        assert result.magnitude == 17.2
        assert result.url.endswith("/AT2026xyz")
        assert result.raw_data["name"] == "AT2026xyz"

    def test_check_satellite_returns_query_result_list_with_raw_tle_fields(self):
        service = QueryService(timeout=3)
        response = Mock()
        response.status_code = 200
        response.text = "<html><body><pre>\n1 observations found\n1 objects\n1 objects after removing slow ones\n     TLE01    C2026 01 18.58704 02 28 24.667+27 54 16.53         18.52V      500\n     10925U = 1978-055A   e=0.68; P=717.9 min; i=64.0: MOLNIYA 1-40\n             no observed motion (single obs)  dist= 19425.4 km; offset= 2.1832 deg\n             motion 31.5952\"/sec at PA 158.1 (computed)\n</pre></body></html>"

        with patch("requests.post", return_value=response):
            results = service.check_satellite(
                ra_deg=10.0,
                dec_deg=20.0,
                obs_datetime=datetime(2026, 1, 18, 14, 5, 20),
                radius_arcsec=5.0 * 3600.0,
            )

        assert len(results) == 1
        result = results[0]
        assert result.source == "Satellite"
        assert result.name == "MOLNIYA 1-40"
        assert result.object_type == "satellite"
        assert result.url.endswith("sat_id2.htm")
        assert round(result.distance_arcsec, 2) == round(2.1832 * 3600.0, 2)
        assert result.raw_data["norad_id"] == "10925U"
        assert result.raw_data["international_designator"] == "1978-055A"

    def test_execute_query_returns_unsupported_type_error(self):
        service = QueryService(timeout=3)

        response = service.execute_query("unknown", 10.0, 20.0)

        assert response == []
        assert response.error == "不支持的查询类型: unknown"