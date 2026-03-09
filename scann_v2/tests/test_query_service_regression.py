from datetime import datetime
from unittest.mock import Mock, patch

from scann.services.query_service import QueryResult, QueryService


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
        assert result.raw_data == {}

    def test_query_mpc_returns_key_fields_with_formatted_name_and_url(self):
        service = QueryService(timeout=3)
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
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

        with patch("requests.get", return_value=response):
            results = service.query_mpc(157.5, 15.5)

        assert len(results) == 1
        result = results[0]
        assert result.source == "MPC"
        assert result.name == "1 Ceres"
        assert result.object_type == "asteroid"
        assert result.magnitude == 9.0
        assert "show_object?object_id=1 Ceres" in result.url
        assert result.raw_data["name"] == "Ceres"

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
        response.text = "ISS (ZARYA)\n1 25544U 98067A   20001.00000000  .00000000  00000-0  00000-0 0  9999\n2 25544  51.6416 247.4627 0004576 359.2713 200.8514 15.49135398 12345"

        with patch("requests.get", return_value=response):
            results = service.check_satellite(
                ra_deg=10.0,
                dec_deg=20.0,
                obs_datetime=datetime(2020, 1, 1, 12, 0, 0),
            )

        assert len(results) == 1
        result = results[0]
        assert result.source == "Satellite"
        assert result.name == "ISS (ZARYA)"
        assert result.object_type == "satellite"
        assert result.url.endswith("ISS (ZARYA)")
        assert result.raw_data["line1"].startswith("1 25544U")
        assert result.raw_data["line2"].startswith("2 25544")