from datetime import datetime
from unittest.mock import Mock, patch

from scann.services.query_clients import MpcClient, SatelliteClient, TnsClient, VsxClient


def test_vsx_client_normalizes_results():
    client = VsxClient(timeout=3)
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
        results = client.query(180.0, 30.0)

    assert len(results) == 1
    assert results[0].source == "VSX"
    assert results[0].name == "V1234 Sgr"


def test_mpc_client_formats_name_and_distance():
    client = MpcClient(timeout=3)
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
        results = client.query(157.5, 15.5)

    assert len(results) == 1
    assert results[0].name == "1 Ceres"
    assert results[0].distance_arcsec < 1.0


def test_tns_client_maps_object_type_and_magnitude():
    client = TnsClient(timeout=3)
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
        results = client.query(187.5, 45.0)

    assert len(results) == 1
    assert results[0].object_type == "Supernova"
    assert results[0].magnitude == 17.2


def test_satellite_client_parses_tle_triplets():
    client = SatelliteClient(timeout=3)
    response = Mock()
    response.status_code = 200
    response.text = "ISS (ZARYA)\n1 25544U 98067A   20001.00000000  .00000000  00000-0  00000-0 0  9999\n2 25544  51.6416 247.4627 0004576 359.2713 200.8514 15.49135398 12345"

    with patch("requests.get", return_value=response):
        results = client.check(10.0, 20.0, datetime(2020, 1, 1, 12, 0, 0))

    assert len(results) == 1
    assert results[0].name == "ISS (ZARYA)"
    assert results[0].raw_data["line1"].startswith("1 25544U")