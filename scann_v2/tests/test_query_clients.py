from datetime import datetime
from unittest.mock import Mock, patch

import requests

from scann.services.query_clients import MpcClient, SatelliteClient, TnsClient, VsxClient


def _mpchecker_html(*lines: str) -> str:
    content = "\n".join(lines)
    return f"<html><body><pre>{content}</pre></body></html>"


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
    assert results[0].object_type == "EA"


def test_vsx_client_supports_current_api_field_names():
    client = VsxClient(timeout=3)
    response = Mock()
    response.json.return_value = {
        "VSXObjects": {
            "VSXObject": [
                {
                    "Name": "Gaia DR3\n738410813551247232",
                    "RA2000": "359.60107",
                    "Declination2000": "0.90172",
                    "VariabilityType": "ROT",
                    "Category": "Variable",
                }
            ]
        }
    }
    response.raise_for_status.return_value = None

    with patch("requests.get", return_value=response):
        results = client.query(359.60107, 0.90172)

    assert len(results) == 1
    assert results[0].name == "Gaia DR3 738410813551247232"
    assert results[0].object_type == "ROT"
    assert results[0].distance_arcsec == 0.0
    assert results[0].raw_data["Category"] == "Variable"


def test_vsx_client_retries_without_proxy_after_proxy_error():
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

    session = Mock()
    session.get.return_value = response

    with patch("requests.get", side_effect=requests.exceptions.ProxyError("bad proxy")):
        with patch("requests.Session", return_value=session):
            results = client.query(180.0, 30.0)

    assert len(results) == 1
    assert results[0].name == "V1234 Sgr"
    assert session.trust_env is False
    session.get.assert_called_once()
    session.close.assert_called_once_with()


def test_mpc_client_formats_name_and_distance():
    client = MpcClient(timeout=3)
    response = Mock()
    response.status_code = 200
    response.text = _mpchecker_html(
        " Object designation         R.A.      Decl.     V       Offsets     Motion/hr   Orbit  Further observations?",
        "                           h  m  s      \u00b0  '  \"        R.A.   Decl.  R.A.  Decl.        Comment",
        "            1 Ceres       10 30 00.0 +15 30 00   9.0   0.0E   0.0N     0+     0+    6o  None needed at this time.",
    )

    with patch("requests.post", return_value=response):
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
    response.text = "<html><body><pre>\n1 observations found\n1 objects\n1 objects after removing slow ones\n     TLE01    C2026 01 18.58704 02 28 24.667+27 54 16.53         18.52V      500\n     10925U = 1978-055A   e=0.68; P=717.9 min; i=64.0: MOLNIYA 1-40\n             no observed motion (single obs)  dist= 19425.4 km; offset= 2.1832 deg\n             motion 31.5952\"/sec at PA 158.1 (computed)\n</pre></body></html>"

    with patch("requests.post", return_value=response):
        results = client.check(
            10.0,
            20.0,
            datetime(2026, 1, 18, 14, 5, 20),
            radius_arcsec=5.0 * 3600.0,
        )

    assert len(results) == 1
    assert results[0].name == "MOLNIYA 1-40"
    assert round(results[0].distance_arcsec, 2) == round(2.1832 * 3600.0, 2)
    assert results[0].raw_data["norad_id"] == "10925U"


def test_satellite_client_filters_and_sorts_by_distance():
    client = SatelliteClient(timeout=3)
    response = Mock()
    response.status_code = 200
    response.text = "<html><body><pre>\n1 observations found\n2 objects\n2 objects after removing slow ones\n     TLE01    C2026 01 18.58704 02 28 24.667+27 54 16.53         18.52V      500\n     10925U = 1978-055A   e=0.68; P=717.9 min; i=64.0: SAT-A\n             no observed motion (single obs)  dist= 19425.4 km; offset= 0.1000 deg\n             motion 31.5952\"/sec at PA 158.1 (computed)\n\n     TLE01    C2026 01 18.58704 02 28 24.667+27 54 16.53         18.52V      500\n     48262U = 1980-063F   e=0.72; P=702.0 min; i=63.6: SAT-B\n             no observed motion (single obs)  dist= 17546.5 km; offset= 0.3000 deg\n             motion 47.2848\"/sec at PA  31.1 (computed)\n</pre></body></html>"

    with patch("requests.post", return_value=response):
        results = client.check(
            10.0,
            20.0,
            datetime(2026, 1, 18, 14, 5, 20),
            radius_arcsec=900.0,
        )

    assert [result.name for result in results] == ["SAT-A"]