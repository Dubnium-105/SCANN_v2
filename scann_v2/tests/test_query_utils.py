from scann.services.query_utils import calculate_distance, dms_to_degrees, hms_to_degrees


def test_hms_to_degrees_supports_hms_format():
    assert hms_to_degrees("12:00:00") == 180.0


def test_dms_to_degrees_supports_signed_dms_format():
    assert dms_to_degrees("-30:30:00") == -30.5


def test_calculate_distance_returns_arcseconds():
    assert abs(calculate_distance(180.0, 0.0, 181.0, 0.0) - 3600.0) < 1.0