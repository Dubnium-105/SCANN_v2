"""CoordinateLabel 兼容性测试"""

import pytest


def test_coordinate_label_set_wcs_coordinates_alias(qapp):
    pytest.importorskip("PyQt5")
    from scann.gui.widgets.coordinate_label import CoordinateLabel

    label = CoordinateLabel("")
    label.set_wcs_coordinates("12:34:56", "+01:02:03")
    assert "RA: 12:34:56" in label.text()
    assert "Dec: +01:02:03" in label.text()
