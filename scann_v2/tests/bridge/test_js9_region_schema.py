"""JS9 Region Schema 校验测试

测试 JS9RegionRecord 数据模型的 schema 校验功能
"""

import pytest
from pydantic import ValidationError


def test_valid_box_region(bridge_module):
    """测试有效的 box region"""
    data = {
        "shape": "box",
        "x": 100.5,
        "y": 200.3,
        "width": 50.0,
        "height": 60.0,
        "label": "real",
        "detail_type": "asteroid",
        "confidence": 0.95,
    }
    region = bridge_module.JS9RegionRecord.from_json(data)
    assert region.shape == "box"
    assert region.x == 100.5
    assert region.y == 200.3
    assert region.width == 50.0
    assert region.height == 60.0
    assert region.label == "real"
    assert region.detail_type == "asteroid"
    assert region.confidence == 0.95


def test_valid_circle_region(bridge_module):
    """测试有效的 circle region"""
    data = {
        "shape": "circle",
        "x": 300.0,
        "y": 400.0,
        "radius": 25.5,
        "label": "bogus",
        "detail_type": "noise",
        "confidence": 0.8,
    }
    region = bridge_module.JS9RegionRecord.from_json(data)
    assert region.shape == "circle"
    assert region.x == 300.0
    assert region.y == 400.0
    assert region.radius == 25.5
    assert region.label == "bogus"
    assert region.detail_type == "noise"


def test_valid_polygon_region(bridge_module):
    """测试有效的 polygon region"""
    data = {
        "shape": "polygon",
        "x": 150.0,
        "y": 250.0,
        "width": 100.0,
        "height": 80.0,
        "label": "real",
        "detail_type": "supernova",
    }
    region = bridge_module.JS9RegionRecord.from_json(data)
    assert region.shape == "polygon"
    assert region.label == "real"
    assert region.detail_type == "supernova"


def test_minimal_region(bridge_module):
    """测试最小必填字段的 region"""
    data = {
        "shape": "box",
        "x": 10.0,
        "y": 20.0,
    }
    region = bridge_module.JS9RegionRecord.from_json(data)
    assert region.shape == "box"
    assert region.x == 10.0
    assert region.y == 20.0
    assert region.width is None
    assert region.height is None
    assert region.label is None
    assert region.detail_type is None
    assert region.confidence == 1.0  # 默认值


def test_missing_shape_field(bridge_module):
    """测试缺少 shape 字段"""
    data = {
        "x": 100.0,
        "y": 200.0,
    }
    with pytest.raises(ValidationError):
        bridge_module.JS9RegionRecord.from_json(data)


def test_missing_x_field(bridge_module):
    """测试缺少 x 字段"""
    data = {
        "shape": "box",
        "y": 200.0,
    }
    with pytest.raises(ValidationError):
        bridge_module.JS9RegionRecord.from_json(data)


def test_missing_y_field(bridge_module):
    """测试缺少 y 字段"""
    data = {
        "shape": "box",
        "x": 100.0,
    }
    with pytest.raises(ValidationError):
        bridge_module.JS9RegionRecord.from_json(data)


def test_invalid_shape_value(bridge_module):
    """测试无效的 shape 值（但这会被接受，因为 shape 是字符串类型）"""
    # shape 字段是字符串类型，不会自动验证枚举值
    # 实际验证会在 to_bbox 方法中进行
    data = {
        "shape": "invalid_shape",
        "x": 100.0,
        "y": 200.0,
    }
    region = bridge_module.JS9RegionRecord.from_json(data)
    assert region.shape == "invalid_shape"


def test_invalid_confidence_out_of_range(bridge_module):
    """测试超出范围的置信度值"""
    # confidence 字段是 float，pydantic 会验证类型但不会自动限制范围
    # 应该在业务逻辑中验证范围
    data = {
        "shape": "box",
        "x": 100.0,
        "y": 200.0,
        "confidence": 1.5,  # 超出 [0, 1] 范围
    }
    region = bridge_module.JS9RegionRecord.from_json(data)
    assert region.confidence == 1.5  # pydantic 不会自动限制范围


def test_invalid_confidence_type(bridge_module):
    """测试错误的置信度类型"""
    data = {
        "shape": "box",
        "x": 100.0,
        "y": 200.0,
        "confidence": "high",  # 应该是 float
    }
    with pytest.raises(ValidationError):
        bridge_module.JS9RegionRecord.from_json(data)


def test_invalid_coordinate_type(bridge_module):
    """测试错误的坐标类型"""
    data = {
        "shape": "box",
        "x": "100",  # 应该是 float
        "y": 200.0,
    }
    # pydantic 会尝试转换字符串为数字
    region = bridge_module.JS9RegionRecord.from_json(data)
    assert region.x == 100.0


def test_extra_fields_allowed(bridge_module):
    """测试允许额外字段"""
    data = {
        "shape": "box",
        "x": 100.0,
        "y": 200.0,
        "extra_field": "some_value",
    }
    region = bridge_module.JS9RegionRecord.from_json(data)
    # Pydantic v2 默认会忽略额外字段（配置 extra='ignore'）
    assert hasattr(region, "shape")
    # extra_field 不会被添加到模型实例
    assert not hasattr(region, "extra_field")


def test_negative_coordinates(bridge_module):
    """测试负坐标值"""
    data = {
        "shape": "box",
        "x": -50.0,
        "y": -30.0,
        "width": 100.0,
        "height": 80.0,
    }
    region = bridge_module.JS9RegionRecord.from_json(data)
    assert region.x == -50.0
    assert region.y == -30.0


def test_zero_dimensions(bridge_module):
    """测试零尺寸"""
    data = {
        "shape": "box",
        "x": 100.0,
        "y": 200.0,
        "width": 0.0,
        "height": 0.0,
    }
    region = bridge_module.JS9RegionRecord.from_json(data)
    assert region.width == 0.0
    assert region.height == 0.0


def test_very_large_coordinates(bridge_module):
    """测试非常大的坐标值"""
    data = {
        "shape": "box",
        "x": 999999.0,
        "y": 999999.0,
        "width": 1000.0,
        "height": 1000.0,
    }
    region = bridge_module.JS9RegionRecord.from_json(data)
    assert region.x == 999999.0
    assert region.y == 999999.0


def test_floating_point_coordinates(bridge_module):
    """测试浮点坐标精度"""
    data = {
        "shape": "box",
        "x": 123.456789,
        "y": 987.654321,
        "width": 45.123456,
        "height": 67.789012,
    }
    region = bridge_module.JS9RegionRecord.from_json(data)
    assert region.x == 123.456789
    assert region.y == 987.654321
    assert region.width == 45.123456
    assert region.height == 67.789012


def test_special_detail_types(bridge_module):
    """测试特殊 detail_type"""
    real_types = ["asteroid", "supernova", "variable_star"]
    bogus_types = ["satellite_trail", "noise", "diffraction_spike", "cmos_condensation", "corresponding"]

    for detail_type in real_types:
        data = {
            "shape": "box",
            "x": 100.0,
            "y": 200.0,
            "detail_type": detail_type,
        }
        region = bridge_module.JS9RegionRecord.from_json(data)
        assert region.detail_type == detail_type

    for detail_type in bogus_types:
        data = {
            "shape": "box",
            "x": 100.0,
            "y": 200.0,
            "detail_type": detail_type,
        }
        region = bridge_module.JS9RegionRecord.from_json(data)
        assert region.detail_type == detail_type


def test_empty_strings(bridge_module):
    """测试空字符串字段"""
    data = {
        "shape": "box",
        "x": 100.0,
        "y": 200.0,
        "label": "",
        "detail_type": "",
    }
    region = bridge_module.JS9RegionRecord.from_json(data)
    assert region.label == ""
    assert region.detail_type == ""
