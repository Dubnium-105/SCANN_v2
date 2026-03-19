"""JS9 Region 到 BBox 转换测试

测试 js9_region_to_bbox 函数和各种形状的转换逻辑
"""

import pytest


def test_box_basic_conversion(bridge_module):
    """测试基本的 box 转换"""
    region_data = {
        "shape": "box",
        "x": 100.5,
        "y": 200.3,
        "width": 50.0,
        "height": 60.0,
        "label": "real",
        "detail_type": "asteroid",
        "confidence": 0.95,
    }
    bbox = bridge_module.js9_region_to_bbox(region_data)
    assert bbox["x"] == 100.5
    assert bbox["y"] == 200.3
    assert bbox["width"] == 50.0
    assert bbox["height"] == 60.0
    assert bbox["label"] == "real"
    assert bbox["detail_type"] == "asteroid"
    assert bbox["confidence"] == 0.95


def test_box_with_image_bounds(bridge_module):
    """测试带图像边界的 box 转换"""
    region_data = {
        "shape": "box",
        "x": 950.0,
        "y": 1950.0,
        "width": 100.0,
        "height": 100.0,
    }
    bbox = bridge_module.js9_region_to_bbox(region_data, image_width=1000, image_height=2000)
    # 在边界内，应该保留原值
    assert bbox["x"] == 950.0
    assert bbox["y"] == 1950.0
    assert bbox["width"] == 50.0  # 裁剪到边界
    assert bbox["height"] == 50.0  # 裁剪到边界


def test_box_clipped_to_bounds(bridge_module):
    """测试超出边界的 box 被裁剪"""
    region_data = {
        "shape": "box",
        "x": -10.0,
        "y": -20.0,
        "width": 100.0,
        "height": 100.0,
    }
    bbox = bridge_module.js9_region_to_bbox(region_data, image_width=500, image_height=500)
    # 负坐标被裁剪到 0
    assert bbox["x"] == 0.0
    assert bbox["y"] == 0.0
    assert bbox["width"] == 100.0  # 在边界内
    assert bbox["height"] == 100.0  # 在边界内


def test_box_outside_image(bridge_module):
    """测试完全在图像外的 box"""
    region_data = {
        "shape": "box",
        "x": 1000.0,
        "y": 1000.0,
        "width": 100.0,
        "height": 100.0,
    }
    bbox = bridge_module.js9_region_to_bbox(region_data, image_width=500, image_height=500)
    # 超出边界，被裁剪到图像边缘
    assert bbox["x"] == 500.0  # max(0, min(1000, 500))
    assert bbox["y"] == 500.0
    assert bbox["width"] == 0.0  # 超出边界
    assert bbox["height"] == 0.0


def test_box_zero_dimensions(bridge_module):
    """测试零尺寸的 box"""
    region_data = {
        "shape": "box",
        "x": 100.0,
        "y": 200.0,
        "width": 0.0,
        "height": 0.0,
    }
    bbox = bridge_module.js9_region_to_bbox(region_data)
    assert bbox["x"] == 100.0
    assert bbox["y"] == 200.0
    assert bbox["width"] == 0.0
    assert bbox["height"] == 0.0


def test_circle_basic_conversion(bridge_module):
    """测试基本的 circle 转换"""
    region_data = {
        "shape": "circle",
        "x": 300.0,
        "y": 400.0,
        "radius": 25.5,
        "label": "bogus",
        "detail_type": "noise",
        "confidence": 0.8,
    }
    bbox = bridge_module.js9_region_to_bbox(region_data)
    # circle 转换为: (x-radius, y-radius, radius*2, radius*2)
    assert bbox["x"] == 274.5  # 300 - 25.5
    assert bbox["y"] == 374.5  # 400 - 25.5
    assert bbox["width"] == 51.0  # 25.5 * 2
    assert bbox["height"] == 51.0  # 25.5 * 2
    assert bbox["label"] == "bogus"
    assert bbox["detail_type"] == "noise"


def test_circle_clipped_to_bounds(bridge_module):
    """测试超出边界的 circle 被裁剪"""
    region_data = {
        "shape": "circle",
        "x": 10.0,
        "y": 10.0,
        "radius": 50.0,
    }
    bbox = bridge_module.js9_region_to_bbox(region_data, image_width=500, image_height=500)
    # 圆心在 (10,10)，半径 50
    # bbox 应该是 (-40, -40, 100, 100)，然后裁剪到 (0, 0, 100, 100)
    assert bbox["x"] == 0.0  # 负值被裁剪
    assert bbox["y"] == 0.0  # 负值被裁剪
    assert bbox["width"] == 100.0  # 500 - 0
    assert bbox["height"] == 100.0  # 500 - 0


def test_circle_zero_radius(bridge_module):
    """测试零半径的 circle"""
    region_data = {
        "shape": "circle",
        "x": 100.0,
        "y": 200.0,
        "radius": 0.0,
    }
    bbox = bridge_module.js9_region_to_bbox(region_data)
    assert bbox["x"] == 100.0
    assert bbox["y"] == 200.0
    assert bbox["width"] == 0.0
    assert bbox["height"] == 0.0


def test_polygon_basic_conversion(bridge_module):
    """测试基本的 polygon 转换"""
    region_data = {
        "shape": "polygon",
        "x": 0,
        "y": 0,
        "width": 0,
        "height": 0,
        "vertices": [[50, 50], [150, 30], [180, 150], [40, 160]],
    }
    bbox = bridge_module.js9_region_to_bbox(region_data)
    # 计算边界框
    assert bbox["x"] == 40.0  # min x
    assert bbox["y"] == 30.0  # min y
    assert bbox["width"] == 140.0  # max x - min x = 180 - 40
    assert bbox["height"] == 130.0  # max y - min y = 160 - 30


def test_polygon_clipped_to_bounds(bridge_module):
    """测试超出边界的 polygon 被裁剪"""
    region_data = {
        "shape": "polygon",
        "x": 0,
        "y": 0,
        "width": 0,
        "height": 0,
        "vertices": [[-10, -10], [20, 10], [10, 20]],
    }
    bbox = bridge_module.js9_region_to_bbox(region_data, image_width=500, image_height=500)
    # 边界框应该被裁剪到非负值
    # 顶点：x=[-10,20,10], y=[-10,10,20]
    # 原始边界框：x=-10, y=-10, width=30, height=30
    # 裁剪后：x=0, y=0, width=30, height=30
    assert bbox["x"] == 0.0
    assert bbox["y"] == 0.0
    assert bbox["width"] == 30.0  # max_x - 0 = 20 - (-10) = 30
    assert bbox["height"] == 30.0  # max_y - 0 = 20 - (-10) = 30


def test_polygon_fallback_to_box(bridge_module):
    """测试没有顶点信息的 polygon 降级为 box"""
    region_data = {
        "shape": "polygon",
        "x": 100.0,
        "y": 200.0,
        "width": 50.0,
        "height": 60.0,
    }
    bbox = bridge_module.js9_region_to_bbox(region_data)
    # 没有 vertices，使用 x, y, width, height
    assert bbox["x"] == 100.0
    assert bbox["y"] == 200.0
    assert bbox["width"] == 50.0
    assert bbox["height"] == 60.0


def test_polygon_empty_vertices(bridge_module):
    """测试空顶点列表的 polygon"""
    region_data = {
        "shape": "polygon",
        "x": 100.0,
        "y": 200.0,
        "width": 30.0,
        "height": 40.0,
        "vertices": [],
    }
    bbox = bridge_module.js9_region_to_bbox(region_data)
    # 空顶点，降级使用 x, y, width, height
    assert bbox["x"] == 100.0
    assert bbox["y"] == 200.0
    assert bbox["width"] == 30.0
    assert bbox["height"] == 40.0


def test_invalid_region_data(bridge_module):
    """测试无效的 region 数据"""
    region_data = {
        "shape": "box",
        # 缺少必需字段
    }
    with pytest.raises(ValueError, match="Invalid JS9 region data"):
        bridge_module.js9_region_to_bbox(region_data)


def test_unknown_shape_fallback(bridge_module):
    """测试未知形状降级为 box"""
    region_data = {
        "shape": "unknown_shape",
        "x": 100.0,
        "y": 200.0,
        "width": 50.0,
        "height": 60.0,
    }
    bbox = bridge_module.js9_region_to_bbox(region_data)
    # 未知形状应该降级为 box 行为
    assert bbox["x"] == 100.0
    assert bbox["y"] == 200.0
    assert bbox["width"] == 50.0
    assert bbox["height"] == 60.0


def test_default_confidence(bridge_module):
    """测试默认置信度"""
    region_data = {
        "shape": "box",
        "x": 100.0,
        "y": 200.0,
        "width": 50.0,
        "height": 60.0,
    }
    bbox = bridge_module.js9_region_to_bbox(region_data)
    assert bbox["confidence"] == 1.0  # 默认值


def test_none_label_and_detail_type(bridge_module):
    """测试 None 值的 label 和 detail_type"""
    region_data = {
        "shape": "box",
        "x": 100.0,
        "y": 200.0,
        "width": 50.0,
        "height": 60.0,
    }
    bbox = bridge_module.js9_region_to_bbox(region_data)
    assert bbox["label"] is None
    assert bbox["detail_type"] is None


def test_very_small_bbox(bridge_module):
    """测试非常小的 bbox"""
    region_data = {
        "shape": "box",
        "x": 100.5,
        "y": 200.5,
        "width": 0.1,
        "height": 0.1,
    }
    bbox = bridge_module.js9_region_to_bbox(region_data)
    assert bbox["x"] == 100.5
    assert bbox["y"] == 200.5
    assert bbox["width"] == 0.1
    assert bbox["height"] == 0.1


def test_negative_width_height(bridge_module):
    """测试负宽度或高度（会被裁剪为 0）"""
    region_data = {
        "shape": "box",
        "x": 100.0,
        "y": 200.0,
        "width": -50.0,
        "height": -60.0,
    }
    bbox = bridge_module.js9_region_to_bbox(region_data)
    # 负值会被裁剪为 0
    assert bbox["width"] == 0.0
    assert bbox["height"] == 0.0


def test_partial_out_of_bounds(bridge_module):
    """测试部分超出边界"""
    region_data = {
        "shape": "box",
        "x": 900.0,
        "y": 900.0,
        "width": 200.0,
        "height": 200.0,
    }
    bbox = bridge_module.js9_region_to_bbox(region_data, image_width=1000, image_height=1000)
    # 部分超出，被裁剪到边界
    assert bbox["x"] == 900.0
    assert bbox["y"] == 900.0
    assert bbox["width"] == 100.0  # 1000 - 900
    assert bbox["height"] == 100.0  # 1000 - 900


def test_multiple_regions_conversion(bridge_module):
    """测试多个 region 的转换"""
    regions = [
        {
            "shape": "box",
            "x": 100.0,
            "y": 200.0,
            "width": 50.0,
            "height": 60.0,
            "label": "real",
            "detail_type": "asteroid",
        },
        {
            "shape": "circle",
            "x": 300.0,
            "y": 400.0,
            "radius": 25.0,
            "label": "bogus",
            "detail_type": "noise",
        },
    ]
    
    bboxes = [bridge_module.js9_region_to_bbox(r) for r in regions]
    
    assert len(bboxes) == 2
    assert bboxes[0]["label"] == "real"
    assert bboxes[1]["label"] == "bogus"


def test_standard_hd_image(bridge_module):
    """测试标准高清图像（1920x1080）"""
    region_data = {
        "shape": "box",
        "x": 960.0,
        "y": 540.0,
        "width": 200.0,
        "height": 150.0,
    }
    bbox = bridge_module.js9_region_to_bbox(region_data, image_width=1920, image_height=1080)
    assert bbox["x"] == 960.0
    assert bbox["y"] == 540.0
    assert bbox["width"] == 200.0
    assert bbox["height"] == 150.0


def test_fits_image(bridge_module):
    """测试 FITS 图像（通常较大，如 4096x4096）"""
    region_data = {
        "shape": "box",
        "x": 2048.0,
        "y": 2048.0,
        "width": 500.0,
        "height": 500.0,
    }
    bbox = bridge_module.js9_region_to_bbox(region_data, image_width=4096, image_height=4096)
    assert bbox["x"] == 2048.0
    assert bbox["y"] == 2048.0
    assert bbox["width"] == 500.0
    assert bbox["height"] == 500.0


def test_unlimited_bounds(bridge_module):
    """测试无边界限制（默认值）"""
    region_data = {
        "shape": "box",
        "x": -1000.0,
        "y": -1000.0,
        "width": 5000.0,
        "height": 5000.0,
    }
    bbox = bridge_module.js9_region_to_bbox(region_data)  # 使用默认的 image_width/height
    # 使用默认的大值，不会被完全裁剪
    assert bbox["x"] == 0.0  # 负值被裁剪
    assert bbox["y"] == 0.0  # 负值被裁剪
    assert bbox["width"] == 5000.0  # 在默认边界内
    assert bbox["height"] == 5000.0  # 在默认边界内
