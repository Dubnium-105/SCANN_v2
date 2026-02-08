"""NMS 和 IoU 计算测试"""

import pytest
from scann.ai.inference import InferenceEngine
from scann.core.models import Detection, MarkerType


class TestNMS:
    """测试非极大值抑制"""

    def test_nms_empty_list(self):
        """测试：空列表"""
        engine = InferenceEngine.__new__(InferenceEngine)
        result = engine._nms([], 0.5)
        assert result == []

    def test_nms_single_detection(self):
        """测试：单个检测"""
        engine = InferenceEngine.__new__(InferenceEngine)
        detections = [
            Detection(
                x=100, y=100, confidence=0.9,
                width=200, height=200, marker_type=MarkerType.BOUNDING_BOX
            )
        ]
        result = engine._nms(detections, 0.5)
        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_nms_overlapping_boxes(self):
        """测试：重叠的边界框"""
        engine = InferenceEngine.__new__(InferenceEngine)

        # 创建重叠的检测框
        detections = [
            Detection(x=100, y=100, confidence=0.95, width=200, height=200, marker_type=MarkerType.BOUNDING_BOX),
            Detection(x=105, y=105, confidence=0.85, width=200, height=200, marker_type=MarkerType.BOUNDING_BOX),
            Detection(x=110, y=110, confidence=0.80, width=200, height=200, marker_type=MarkerType.BOUNDING_BOX),
        ]

        result = engine._nms(detections, 0.5)

        # 应该只保留一个（置信度最高的）
        assert len(result) == 1
        assert result[0].confidence == 0.95

    def test_nms_non_overlapping_boxes(self):
        """测试：不重叠的边界框"""
        engine = InferenceEngine.__new__(InferenceEngine)

        detections = [
            Detection(x=50, y=50, confidence=0.9, width=100, height=100, marker_type=MarkerType.BOUNDING_BOX),
            Detection(x=150, y=150, confidence=0.85, width=100, height=100, marker_type=MarkerType.BOUNDING_BOX),
            Detection(x=250, y=250, confidence=0.8, width=100, height=100, marker_type=MarkerType.BOUNDING_BOX),
        ]

        result = engine._nms(detections, 0.5)

        # 应该保留所有检测
        assert len(result) == 3

    def test_nms_iou_threshold_impact(self):
        """测试：IoU 阈值影响"""
        engine = InferenceEngine.__new__(InferenceEngine)

        # 创建重叠程度较小的检测框
        # 检测1：[0, 0, 100, 100]，面积 10000
        # 检测2：[80, 80, 180, 180]，面积 10000
        # 交集：[80, 80, 100, 100]，面积 20x20 = 400
        # IoU = 400 / (10000 + 10000 - 400) = 400 / 19600 ≈ 0.02
        detections = [
            Detection(x=50, y=50, confidence=0.95, width=100, height=100, marker_type=MarkerType.BOUNDING_BOX),
            Detection(x=130, y=130, confidence=0.85, width=100, height=100, marker_type=MarkerType.BOUNDING_BOX),
        ]

        # IoU 约 0.02，使用不同的阈值测试

        # 阈值 0.01：IoU (0.02) > 阈值，应该合并为 1 个
        result_low = engine._nms(detections, 0.01)
        assert len(result_low) == 1

        # 阈值 0.05：IoU (0.02) < 阈值，应该保留 2 个
        result_high = engine._nms(detections, 0.05)
        assert len(result_high) == 2


class TestIoUCalculation:
    """测试 IoU 计算"""

    def test_iou_identical_boxes(self):
        """测试：相同的边界框"""
        engine = InferenceEngine.__new__(InferenceEngine)

        bbox1 = [0, 0, 100, 100]
        bbox2 = [0, 0, 100, 100]

        iou = engine._calculate_iou(bbox1, bbox2)
        assert iou == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        """测试：不重叠的边界框"""
        engine = InferenceEngine.__new__(InferenceEngine)

        bbox1 = [0, 0, 100, 100]
        bbox2 = [200, 200, 300, 300]

        iou = engine._calculate_iou(bbox1, bbox2)
        assert iou == 0.0

    def test_iou_partial_overlap(self):
        """测试：部分重叠"""
        engine = InferenceEngine.__new__(InferenceEngine)

        # 两个 100x100 的框，重叠区域为 50x50
        bbox1 = [0, 0, 100, 100]
        bbox2 = [50, 50, 150, 150]

        # 交集：50x50 = 2500
        # 并集：10000 + 10000 - 2500 = 17500
        # IoU = 2500 / 17500 ≈ 0.1429
        iou = engine._calculate_iou(bbox1, bbox2)
        assert iou == pytest.approx(0.142857, abs=0.001)

    def test_iou_one_inside_another(self):
        """测试：一个框在另一个框内部"""
        engine = InferenceEngine.__new__(InferenceEngine)

        bbox1 = [0, 0, 100, 100]
        bbox2 = [25, 25, 75, 75]

        # bbox2 完全在 bbox1 内部
        # 交集：50x50 = 2500
        # 并集：10000
        # IoU = 2500 / 10000 = 0.25
        iou = engine._calculate_iou(bbox1, bbox2)
        assert iou == pytest.approx(0.25)

    def test_iou_touching_edges(self):
        """测试：边缘接触的框"""
        engine = InferenceEngine.__new__(InferenceEngine)

        bbox1 = [0, 0, 100, 100]
        bbox2 = [100, 0, 200, 100]

        # 只在边缘接触，没有重叠面积
        iou = engine._calculate_iou(bbox1, bbox2)
        assert iou == 0.0

    def test_iou_negative_coordinates(self):
        """测试：负坐标的边界框"""
        engine = InferenceEngine.__new__(InferenceEngine)

        bbox1 = [-50, -50, 50, 50]
        bbox2 = [0, 0, 100, 100]

        # 交集：50x50 = 2500
        # 并集：10000 + 10000 - 2500 = 17500
        iou = engine._calculate_iou(bbox1, bbox2)
        assert iou == pytest.approx(0.142857, abs=0.001)

    def test_iou_line_overlap(self):
        """测试：只有一条边重叠"""
        engine = InferenceEngine.__new__(InferenceEngine)

        bbox1 = [0, 0, 100, 100]
        bbox2 = [0, 100, 100, 200]

        # 只有边接触，没有面积重叠
        iou = engine._calculate_iou(bbox1, bbox2)
        assert iou == 0.0

