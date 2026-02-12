"""标注数据模型单元测试

测试 AnnotationSample, BBox, AnnotationStats, ExportResult 等数据类。
"""

from __future__ import annotations

import pytest

from scann.core.annotation_models import (
    AnnotationAction,
    AnnotationLabel,
    AnnotationSample,
    AnnotationStats,
    BBox,
    DetailType,
    DETAIL_TYPE_DISPLAY,
    DETAIL_TYPE_TO_LABEL,
    ExportResult,
    SHORTCUT_TO_DETAIL_TYPE,
)


# ─────────────────────── 枚举测试 ───────────────────────


class TestAnnotationLabel:
    def test_values(self):
        assert AnnotationLabel.REAL.value == "real"
        assert AnnotationLabel.BOGUS.value == "bogus"

    def test_str_enum(self):
        """AnnotationLabel 继承 str，可直接用作字符串"""
        assert AnnotationLabel.REAL == "real"
        assert AnnotationLabel.BOGUS == "bogus"


class TestDetailType:
    def test_real_types(self):
        assert DetailType.ASTEROID.value == "asteroid"
        assert DetailType.SUPERNOVA.value == "supernova"
        assert DetailType.VARIABLE_STAR.value == "variable_star"

    def test_bogus_types(self):
        assert DetailType.SATELLITE_TRAIL.value == "satellite_trail"
        assert DetailType.NOISE.value == "noise"
        assert DetailType.DIFFRACTION_SPIKE.value == "diffraction_spike"
        assert DetailType.CMOS_CONDENSATION.value == "cmos_condensation"
        assert DetailType.CORRESPONDING.value == "corresponding"

    def test_total_count(self):
        """确保有 3 个真类和 5 个假类"""
        real_types = [dt for dt, lbl in DETAIL_TYPE_TO_LABEL.items()
                      if lbl == AnnotationLabel.REAL]
        bogus_types = [dt for dt, lbl in DETAIL_TYPE_TO_LABEL.items()
                       if lbl == AnnotationLabel.BOGUS]
        assert len(real_types) == 3
        assert len(bogus_types) == 5

    def test_all_types_have_display_text(self):
        """所有 DetailType 都应有显示文本"""
        for dt in DetailType:
            assert dt in DETAIL_TYPE_DISPLAY

    def test_all_types_have_label_mapping(self):
        """所有 DetailType 都应有大类映射"""
        for dt in DetailType:
            assert dt in DETAIL_TYPE_TO_LABEL


class TestShortcutMapping:
    def test_y_shortcuts(self):
        assert SHORTCUT_TO_DETAIL_TYPE["Y1"] == DetailType.ASTEROID
        assert SHORTCUT_TO_DETAIL_TYPE["Y2"] == DetailType.SUPERNOVA
        assert SHORTCUT_TO_DETAIL_TYPE["Y3"] == DetailType.VARIABLE_STAR

    def test_n_shortcuts(self):
        assert SHORTCUT_TO_DETAIL_TYPE["N1"] == DetailType.SATELLITE_TRAIL
        assert SHORTCUT_TO_DETAIL_TYPE["N2"] == DetailType.NOISE
        assert SHORTCUT_TO_DETAIL_TYPE["N3"] == DetailType.DIFFRACTION_SPIKE
        assert SHORTCUT_TO_DETAIL_TYPE["N4"] == DetailType.CMOS_CONDENSATION
        assert SHORTCUT_TO_DETAIL_TYPE["N5"] == DetailType.CORRESPONDING

    def test_total_shortcuts(self):
        assert len(SHORTCUT_TO_DETAIL_TYPE) == 8


# ─────────────────────── BBox 测试 ───────────────────────


class TestBBox:
    def test_creation_with_defaults(self):
        bbox = BBox(x=10, y=20, width=30, height=40)
        assert bbox.x == 10
        assert bbox.y == 20
        assert bbox.width == 30
        assert bbox.height == 40
        assert bbox.label == "real"
        assert bbox.confidence == 1.0
        assert bbox.detail_type is None

    def test_creation_with_detail_type(self):
        bbox = BBox(x=0, y=0, width=24, height=24,
                    label="bogus", detail_type="noise", confidence=0.85)
        assert bbox.label == "bogus"
        assert bbox.detail_type == "noise"
        assert bbox.confidence == 0.85

    def test_center(self):
        bbox = BBox(x=10, y=20, width=30, height=40)
        assert bbox.center == (25, 40)

    def test_area(self):
        bbox = BBox(x=0, y=0, width=30, height=40)
        assert bbox.area == 1200

    def test_contains_inside(self):
        bbox = BBox(x=10, y=20, width=30, height=40)
        assert bbox.contains(15, 25) is True
        assert bbox.contains(10, 20) is True  # 左上角 (包含)

    def test_contains_outside(self):
        bbox = BBox(x=10, y=20, width=30, height=40)
        assert bbox.contains(5, 25) is False   # 左侧
        assert bbox.contains(50, 25) is False  # 右侧
        assert bbox.contains(15, 10) is False  # 上方
        assert bbox.contains(15, 70) is False  # 下方

    def test_contains_boundary(self):
        bbox = BBox(x=10, y=20, width=30, height=40)
        # 右边界 x=40 不包含 (半开区间)
        assert bbox.contains(40, 25) is False
        # 下边界 y=60 不包含
        assert bbox.contains(15, 60) is False

    def test_to_dict(self):
        bbox = BBox(x=10, y=20, width=30, height=40,
                    label="bogus", confidence=0.9, detail_type="noise")
        d = bbox.to_dict()
        assert d["x"] == 10
        assert d["y"] == 20
        assert d["width"] == 30
        assert d["height"] == 40
        assert d["label"] == "bogus"
        assert d["confidence"] == 0.9
        assert d["detail_type"] == "noise"

    def test_to_dict_without_detail_type(self):
        bbox = BBox(x=0, y=0, width=10, height=10)
        d = bbox.to_dict()
        assert "detail_type" not in d

    def test_from_dict(self):
        data = {"x": 5, "y": 10, "width": 20, "height": 25,
                "label": "real", "confidence": 0.95, "detail_type": "asteroid"}
        bbox = BBox.from_dict(data)
        assert bbox.x == 5
        assert bbox.y == 10
        assert bbox.width == 20
        assert bbox.height == 25
        assert bbox.label == "real"
        assert bbox.confidence == 0.95
        assert bbox.detail_type == "asteroid"

    def test_from_dict_defaults(self):
        """缺少可选字段时使用默认值"""
        data = {"x": 0, "y": 0, "width": 10, "height": 10}
        bbox = BBox.from_dict(data)
        assert bbox.label == "real"
        assert bbox.confidence == 1.0
        assert bbox.detail_type is None

    def test_roundtrip_serialization(self):
        """to_dict / from_dict 往返不丢失数据"""
        original = BBox(x=100, y=200, width=50, height=60,
                        label="bogus", confidence=0.77, detail_type="satellite_trail")
        restored = BBox.from_dict(original.to_dict())
        assert restored.x == original.x
        assert restored.y == original.y
        assert restored.width == original.width
        assert restored.height == original.height
        assert restored.label == original.label
        assert restored.confidence == original.confidence
        assert restored.detail_type == original.detail_type


# ─────────────────────── AnnotationSample 测试 ───────────────────────


class TestAnnotationSample:
    def test_creation_minimal(self):
        sample = AnnotationSample(id="001", source_path="/data/img.png",
                                  display_name="img.png")
        assert sample.id == "001"
        assert sample.source_path == "/data/img.png"
        assert sample.display_name == "img.png"
        assert sample.label is None
        assert sample.detail_type is None
        assert sample.bboxes == []
        assert sample.ai_suggestion is None
        assert sample.ai_confidence is None
        assert sample.metadata == {}

    def test_is_labeled(self):
        unlabeled = AnnotationSample(id="1", source_path="", display_name="")
        assert unlabeled.is_labeled is False

        labeled = AnnotationSample(id="2", source_path="", display_name="",
                                   label="real")
        assert labeled.is_labeled is True

    def test_label_display_unlabeled(self):
        s = AnnotationSample(id="1", source_path="", display_name="")
        assert s.label_display == "未标注"

    def test_label_display_real_no_detail(self):
        s = AnnotationSample(id="1", source_path="", display_name="",
                             label="real")
        assert s.label_display == "A.真"

    def test_label_display_bogus_no_detail(self):
        s = AnnotationSample(id="1", source_path="", display_name="",
                             label="bogus")
        assert s.label_display == "B.假"

    def test_label_display_with_detail_type(self):
        s = AnnotationSample(id="1", source_path="", display_name="",
                             label="real", detail_type="asteroid")
        assert "小行星" in s.label_display

    def test_label_display_with_unknown_detail(self):
        s = AnnotationSample(id="1", source_path="", display_name="",
                             label="real", detail_type="unknown_type")
        assert s.label_display == "unknown_type"

    def test_to_dict_minimal(self):
        s = AnnotationSample(id="1", source_path="/a.png", display_name="a.png")
        d = s.to_dict()
        assert d["id"] == "1"
        assert d["source_path"] == "/a.png"
        assert d["display_name"] == "a.png"
        # 未标注时不应包含 label 等可选字段
        assert "label" not in d
        assert "bboxes" not in d

    def test_to_dict_full(self):
        s = AnnotationSample(
            id="42", source_path="/img.fits", display_name="img.fits",
            label="bogus", detail_type="noise",
            bboxes=[BBox(x=0, y=0, width=10, height=10)],
            ai_suggestion="bogus", ai_confidence=0.88,
            metadata={"observer": "test"},
        )
        d = s.to_dict()
        assert d["label"] == "bogus"
        assert d["detail_type"] == "noise"
        assert len(d["bboxes"]) == 1
        assert d["ai_suggestion"] == "bogus"
        assert d["ai_confidence"] == 0.88
        assert d["metadata"]["observer"] == "test"

    def test_from_dict(self):
        data = {
            "id": "99",
            "source_path": "/test.png",
            "display_name": "test.png",
            "label": "real",
            "detail_type": "supernova",
            "bboxes": [{"x": 1, "y": 2, "width": 3, "height": 4}],
            "ai_suggestion": "real",
            "ai_confidence": 0.95,
            "metadata": {"key": "val"},
        }
        s = AnnotationSample.from_dict(data)
        assert s.id == "99"
        assert s.label == "real"
        assert s.detail_type == "supernova"
        assert len(s.bboxes) == 1
        assert s.bboxes[0].x == 1
        assert s.ai_suggestion == "real"
        assert s.ai_confidence == 0.95

    def test_roundtrip(self):
        original = AnnotationSample(
            id="A1", source_path="/data/x.fits", display_name="x.fits",
            label="real", detail_type="asteroid",
            bboxes=[BBox(10, 20, 30, 40, "real", 0.99, "asteroid")],
            ai_suggestion="real", ai_confidence=0.92,
            metadata={"note": "bright"},
        )
        restored = AnnotationSample.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.label == original.label
        assert restored.detail_type == original.detail_type
        assert len(restored.bboxes) == len(original.bboxes)
        assert restored.ai_suggestion == original.ai_suggestion


# ─────────────────────── AnnotationStats 测试 ───────────────────────


class TestAnnotationStats:
    def test_defaults(self):
        stats = AnnotationStats()
        assert stats.total == 0
        assert stats.labeled == 0
        assert stats.unlabeled == 0
        assert stats.progress_percent == 0.0
        assert stats.label_counts == {}

    def test_update_from_empty_samples(self):
        stats = AnnotationStats()
        stats.update_from_samples([])
        assert stats.total == 0
        assert stats.progress_percent == 0.0

    def test_update_from_samples(self):
        samples = [
            AnnotationSample(id="1", source_path="", display_name="",
                             label="real", detail_type="asteroid"),
            AnnotationSample(id="2", source_path="", display_name="",
                             label="bogus", detail_type="noise"),
            AnnotationSample(id="3", source_path="", display_name=""),
            AnnotationSample(id="4", source_path="", display_name="",
                             label="real", detail_type="asteroid"),
        ]
        stats = AnnotationStats()
        stats.update_from_samples(samples)
        assert stats.total == 4
        assert stats.labeled == 3
        assert stats.unlabeled == 1
        assert stats.progress_percent == 75.0
        assert stats.label_counts["asteroid"] == 2
        assert stats.label_counts["noise"] == 1

    def test_real_count(self):
        stats = AnnotationStats()
        stats.label_counts = {"asteroid": 10, "supernova": 3, "noise": 5}
        assert stats.real_count == 13

    def test_bogus_count(self):
        stats = AnnotationStats()
        stats.label_counts = {
            "satellite_trail": 5,
            "noise": 10,
            "diffraction_spike": 2,
            "cmos_condensation": 1,
            "corresponding": 3,
        }
        assert stats.bogus_count == 21

    def test_mixed_label_types(self):
        """当只有大类标签 (无 detail_type) 时也能正确计数"""
        samples = [
            AnnotationSample(id="1", source_path="", display_name="",
                             label="real"),
            AnnotationSample(id="2", source_path="", display_name="",
                             label="bogus"),
        ]
        stats = AnnotationStats()
        stats.update_from_samples(samples)
        assert stats.labeled == 2
        assert stats.real_count == 1
        assert stats.bogus_count == 1


# ─────────────────────── ExportResult 测试 ───────────────────────


class TestExportResult:
    def test_defaults(self):
        r = ExportResult()
        assert r.success is True
        assert r.output_dir == ""
        assert r.total_exported == 0
        assert r.train_count == 0
        assert r.val_count == 0
        assert r.format == "native"
        assert r.error_message == ""

    def test_failed_export(self):
        r = ExportResult(success=False, error_message="Permission denied")
        assert r.success is False
        assert r.error_message == "Permission denied"


# ─────────────────────── AnnotationAction 测试 ───────────────────────


class TestAnnotationAction:
    def test_creation(self):
        action = AnnotationAction(
            action_type="label",
            sample_id="001",
            old_value={"label": None},
            new_value={"label": "real", "detail_type": "asteroid"},
        )
        assert action.action_type == "label"
        assert action.sample_id == "001"
        assert action.old_value["label"] is None
        assert action.new_value["label"] == "real"
