"""FitsAnnotationBackend 单元测试

v2 FITS 全图检测标注后端: 加载 FITS 图像、边界框标注、JSON 持久化、撤销重做、统计、导出。
TDD — 此文件先于实现代码编写。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scann.core.annotation_models import (
    AnnotationSample,
    AnnotationStats,
    BBox,
    ExportResult,
)


# ─── Fixtures ───


@pytest.fixture
def fits_dataset(tmp_dir: Path) -> Path:
    """创建一个模拟 v2 FITS 标注数据集

    结构:
      fits_data/
        new/
          field_001.fits
          field_002.fits
          field_003.fits
        old/
          field_001.fits
          field_002.fits
          field_003.fits
        annotations.json  (可选，首次可为空)
    """
    try:
        from astropy.io import fits as astro_fits
    except ImportError:
        pytest.skip("astropy not installed")

    ds = tmp_dir / "fits_data"
    new_dir = ds / "new"
    old_dir = ds / "old"
    new_dir.mkdir(parents=True)
    old_dir.mkdir(parents=True)

    rng = np.random.default_rng(42)
    hdr = astro_fits.Header()
    hdr["OBJECT"] = "TestField"
    hdr["DATE-OBS"] = "2026-01-15T20:30:00"

    for i in range(3):
        data = rng.normal(loc=1000, scale=50, size=(256, 256)).astype(np.uint16)
        astro_fits.writeto(
            str(new_dir / f"field_{i + 1:03d}.fits"), data, header=hdr, overwrite=True
        )
        # 旧图 — 微偏移
        old_data = np.roll(data, shift=(2, -1), axis=(0, 1))
        astro_fits.writeto(
            str(old_dir / f"field_{i + 1:03d}.fits"), old_data, header=hdr, overwrite=True
        )

    return ds


@pytest.fixture
def fits_dataset_with_annotations(fits_dataset: Path) -> Path:
    """创建含预存 JSON 标注的 FITS 数据集"""
    annotations = {
        "version": "2.0",
        "images": [
            {
                "id": "field_001",
                "file_name": "field_001.fits",
                "annotations": [
                    {
                        "x": 100,
                        "y": 50,
                        "width": 24,
                        "height": 24,
                        "label": "real",
                        "detail_type": "asteroid",
                        "confidence": 1.0,
                    }
                ],
            }
        ],
    }
    ann_path = fits_dataset / "annotations.json"
    ann_path.write_text(json.dumps(annotations, indent=2), encoding="utf-8")
    return fits_dataset


@pytest.fixture
def fits_backend(fits_dataset: Path):
    """创建已加载数据集的 FitsAnnotationBackend"""
    from scann.core.fits_annotation_backend import FitsAnnotationBackend

    b = FitsAnnotationBackend()
    b.load_samples(str(fits_dataset))
    return b


@pytest.fixture
def fits_backend_with_annotations(fits_dataset_with_annotations: Path):
    """创建含预存标注的 FitsAnnotationBackend"""
    from scann.core.fits_annotation_backend import FitsAnnotationBackend

    b = FitsAnnotationBackend()
    b.load_samples(str(fits_dataset_with_annotations))
    return b


# ─── 加载测试 ───


class TestFitsLoad:
    def test_load_samples_auto_pairing(self, fits_dataset: Path):
        """自动配对 new/old 文件夹中的同名 FITS 文件"""
        from scann.core.fits_annotation_backend import FitsAnnotationBackend

        b = FitsAnnotationBackend()
        samples = b.load_samples(str(fits_dataset))
        assert len(samples) == 3

    def test_loaded_samples_have_ids(self, fits_backend):
        for s in fits_backend.samples:
            assert s.id  # 非空
            assert s.display_name.endswith(".fits")

    def test_load_with_existing_annotations(self, fits_backend_with_annotations):
        """加载时应合并 JSON 中已有的标注"""
        samples = fits_backend_with_annotations.samples
        # field_001 应已有 1 个标注框
        s1 = next(s for s in samples if "field_001" in s.display_name)
        assert len(s1.bboxes) == 1
        assert s1.bboxes[0].label == "real"
        assert s1.bboxes[0].detail_type == "asteroid"

    def test_load_empty_directory(self, tmp_dir: Path):
        from scann.core.fits_annotation_backend import FitsAnnotationBackend

        empty = tmp_dir / "empty_fits"
        empty.mkdir()
        b = FitsAnnotationBackend()
        samples = b.load_samples(str(empty))
        assert samples == []

    def test_load_nonexistent_path(self, tmp_dir: Path):
        from scann.core.fits_annotation_backend import FitsAnnotationBackend

        b = FitsAnnotationBackend()
        with pytest.raises(FileNotFoundError):
            b.load_samples(str(tmp_dir / "nonexistent"))

    def test_samples_sorted(self, fits_backend):
        names = [s.display_name for s in fits_backend.samples]
        assert names == sorted(names)


# ─── supports_bbox ───


class TestFitsSupportsBBox:
    def test_supports_bbox(self, fits_backend):
        assert fits_backend.supports_bbox() is True


# ─── 标注测试 (边界框) ───


class TestFitsAnnotation:
    def test_add_bbox_annotation(self, fits_backend):
        """添加一个带标签的边界框"""
        sample = fits_backend.samples[0]
        bbox = BBox(x=100, y=50, width=24, height=24)
        fits_backend.save_annotation(
            sample.id, "real", bbox=bbox, detail_type="asteroid"
        )

        updated = fits_backend.get_sample(sample.id)
        assert len(updated.bboxes) == 1
        assert updated.bboxes[0].x == 100
        assert updated.bboxes[0].detail_type == "asteroid"

    def test_add_multiple_bboxes(self, fits_backend):
        """同一张图添加多个标注框"""
        sample = fits_backend.samples[0]
        fits_backend.save_annotation(
            sample.id, "real",
            bbox=BBox(x=10, y=20, width=30, height=30),
            detail_type="asteroid",
        )
        fits_backend.save_annotation(
            sample.id, "bogus",
            bbox=BBox(x=200, y=100, width=20, height=20),
            detail_type="noise",
        )

        updated = fits_backend.get_sample(sample.id)
        assert len(updated.bboxes) == 2

    def test_classify_without_bbox(self, fits_backend):
        """v2 也支持纯分类 (不加边界框)"""
        sample = fits_backend.samples[0]
        fits_backend.save_annotation(sample.id, "bogus", detail_type="noise")

        updated = fits_backend.get_sample(sample.id)
        assert updated.label == "bogus"
        assert updated.detail_type == "noise"
        assert len(updated.bboxes) == 0


# ─── JSON 持久化测试 ───


class TestFitsPersistence:
    def test_save_creates_json(self, fits_backend, fits_dataset: Path):
        """标注后应自动保存 annotations.json"""
        sample = fits_backend.samples[0]
        fits_backend.save_annotation(
            sample.id, "real",
            bbox=BBox(x=50, y=60, width=24, height=24),
            detail_type="asteroid",
        )

        ann_file = fits_dataset / "annotations.json"
        assert ann_file.exists()

        data = json.loads(ann_file.read_text(encoding="utf-8"))
        assert "images" in data
        assert len(data["images"]) >= 1

    def test_reload_preserves_annotations(self, fits_backend, fits_dataset: Path):
        """保存后重新加载，标注信息不丢失"""
        from scann.core.fits_annotation_backend import FitsAnnotationBackend

        sample = fits_backend.samples[0]
        fits_backend.save_annotation(
            sample.id, "real",
            bbox=BBox(x=50, y=60, width=24, height=24),
            detail_type="asteroid",
        )

        # 重新加载
        b2 = FitsAnnotationBackend()
        b2.load_samples(str(fits_dataset))
        s = next(s for s in b2.samples if s.id == sample.id)
        assert len(s.bboxes) == 1
        assert s.bboxes[0].x == 50
        assert s.bboxes[0].detail_type == "asteroid"

    def test_json_format_compatible(self, fits_backend, fits_dataset: Path):
        """JSON 格式应包含 version 字段"""
        sample = fits_backend.samples[0]
        fits_backend.save_annotation(sample.id, "real", detail_type="asteroid")

        data = json.loads(
            (fits_dataset / "annotations.json").read_text(encoding="utf-8")
        )
        assert "version" in data
        assert data["version"] == "2.0"


# ─── 图像数据测试 ───


class TestFitsImageData:
    def test_get_image_data_returns_ndarray(self, fits_backend):
        sample = fits_backend.samples[0]
        img = fits_backend.get_image_data(sample)
        assert isinstance(img, np.ndarray)

    def test_get_image_data_shape(self, fits_backend):
        sample = fits_backend.samples[0]
        img = fits_backend.get_image_data(sample)
        assert img.shape == (256, 256)

    def test_get_display_info(self, fits_backend):
        sample = fits_backend.samples[0]
        info = fits_backend.get_display_info(sample)
        assert "file_name" in info
        assert info["has_new_image"] is True
        assert info["has_old_image"] is True

    def test_get_new_and_old_image(self, fits_backend):
        """v2 后端应能分别获取新图和旧图"""
        sample = fits_backend.samples[0]
        new_img = fits_backend.get_image_data(sample, image_type="new")
        old_img = fits_backend.get_image_data(sample, image_type="old")
        assert isinstance(new_img, np.ndarray)
        assert isinstance(old_img, np.ndarray)
        # 新旧图应不完全相同 (有偏移)
        assert not np.array_equal(new_img, old_img)


# ─── 撤销/重做测试 ───


class TestFitsUndoRedo:
    def test_undo_add_bbox(self, fits_backend):
        sample = fits_backend.samples[0]
        fits_backend.save_annotation(
            sample.id, "real",
            bbox=BBox(x=10, y=20, width=30, height=30),
            detail_type="asteroid",
        )
        assert len(sample.bboxes) == 1

        result = fits_backend.undo()
        assert result is True
        assert len(sample.bboxes) == 0

    def test_redo_add_bbox(self, fits_backend):
        sample = fits_backend.samples[0]
        fits_backend.save_annotation(
            sample.id, "real",
            bbox=BBox(x=10, y=20, width=30, height=30),
            detail_type="asteroid",
        )
        fits_backend.undo()
        assert len(sample.bboxes) == 0

        fits_backend.redo()
        assert len(sample.bboxes) == 1

    def test_undo_classify(self, fits_backend):
        sample = fits_backend.samples[0]
        fits_backend.save_annotation(sample.id, "bogus", detail_type="noise")
        assert sample.label == "bogus"

        fits_backend.undo()
        assert sample.label is None

    def test_can_undo_redo_flags(self, fits_backend):
        assert fits_backend.can_undo is False
        assert fits_backend.can_redo is False

        fits_backend.save_annotation(
            fits_backend.samples[0].id, "real",
            bbox=BBox(x=0, y=0, width=10, height=10),
        )
        assert fits_backend.can_undo is True


# ─── 统计测试 ───


class TestFitsStatistics:
    def test_initial_stats(self, fits_backend):
        stats = fits_backend.get_statistics()
        assert stats.total == 3
        assert stats.labeled == 0

    def test_stats_after_bbox_annotation(self, fits_backend):
        fits_backend.save_annotation(
            fits_backend.samples[0].id, "real",
            bbox=BBox(x=10, y=20, width=30, height=30),
            detail_type="asteroid",
        )
        fits_backend.save_annotation(
            fits_backend.samples[0].id, "bogus",
            bbox=BBox(x=200, y=100, width=20, height=20),
            detail_type="noise",
        )

        stats = fits_backend.get_statistics()
        # 有标注框的图算已标注
        assert stats.labeled >= 1

    def test_stats_with_preexisting_annotations(
        self, fits_backend_with_annotations
    ):
        stats = fits_backend_with_annotations.get_statistics()
        assert stats.labeled >= 1
        assert stats.label_counts.get("asteroid", 0) >= 1


# ─── 导出测试 ───


class TestFitsExport:
    def test_export_native_json(self, fits_backend, tmp_dir: Path):
        """原生格式导出 = JSON"""
        fits_backend.save_annotation(
            fits_backend.samples[0].id, "real",
            bbox=BBox(x=10, y=20, width=30, height=30),
            detail_type="asteroid",
        )

        out_dir = tmp_dir / "export_json"
        result = fits_backend.export_dataset(str(out_dir), format="native")
        assert result.success is True
        assert (Path(result.output_dir) / "annotations.json").exists()

    def test_export_csv(self, fits_backend, tmp_dir: Path):
        fits_backend.save_annotation(
            fits_backend.samples[0].id, "real",
            bbox=BBox(x=10, y=20, width=30, height=30),
            detail_type="asteroid",
        )

        out_dir = tmp_dir / "export_csv"
        result = fits_backend.export_dataset(str(out_dir), format="csv")
        assert result.success is True
        csv_path = Path(result.output_dir) / "annotations.csv"
        assert csv_path.exists()

    def test_export_with_val_split(self, fits_backend, tmp_dir: Path):
        for s in fits_backend.samples:
            fits_backend.save_annotation(
                s.id, "real",
                bbox=BBox(x=10, y=20, width=30, height=30),
                detail_type="asteroid",
            )

        out_dir = tmp_dir / "export_split"
        result = fits_backend.export_dataset(str(out_dir), val_split=0.3)
        assert result.success is True
        assert result.train_count + result.val_count == result.total_exported


# ─── 筛选测试 ───


class TestFitsFiltering:
    def test_filter_all(self, fits_backend):
        assert len(fits_backend.get_filtered_samples("all")) == 3

    def test_filter_unlabeled(self, fits_backend):
        fits_backend.save_annotation(
            fits_backend.samples[0].id, "real", detail_type="asteroid"
        )
        unlabeled = fits_backend.get_filtered_samples("unlabeled")
        assert len(unlabeled) == 2


# ─── FW_ 前缀文件名配对测试 ───


class TestFwPrefixPairing:
    """测试 old 目录下带 FW_ 前缀的文件与 new 目录下的文件配对"""

    @pytest.fixture
    def fw_dataset(self, tmp_dir: Path) -> Path:
        """创建 new 无前缀, old 有 FW_ 前缀的 FITS 数据集"""
        try:
            from astropy.io import fits as astro_fits
        except ImportError:
            pytest.skip("astropy not installed")

        ds = tmp_dir / "fw_data"
        new_dir = ds / "new"
        old_dir = ds / "old"
        new_dir.mkdir(parents=True)
        old_dir.mkdir(parents=True)

        rng = np.random.default_rng(99)
        hdr = astro_fits.Header()

        for name in ["star_001", "star_002", "star_003"]:
            data = rng.normal(loc=1000, scale=50, size=(64, 64)).astype(np.uint16)
            # new 目录: star_001.fits
            astro_fits.writeto(
                str(new_dir / f"{name}.fits"), data, header=hdr, overwrite=True
            )
            # old 目录: FW_star_001.fits (带 FW_ 前缀)
            old_data = np.roll(data, 1, axis=0)
            astro_fits.writeto(
                str(old_dir / f"FW_{name}.fits"), old_data, header=hdr, overwrite=True
            )

        return ds

    def test_fw_prefix_pairing(self, fw_dataset: Path):
        """FW_ 前缀的旧图应与同名新图配对"""
        from scann.core.fits_annotation_backend import FitsAnnotationBackend

        b = FitsAnnotationBackend()
        samples = b.load_samples(str(fw_dataset))
        # 应配对为 3 个样本，而非 6 个
        assert len(samples) == 3

    def test_fw_prefix_old_image_accessible(self, fw_dataset: Path):
        """配对后应能正确加载旧图数据"""
        from scann.core.fits_annotation_backend import FitsAnnotationBackend

        b = FitsAnnotationBackend()
        samples = b.load_samples(str(fw_dataset))
        sample = samples[0]

        # 加载新图
        new_data = b.get_image_data(sample, image_type="new")
        assert new_data is not None
        assert new_data.shape == (64, 64)

        # 加载旧图
        old_data = b.get_image_data(sample, image_type="old")
        assert old_data is not None
        assert old_data.shape == (64, 64)

    def test_exact_match_takes_priority(self, tmp_dir: Path):
        """精确名称匹配应优先于前缀去除匹配"""
        try:
            from astropy.io import fits as astro_fits
        except ImportError:
            pytest.skip("astropy not installed")

        ds = tmp_dir / "exact_data"
        new_dir = ds / "new"
        old_dir = ds / "old"
        new_dir.mkdir(parents=True)
        old_dir.mkdir(parents=True)

        hdr = astro_fits.Header()
        data = np.ones((32, 32), dtype=np.uint16) * 1000

        # new: img_001.fits
        astro_fits.writeto(str(new_dir / "img_001.fits"), data, header=hdr, overwrite=True)
        # old: img_001.fits (精确匹配) + FW_img_001.fits (前缀匹配)
        astro_fits.writeto(str(old_dir / "img_001.fits"), data, header=hdr, overwrite=True)
        astro_fits.writeto(str(old_dir / "FW_img_001.fits"), data, header=hdr, overwrite=True)

        from scann.core.fits_annotation_backend import FitsAnnotationBackend

        b = FitsAnnotationBackend()
        samples = b.load_samples(str(ds))

        # img_001 精确匹配 old/img_001, FW_img_001 为独立样本
        # 应有 2 个样本: img_001 + FW_img_001
        assert len(samples) == 2
