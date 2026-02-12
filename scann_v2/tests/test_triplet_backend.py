"""TripletAnnotationBackend 单元测试

v1 三联图分类标注后端: 加载 PNG 三联图、分类标注、撤销重做、统计、导出。
TDD — 此文件先于实现代码编写。
"""

from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scann.core.annotation_models import AnnotationLabel, AnnotationSample, DetailType


# ─── Fixtures ───


@pytest.fixture
def triplet_dataset(tmp_dir: Path) -> Path:
    """创建一个模拟 v1 三联图数据集目录

    结构:
      dataset/
        positive/      (已标记为真)
        negative/      (已标记为假)
        unlabeled/     (待标注)
          img_001.png  (80×240 三联图)
          img_002.png
          img_003.png
          img_004.png
          img_005.png
    """
    ds = tmp_dir / "dataset"
    (ds / "positive").mkdir(parents=True)
    (ds / "negative").mkdir(parents=True)
    unlabeled = ds / "unlabeled"
    unlabeled.mkdir(parents=True)

    # 创建 5 张 80×240 三联图 (差异图 | 新图 | 参考图)
    rng = np.random.default_rng(42)
    for i in range(5):
        img_data = rng.integers(0, 255, size=(80, 240), dtype=np.uint8)
        img = Image.fromarray(img_data, mode="L")
        img.save(unlabeled / f"img_{i + 1:03d}.png")

    return ds


@pytest.fixture
def triplet_dataset_with_labeled(triplet_dataset: Path) -> Path:
    """创建含已标注样本的数据集 — 部分在 positive/negative 目录"""
    # 移动 2 张到 positive, 1 张到 negative
    unlabeled = triplet_dataset / "unlabeled"
    positive = triplet_dataset / "positive"
    negative = triplet_dataset / "negative"

    shutil.copy(unlabeled / "img_001.png", positive / "img_001.png")
    shutil.copy(unlabeled / "img_002.png", negative / "img_002.png")
    # 删除原始 unlabeled 中的已标注文件
    (unlabeled / "img_001.png").unlink()
    (unlabeled / "img_002.png").unlink()

    return triplet_dataset


@pytest.fixture
def backend(triplet_dataset: Path):
    """创建已加载数据集的 TripletAnnotationBackend"""
    from scann.core.triplet_backend import TripletAnnotationBackend

    b = TripletAnnotationBackend()
    b.load_samples(str(triplet_dataset))
    return b


# ─── 加载测试 ───


class TestTripletLoad:
    def test_load_unlabeled_samples(self, triplet_dataset: Path):
        from scann.core.triplet_backend import TripletAnnotationBackend

        b = TripletAnnotationBackend()
        samples = b.load_samples(str(triplet_dataset))
        assert len(samples) == 5
        for s in samples:
            assert s.label is None
            assert s.is_labeled is False
            assert s.display_name.endswith(".png")

    def test_load_with_existing_labels(self, triplet_dataset_with_labeled: Path):
        from scann.core.triplet_backend import TripletAnnotationBackend

        b = TripletAnnotationBackend()
        samples = b.load_samples(str(triplet_dataset_with_labeled))
        # 2 positive + 1 negative + 3 unlabeled = 6? No:
        # Original 5, moved 2 → positive has 2, negative has 1, unlabeled has 3
        # Total should be 5 (no duplication because originals deleted from unlabeled)
        assert len(samples) == 5

        labeled = [s for s in samples if s.is_labeled]
        assert len(labeled) == 2  # positive=1, negative=1
        # 验证标签正确
        labels = {s.display_name: s.label for s in labeled}
        assert labels["img_001.png"] == "real"
        assert labels["img_002.png"] == "bogus"

    def test_load_filter_unlabeled(self, triplet_dataset_with_labeled: Path):
        from scann.core.triplet_backend import TripletAnnotationBackend

        b = TripletAnnotationBackend()
        samples = b.load_samples(str(triplet_dataset_with_labeled), filter="unlabeled")
        assert all(s.label is None for s in samples)
        assert len(samples) == 3

    def test_load_empty_folder(self, tmp_dir: Path):
        from scann.core.triplet_backend import TripletAnnotationBackend

        empty = tmp_dir / "empty_ds"
        empty.mkdir()
        b = TripletAnnotationBackend()
        samples = b.load_samples(str(empty))
        assert samples == []

    def test_load_nonexistent_path_raises(self, tmp_dir: Path):
        from scann.core.triplet_backend import TripletAnnotationBackend

        b = TripletAnnotationBackend()
        with pytest.raises(FileNotFoundError):
            b.load_samples(str(tmp_dir / "nonexistent"))

    def test_samples_sorted_by_name(self, triplet_dataset: Path):
        from scann.core.triplet_backend import TripletAnnotationBackend

        b = TripletAnnotationBackend()
        samples = b.load_samples(str(triplet_dataset))
        names = [s.display_name for s in samples]
        assert names == sorted(names)


# ─── 标注测试 ───


class TestTripletAnnotation:
    def test_save_annotation_real(self, backend):
        sample = backend.samples[0]
        backend.save_annotation(sample.id, "real", detail_type="asteroid")

        updated = backend.get_sample(sample.id)
        assert updated.label == "real"
        assert updated.detail_type == "asteroid"

    def test_save_annotation_bogus(self, backend):
        sample = backend.samples[1]
        backend.save_annotation(sample.id, "bogus", detail_type="noise")

        updated = backend.get_sample(sample.id)
        assert updated.label == "bogus"
        assert updated.detail_type == "noise"

    def test_save_annotation_moves_file(self, backend, triplet_dataset: Path):
        """标注为 real 后文件应出现在 positive/ 目录"""
        sample = backend.samples[0]
        original_path = Path(sample.source_path)
        assert original_path.exists()

        backend.save_annotation(sample.id, "real", detail_type="asteroid")

        # 文件应移动到 positive/
        expected = triplet_dataset / "positive" / original_path.name
        assert expected.exists()

    def test_save_annotation_bogus_moves_to_negative(self, backend, triplet_dataset: Path):
        """标注为 bogus 后文件应出现在 negative/ 目录"""
        sample = backend.samples[0]
        backend.save_annotation(sample.id, "bogus", detail_type="satellite_trail")

        expected = triplet_dataset / "negative" / Path(sample.source_path).name
        assert expected.exists()

    def test_relabel_moves_between_folders(self, backend, triplet_dataset: Path):
        """先标 real 再改标 bogus，文件应从 positive 移到 negative"""
        sample = backend.samples[0]
        backend.save_annotation(sample.id, "real", detail_type="asteroid")
        assert (triplet_dataset / "positive" / Path(sample.source_path).name).exists()

        backend.save_annotation(sample.id, "bogus", detail_type="noise")
        assert (triplet_dataset / "negative" / Path(sample.source_path).name).exists()
        # positive 中应不再有该文件
        assert not (triplet_dataset / "positive" / Path(sample.source_path).name).exists()

    def test_save_annotation_nonexistent_sample(self, backend):
        """标注不存在的样本不应报错 (静默忽略)"""
        backend.save_annotation("nonexistent_id", "real")  # 不抛异常


# ─── supports_bbox ───


class TestTripletSupportsBBox:
    def test_not_support_bbox(self, backend):
        assert backend.supports_bbox() is False


# ─── 图像数据测试 ───


class TestTripletImageData:
    def test_get_image_data_returns_pil_image(self, backend):
        sample = backend.samples[0]
        img = backend.get_image_data(sample)
        assert isinstance(img, Image.Image)

    def test_get_image_data_correct_size(self, backend):
        sample = backend.samples[0]
        img = backend.get_image_data(sample)
        assert img.size == (240, 80)  # width × height (PIL)

    def test_get_display_info(self, backend):
        sample = backend.samples[0]
        info = backend.get_display_info(sample)
        assert "file_name" in info
        assert info["file_name"].endswith(".png")
        assert info["has_new_image"] is False  # v1 不区分新旧图
        assert info["has_old_image"] is False


# ─── 撤销/重做测试 ───


class TestTripletUndoRedo:
    def test_undo_label(self, backend):
        sample = backend.samples[0]
        backend.save_annotation(sample.id, "real", detail_type="asteroid")
        assert sample.label == "real"

        result = backend.undo()
        assert result is True
        assert sample.label is None
        assert sample.detail_type is None

    def test_redo_label(self, backend):
        sample = backend.samples[0]
        backend.save_annotation(sample.id, "real", detail_type="asteroid")
        backend.undo()
        assert sample.label is None

        result = backend.redo()
        assert result is True
        assert sample.label == "real"
        assert sample.detail_type == "asteroid"

    def test_undo_empty_stack(self, backend):
        assert backend.undo() is False

    def test_redo_empty_stack(self, backend):
        assert backend.redo() is False

    def test_can_undo_redo_flags(self, backend):
        assert backend.can_undo is False
        assert backend.can_redo is False

        backend.save_annotation(backend.samples[0].id, "real")
        assert backend.can_undo is True
        assert backend.can_redo is False

        backend.undo()
        assert backend.can_undo is False
        assert backend.can_redo is True

    def test_new_action_clears_redo(self, backend):
        s0, s1 = backend.samples[0], backend.samples[1]
        backend.save_annotation(s0.id, "real")
        backend.undo()
        assert backend.can_redo is True

        backend.save_annotation(s1.id, "bogus")
        assert backend.can_redo is False

    def test_undo_restores_file_position(self, backend, triplet_dataset: Path):
        """撤销后文件应恢复原位"""
        sample = backend.samples[0]
        original_path = sample.source_path
        backend.save_annotation(sample.id, "real", detail_type="asteroid")
        backend.undo()

        # 文件应回到 unlabeled (原始位置)
        assert Path(original_path).exists()


# ─── 统计测试 ───


class TestTripletStatistics:
    def test_initial_stats(self, backend):
        stats = backend.get_statistics()
        assert stats.total == 5
        assert stats.labeled == 0
        assert stats.unlabeled == 5
        assert stats.progress_percent == 0.0

    def test_stats_after_labeling(self, backend):
        backend.save_annotation(backend.samples[0].id, "real", detail_type="asteroid")
        backend.save_annotation(backend.samples[1].id, "bogus", detail_type="noise")

        stats = backend.get_statistics()
        assert stats.total == 5
        assert stats.labeled == 2
        assert stats.unlabeled == 3
        assert stats.progress_percent == pytest.approx(40.0)
        assert stats.label_counts.get("asteroid", 0) == 1
        assert stats.label_counts.get("noise", 0) == 1


# ─── 筛选测试 ───


class TestTripletFiltering:
    def test_filter_all(self, backend):
        assert len(backend.get_filtered_samples("all")) == 5

    def test_filter_unlabeled(self, backend):
        backend.save_annotation(backend.samples[0].id, "real")
        unlabeled = backend.get_filtered_samples("unlabeled")
        assert len(unlabeled) == 4

    def test_filter_real(self, backend):
        backend.save_annotation(backend.samples[0].id, "real")
        backend.save_annotation(backend.samples[1].id, "bogus")
        real = backend.get_filtered_samples("real")
        assert len(real) == 1

    def test_filter_bogus(self, backend):
        backend.save_annotation(backend.samples[0].id, "bogus")
        bogus = backend.get_filtered_samples("bogus")
        assert len(bogus) == 1


# ─── 导出测试 ───


class TestTripletExport:
    def test_export_native(self, backend, tmp_dir: Path):
        """导出原生格式 = 文件夹分类 (positive/negative)"""
        backend.save_annotation(backend.samples[0].id, "real", detail_type="asteroid")
        backend.save_annotation(backend.samples[1].id, "bogus", detail_type="noise")

        out_dir = tmp_dir / "export_native"
        result = backend.export_dataset(str(out_dir), format="native")

        assert result.success is True
        assert result.total_exported == 2
        assert (Path(result.output_dir) / "positive").is_dir()
        assert (Path(result.output_dir) / "negative").is_dir()

    def test_export_csv(self, backend, tmp_dir: Path):
        """导出 CSV 格式: 文件路径 + 标签 + 详细类型"""
        backend.save_annotation(backend.samples[0].id, "real", detail_type="asteroid")
        backend.save_annotation(backend.samples[1].id, "bogus", detail_type="noise")

        out_dir = tmp_dir / "export_csv"
        result = backend.export_dataset(str(out_dir), format="csv")

        assert result.success is True
        csv_file = Path(result.output_dir) / "annotations.csv"
        assert csv_file.exists()

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert "file_path" in rows[0]
        assert "label" in rows[0]
        assert "detail_type" in rows[0]

    def test_export_only_labeled(self, backend, tmp_dir: Path):
        """默认只导出已标注样本"""
        backend.save_annotation(backend.samples[0].id, "real")
        out_dir = tmp_dir / "export_labeled"
        result = backend.export_dataset(str(out_dir))
        assert result.total_exported == 1

    def test_export_include_unlabeled(self, backend, tmp_dir: Path):
        """include_unlabeled=True 时导出所有样本"""
        backend.save_annotation(backend.samples[0].id, "real")
        out_dir = tmp_dir / "export_all"
        result = backend.export_dataset(str(out_dir), include_unlabeled=True)
        assert result.total_exported == 5

    def test_export_with_val_split(self, backend, tmp_dir: Path):
        """验证集拆分"""
        for i, s in enumerate(backend.samples):
            backend.save_annotation(s.id, "real" if i % 2 == 0 else "bogus")

        out_dir = tmp_dir / "export_split"
        result = backend.export_dataset(str(out_dir), val_split=0.2)
        assert result.success is True
        assert result.train_count + result.val_count == result.total_exported
        assert result.val_count >= 1  # 至少有 1 个验证样本
