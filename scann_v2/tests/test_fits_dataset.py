"""FITS Detection Dataset 测试

使用测试驱动开发 (TDD) 实现：
1. FITS 格式的标注数据集
2. 支持滑动窗口提取 patches
3. 标注格式解析
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from scann.ai.dataset import FitsDetectionDataset


class TestFitsDetectionDataset:
    """测试 FITS 检测数据集"""

    def test_initialization_with_json_annotations(self, tmp_path):
        """测试：使用 JSON 格式的标注文件初始化"""
        # 准备
        # 创建模拟 FITS 文件
        fits_file = tmp_path / "test.fits"
        fits_file.write_bytes(b"mock fits data")

        # 创建标注文件
        annotations = {
            "images": [
                {
                    "file": "test.fits",
                    "width": 1000,
                    "height": 1000,
                    "annotations": [
                        {"x": 100, "y": 100, "width": 20, "height": 20, "label": "real"},
                        {"x": 500, "y": 500, "width": 15, "height": 15, "label": "bogus"},
                    ]
                }
            ]
        }
        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations))

        # 执行
        dataset = FitsDetectionDataset(
            image_dir=str(tmp_path),
            annotation_file=str(ann_file),
        )

        # 断言
        assert len(dataset) == 1
        assert len(dataset.samples) == 1
        assert dataset.samples[0]["image"] == str(fits_file)
        assert len(dataset.samples[0]["annotations"]) == 2

    def test_getitem_returns_patch(self, tmp_path):
        """测试：__getitem__ 返回 patch 和标注"""
        # 准备
        # 确保 FITS 文件存在
        fits_file = tmp_path / "test.fits"
        fits_file.write_bytes(b"mock fits data")

        # 创建标注
        annotations = {
            "images": [
                {
                    "file": "test.fits",
                    "width": 1000,
                    "height": 1000,
                    "annotations": [
                        {"x": 500, "y": 500, "width": 20, "height": 20, "label": "real"},
                    ]
                }
            ]
        }
        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations))

        # 模拟 FITS 读取
        mock_image = np.random.rand(1000, 1000).astype(np.float32)

        with patch("scann.core.fits_io.read_fits") as mock_read_fits:
            mock_read_fits.return_value = Mock(data=mock_image, header=Mock(raw={}))

            dataset = FitsDetectionDataset(
                image_dir=str(tmp_path),
                annotation_file=str(ann_file),
                patch_size=224,
            )

            # 执行
            patch_data, targets = dataset[0]

            # 断言
            assert patch_data.shape == (3, 224, 224)  # 三通道
            assert isinstance(targets, list)

    def test_getitem_with_crop(self, tmp_path):
        """测试：__getitem__ 支持裁剪特定区域"""
        # 确保 FITS 文件存在
        fits_file = tmp_path / "test.fits"
        fits_file.write_bytes(b"mock fits data")

        annotations = {
            "images": [
                {
                    "file": "test.fits",
                    "width": 1000,
                    "height": 1000,
                    "annotations": [{"x": 100, "y": 100, "width": 20, "height": 20, "label": "real"}]
                }
            ]
        }
        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations))

        mock_image = np.random.rand(1000, 1000).astype(np.float32)

        with patch("scann.core.fits_io.read_fits") as mock_read_fits:
            mock_read_fits.return_value = Mock(data=mock_image, header=Mock(raw={}))

            dataset = FitsDetectionDataset(
                image_dir=str(tmp_path),
                annotation_file=str(ann_file),
                patch_size=224,
            )

            # 请求特定区域的 patch
            patch_data, targets = dataset.get_crop(0, 0, 0, 224)

            # 断言
            assert patch_data.shape == (3, 224, 224)

    def test_sliding_window_patches(self, tmp_path):
        """测试：滑动窗口提取 patches"""
        # 确保 FITS 文件存在
        fits_file = tmp_path / "test.fits"
        fits_file.write_bytes(b"mock fits data")

        annotations = {
            "images": [
                {
                    "file": "test.fits",
                    "width": 500,
                    "height": 500,
                    "annotations": []
                }
            ]
        }
        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations))

        mock_image = np.random.rand(500, 500).astype(np.float32)

        with patch("scann.core.fits_io.read_fits") as mock_read_fits:
            mock_read_fits.return_value = Mock(data=mock_image, header=Mock(raw={}))

            dataset = FitsDetectionDataset(
                image_dir=str(tmp_path),
                annotation_file=str(ann_file),
                patch_size=224,
                stride=112,  # 50% overlap
            )

            # 获取所有 patches
            patches = list(dataset.iter_patches(0))

            # 断言
            # 500x500 图像，224 patch size，112 stride
            # 应该产生多个 patches
            assert len(patches) > 0
            assert all(patch_data.shape == (3, 224, 224) for patch_data, _ in patches)

    def test_annotation_to_targets(self, tmp_path):
        """测试：将标注转换为训练目标"""
        # 确保 FITS 文件存在
        fits_file = tmp_path / "test.fits"
        fits_file.write_bytes(b"mock fits data")

        annotations = {
            "images": [
                {
                    "file": "test.fits",
                    "width": 1000,
                    "height": 1000,
                    "annotations": [
                        {"x": 100, "y": 100, "width": 20, "height": 20, "label": "real"},
                        {"x": 500, "y": 500, "width": 15, "height": 15, "label": "bogus"},
                    ]
                }
            ]
        }
        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations))

        dataset = FitsDetectionDataset(
            image_dir=str(tmp_path),
            annotation_file=str(ann_file),
        )

        # 执行
        assert len(dataset.samples) > 0, "Should have loaded at least one sample"
        targets = dataset._annotations_to_targets(
            dataset.samples[0]["annotations"],
            crop_box=(0, 0, 224, 224)
        )

        # 断言：应该返回归一化的边界框
        assert isinstance(targets, list)
        # 验证边界框格式 [x_center, y_center, width, height, class_id]
        if targets:
            assert len(targets[0]) == 5
            # 坐标应该在 0-1 范围内
            assert all(0 <= val <= 1 for val in targets[0][:4])

    def test_empty_dataset(self, tmp_path):
        """测试：空数据集"""
        annotations = {"images": []}
        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations))

        dataset = FitsDetectionDataset(
            image_dir=str(tmp_path),
            annotation_file=str(ann_file),
        )

        # 断言
        assert len(dataset) == 0
        assert dataset.samples == []

    def test_label_mapping(self, tmp_path):
        """测试：标签映射"""
        # 确保 FITS 文件存在
        fits_file = tmp_path / "test.fits"
        fits_file.write_bytes(b"mock fits data")

        annotations = {
            "images": [
                {
                    "file": "test.fits",
                    "width": 1000,
                    "height": 1000,
                    "annotations": [
                        {"x": 100, "y": 100, "width": 20, "height": 20, "label": "real"},
                        {"x": 500, "y": 500, "width": 15, "height": 15, "label": "bogus"},
                    ]
                }
            ]
        }
        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations))

        dataset = FitsDetectionDataset(
            image_dir=str(tmp_path),
            annotation_file=str(ann_file),
            label_map={"real": 1, "bogus": 0},
        )

        # 执行
        assert len(dataset.samples) > 0, "Should have loaded at least one sample"
        sample = dataset.samples[0]
        targets = dataset._annotations_to_targets(sample["annotations"], (0, 0, 1000, 1000))

        # 断言：检查标签映射
        assert len(targets) == 2
        assert targets[0][4] == 1  # real -> 1
        assert targets[1][4] == 0  # bogus -> 0

    def test_normalize_patch(self, tmp_path):
        """测试：patch 归一化"""
        # 确保 FITS 文件存在
        fits_file = tmp_path / "test.fits"
        fits_file.write_bytes(b"mock fits data")

        annotations = {
            "images": [
                {
                    "file": "test.fits",
                    "width": 1000,
                    "height": 1000,
                    "annotations": []
                }
            ]
        }
        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations))

        # 创建有明显范围的图像
        mock_image = (np.random.rand(1000, 1000) * 1000 + 100).astype(np.float32)

        with patch("scann.core.fits_io.read_fits") as mock_read_fits:
            mock_read_fits.return_value = Mock(data=mock_image, header=Mock(raw={}))

            dataset = FitsDetectionDataset(
                image_dir=str(tmp_path),
                annotation_file=str(ann_file),
                patch_size=224,
            )

            # 执行
            patch_data, _ = dataset[0]

            # 断言：patch 应该在 0-1 范围内
            assert np.all(patch_data >= 0)
            assert np.all(patch_data <= 1)
            assert patch_data.dtype == np.float32
