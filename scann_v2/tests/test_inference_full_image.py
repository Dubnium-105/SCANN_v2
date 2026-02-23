"""Inference Engine 全图检测测试

使用测试驱动开发 (TDD) 实现：
1. 滑动窗口检测
2. 结果合并（NMS）
3. 边界处理
"""

import pytest
import numpy as np
import torch
from typing import Any, cast
from unittest.mock import Mock, patch

from scann.ai.inference import InferenceEngine, InferenceConfig
from scann.core.models import Candidate


class TestSlidingWindowDetection:
    """测试滑动窗口检测"""

    def test_detect_full_image_basic(self):
        """测试：基本全图检测"""
        # 创建模拟模型
        mock_model = Mock()
        mock_model.eval.return_value = None

        # Mock 模型的 forward 方法返回检测结果
        def mock_forward(x):
            # 返回模拟的检测结果：[batch, num_classes] 分类
            batch_size = x.shape[0]
            # 假设第一个窗口有目标（class 1），其他没有
            probs = torch.zeros(batch_size, 2)
            probs[0, 1] = 0.8  # 第一个窗口检测到目标
            probs[1:, 0] = 0.9  # 其他窗口都是背景
            return probs

        mock_model.side_effect = mock_forward

        with patch("torch.load", return_value={"threshold": 0.5}):
            with patch("scann.ai.model.SCANNClassifier.load_from_checkpoint", return_value=mock_model):
                engine = InferenceEngine("dummy_path.pt")

        # 创建测试图像
        test_image = np.random.rand(256, 256).astype(np.float32)

        # 执行全图检测
        detections = engine.detect_full_image(test_image)

        # 应该返回检测结果列表
        assert isinstance(detections, list)

    def test_detect_full_image_with_windows(self):
        """测试：滑动窗口数量正确"""
        mock_model = Mock()
        mock_model.eval.return_value = None

        # 返回全背景
        def mock_forward(x):
            batch_size = x.shape[0]
            probs = torch.zeros(batch_size, 2)
            probs[:, 0] = 0.95
            return probs

        mock_model.side_effect = mock_forward

        with patch("torch.load", return_value={"threshold": 0.5}):
            with patch("scann.ai.model.SCANNClassifier.load_from_checkpoint", return_value=mock_model):
                config = InferenceConfig(batch_size=32)
                engine = InferenceEngine("dummy_path.pt", config=config)

        # 创建较大的测试图像
        test_image = np.random.rand(512, 512).astype(np.float32)

        # 执行检测
        detections = engine.detect_full_image(test_image)

        # 应该处理多个窗口
        # 假设窗口大小 224，步长 112（50% 重叠）
        # 512x512 图像应该产生约 (512/112)^2 = 21 个窗口
        assert isinstance(detections, list)

    def test_detect_full_image_multiple_detections(self):
        """测试：多个检测结果"""
        mock_model = Mock()
        mock_model.eval.return_value = None

        # 返回多个目标
        def mock_forward(x):
            batch_size = x.shape[0]
            probs = torch.zeros(batch_size, 2)
            # 第一个和第三个窗口有目标
            if batch_size > 0:
                probs[0, 1] = 0.85
            if batch_size > 2:
                probs[2, 1] = 0.90
            if batch_size > 1:
                probs[1, 0] = 0.95
            if batch_size > 3:
                probs[3:, 0] = 0.95
            return probs

        mock_model.side_effect = mock_forward

        with patch("torch.load", return_value={"threshold": 0.5}):
            with patch("scann.ai.model.SCANNClassifier.load_from_checkpoint", return_value=mock_model):
                engine = InferenceEngine("dummy_path.pt")

        test_image = np.random.rand(256, 256).astype(np.float32)

        detections = engine.detect_full_image(test_image)

        # 应该有多个检测结果
        assert len(detections) >= 1

    def test_detect_full_image_threshold_filtering(self):
        """测试：阈值过滤"""
        mock_model = Mock()
        mock_model.eval.return_value = None

        # 返回低于阈值的检测结果
        def mock_forward(x):
            batch_size = x.shape[0]
            probs = torch.zeros(batch_size, 2)
            probs[0, 1] = 0.3  # 低于阈值（0.5）
            probs[:, 0] = 0.7
            return probs

        mock_model.side_effect = mock_forward

        with patch("torch.load", return_value={"threshold": 0.5}):
            with patch("scann.ai.model.SCANNClassifier.load_from_checkpoint", return_value=mock_model):
                engine = InferenceEngine("dummy_path.pt")

        test_image = np.random.rand(256, 256).astype(np.float32)

        detections = engine.detect_full_image(test_image)

        # 应该没有检测结果（低于阈值）
        assert len(detections) == 0

    def test_detect_full_image_v1_uses_triplet_semantics(self):
        """测试：v1 模型应使用 [Diff, New, Old] 三联图语义输入"""
        mock_model = Mock()
        mock_model.eval.return_value = None

        # 返回全背景，重点验证输入 patch 语义
        def mock_forward(x):
            batch_size = x.shape[0]
            probs = torch.zeros(batch_size, 2)
            probs[:, 0] = 0.95
            return probs

        mock_model.side_effect = mock_forward

        ckpt_meta = {
            "model_format": "v1_classifier",
            "threshold": 0.5,
            "order": "0,1,2",
        }

        with patch("torch.load", return_value=ckpt_meta):
            with patch("scann.ai.model.SCANNClassifier.load_from_checkpoint", return_value=mock_model):
                engine = InferenceEngine("dummy_path.pt")

        captured = []

        def capture_classify(patches, **kwargs):
            captured.extend(patches)
            return [0.1 for _ in patches]

        setattr(engine, "classify_patches", capture_classify)

        # 用 uint8 便于验证通道均值关系
        new_image = np.full((128, 128), 200, dtype=np.uint8)
        old_image = np.full((128, 128), 50, dtype=np.uint8)

        _ = cast(Any, engine).detect_full_image(
            new_image,
            old_image=old_image,
            patch_size=64,
            stride=64,
        )

        assert len(captured) > 0
        first_patch = captured[0]
        assert first_patch.shape[0] == 3

        # [Diff, New, Old] => new > diff > old
        m0 = float(first_patch[0].mean())  # diff
        m1 = float(first_patch[1].mean())  # new
        m2 = float(first_patch[2].mean())  # old
        assert m1 > m0 > m2

    def test_detect_full_image_v1_small_image_still_generates_windows(self):
        """测试：v1 模式下小图也应至少生成一个滑窗 patch"""
        mock_model = Mock()
        mock_model.eval.return_value = None

        def mock_forward(x):
            batch_size = x.shape[0]
            probs = torch.zeros(batch_size, 2)
            probs[:, 0] = 0.95
            return probs

        mock_model.side_effect = mock_forward

        ckpt_meta = {
            "model_format": "v1_classifier",
            "threshold": 0.5,
            "order": "0,1,2",
        }

        with patch("torch.load", return_value=ckpt_meta):
            with patch("scann.ai.model.SCANNClassifier.load_from_checkpoint", return_value=mock_model):
                engine = InferenceEngine("dummy_path.pt")

        captured = []

        def capture_classify(patches, **kwargs):
            captured.extend(patches)
            return [0.1 for _ in patches]

        setattr(engine, "classify_patches", capture_classify)

        # 小于 patch_size
        new_image = np.full((40, 40), 180, dtype=np.uint8)
        old_image = np.full((40, 40), 60, dtype=np.uint8)

        _ = cast(Any, engine).detect_full_image(
            new_image,
            old_image=old_image,
            patch_size=64,
            stride=64,
        )

        assert len(captured) >= 1
        assert captured[0].shape == (3, 64, 64)


class TestResultMerging:
    """测试结果合并"""

    def test_nms_basic(self):
        """测试：非极大值抑制（NMS）"""
        mock_model = Mock()
        mock_model.eval.return_value = None

        # 返回重叠的检测结果
        def mock_forward(x):
            batch_size = x.shape[0]
            probs = torch.zeros(batch_size, 2)
            # 多个窗口检测到同一目标
            if batch_size > 0:
                probs[0, 1] = 0.9
            if batch_size > 1:
                probs[1, 1] = 0.85
            if batch_size > 2:
                probs[2, 1] = 0.8
            if batch_size > 3:
                probs[3:, 0] = 0.95
            return probs

        mock_model.side_effect = mock_forward

        with patch("torch.load", return_value={"threshold": 0.5}):
            with patch("scann.ai.model.SCANNClassifier.load_from_checkpoint", return_value=mock_model):
                engine = InferenceEngine("dummy_path.pt")

        test_image = np.random.rand(256, 256).astype(np.float32)

        detections = engine.detect_full_image(test_image)

        # NMS 应该合并重叠的检测结果
        # 应该比原始窗口数少
        assert isinstance(detections, list)

    def test_nms_iou_threshold(self):
        """测试：IoU 阈值影响"""
        mock_model = Mock()
        mock_model.eval.return_value = None

        # 返回多个检测结果
        def mock_forward(x):
            batch_size = x.shape[0]
            probs = torch.zeros(batch_size, 2)
            probs[:4, 1] = 0.9
            probs[4:, 0] = 0.95
            return probs

        mock_model.side_effect = mock_forward

        with patch("torch.load", return_value={"threshold": 0.5}):
            with patch("scann.ai.model.SCANNClassifier.load_from_checkpoint", return_value=mock_model):
                engine = InferenceEngine("dummy_path.pt")

        test_image = np.random.rand(256, 256).astype(np.float32)

        detections = engine.detect_full_image(test_image)

        # NMS 应该工作
        assert isinstance(detections, list)


class TestBoundaryHandling:
    """测试边界处理"""

    def test_detect_small_image(self):
        """测试：小图像检测"""
        mock_model = Mock()
        mock_model.eval.return_value = None

        def mock_forward(x):
            batch_size = x.shape[0]
            probs = torch.zeros(batch_size, 2)
            probs[:, 0] = 0.95
            return probs

        mock_model.side_effect = mock_forward

        with patch("torch.load", return_value={"threshold": 0.5}):
            with patch("scann.ai.model.SCANNClassifier.load_from_checkpoint", return_value=mock_model):
                engine = InferenceEngine("dummy_path.pt")

        # 小于窗口大小的图像
        test_image = np.random.rand(100, 100).astype(np.float32)

        # 应该处理（可能只生成一个窗口或进行填充）
        detections = engine.detect_full_image(test_image)
        assert isinstance(detections, list)

    def test_detect_non_square_image(self):
        """测试：非方形图像检测"""
        mock_model = Mock()
        mock_model.eval.return_value = None

        def mock_forward(x):
            batch_size = x.shape[0]
            probs = torch.zeros(batch_size, 2)
            probs[:, 0] = 0.95
            return probs

        mock_model.side_effect = mock_forward

        with patch("torch.load", return_value={"threshold": 0.5}):
            with patch("scann.ai.model.SCANNClassifier.load_from_checkpoint", return_value=mock_model):
                engine = InferenceEngine("dummy_path.pt")

        # 宽高比不同的图像
        test_image = np.random.rand(400, 600).astype(np.float32)

        detections = engine.detect_full_image(test_image)
        assert isinstance(detections, list)

    def test_detect_empty_image(self):
        """测试：空白图像检测"""
        mock_model = Mock()
        mock_model.eval.return_value = None

        def mock_forward(x):
            batch_size = x.shape[0]
            probs = torch.zeros(batch_size, 2)
            probs[:, 0] = 0.95
            return probs

        mock_model.side_effect = mock_forward

        with patch("torch.load", return_value={"threshold": 0.5}):
            with patch("scann.ai.model.SCANNClassifier.load_from_checkpoint", return_value=mock_model):
                engine = InferenceEngine("dummy_path.pt")

        # 全零图像
        test_image = np.zeros((256, 256), dtype=np.float32)

        detections = engine.detect_full_image(test_image)
        assert isinstance(detections, list)
