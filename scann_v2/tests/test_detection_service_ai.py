"""Detection Service AI 评分测试

使用测试驱动开发 (TDD) 实现：
1. 提取 patch → 推理 → 回填分数
"""

import pytest
from unittest.mock import Mock, MagicMock

import numpy as np

from scann.core.models import Candidate, CandidateFeatures
from scann.services.detection_service import DetectionPipeline, PipelineResult


class TestAIScoring:
    """测试 AI 评分功能"""

    def test_extract_patch_from_image(self):
        """测试：从图像中提取 patch"""
        # 准备
        pipeline = DetectionPipeline()

        # 创建测试图像 (灰度)
        # shape: (10, 10) = (rows, cols) = (y, x)
        image = np.arange(100).reshape(10, 10).astype(np.float32)

        # 提取以 (x=2, y=2) 为中心的 patch，大小为 4x4
        # 注意：x 是列，y 是行
        patch = pipeline._extract_patch(image, 2, 2, 4)

        # 断言：patch 应该是 (4, 4)
        assert patch.shape == (4, 4)
        # 验证内容：以 (2, 2) 为中心，half=2
        # 边界: y[0, 4], x[0, 4]
        expected = np.array([
            [ 0,  1,  2,  3],
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [30, 31, 32, 33],
        ], dtype=np.float32)
        np.testing.assert_array_equal(patch, expected)

    def test_extract_patch_with_padding(self):
        """测试：边界外 padding"""
        pipeline = DetectionPipeline()
        image = np.arange(100).reshape(10, 10).astype(np.float32)

        # 从边界提取 (需要 padding)
        patch = pipeline._extract_patch(image, 0, 0, 4)

        # 断言：应该包含填充值
        assert patch.shape == (4, 4)
        # 左上角应该是填充值
        assert patch[0, 0] == 0

    def test_prepare_triplet_patch(self):
        """测试：准备三元组 patch (new + old + diff)"""
        pipeline = DetectionPipeline()

        # 模拟新旧图
        new_data = np.random.rand(20, 20).astype(np.float32)
        old_data = np.random.rand(20, 20).astype(np.float32)

        # 准备三元组 patch
        patch_3ch = pipeline._prepare_triplet_patch(new_data, old_data, 5, 5, 8)

        # 断言：应该被调整为 (3, 224, 224)
        assert patch_3ch.shape == (3, 224, 224)
        # 值应该在 0-1 范围
        assert np.all(patch_3ch >= 0) and np.all(patch_3ch <= 1)

    def test_ai_score_updates_candidates(self):
        """测试：AI 评分应该更新候选体分数"""
        # 准备
        mock_engine = Mock()
        mock_engine.is_ready = True
        mock_engine.classify_patches.return_value = [0.9, 0.3, 0.7]

        pipeline = DetectionPipeline(inference_engine=mock_engine)

        # 准备候选体
        new_data = np.random.rand(100, 100).astype(np.float32)
        old_data = np.random.rand(100, 100).astype(np.float32)

        candidates = [
            Candidate(x=10, y=10),
            Candidate(x=50, y=50),
            Candidate(x=80, y=80),
        ]

        # 执行
        result = pipeline._ai_score(candidates, new_data, old_data)

        # 断言：所有候选体都应被返回
        assert len(result) == 3
        # 分数应该被更新
        assert result[0].ai_score == pytest.approx(0.9)
        assert result[1].ai_score == pytest.approx(0.3)
        assert result[2].ai_score == pytest.approx(0.7)

    def test_ai_score_with_no_engine(self):
        """测试：没有推理引擎时应返回原候选体"""
        pipeline = DetectionPipeline(inference_engine=None)

        new_data = np.random.rand(100, 100).astype(np.float32)
        old_data = np.random.rand(100, 100).astype(np.float32)

        candidates = [Candidate(x=10, y=10)]

        # 执行
        result = pipeline._ai_score(candidates, new_data, old_data)

        # 断言：候选体不变
        assert len(result) == 1
        assert result[0].ai_score == 0.0

    def test_ai_score_with_empty_candidates(self):
        """测试：空候选体列表应返回空列表"""
        mock_engine = Mock()
        mock_engine.is_ready = True

        pipeline = DetectionPipeline(inference_engine=mock_engine)

        new_data = np.random.rand(100, 100).astype(np.float32)
        old_data = np.random.rand(100, 100).astype(np.float32)

        # 执行
        result = pipeline._ai_score([], new_data, old_data)

        # 断言
        assert result == []

    def test_ai_score_normalization(self):
        """测试：patch 应该正确归一化"""
        mock_engine = Mock()
        mock_engine.is_ready = True

        # 捕获传递给 classify_patches 的 patches
        captured_patches = []

        def capture_patches(patches, **kwargs):
            captured_patches.extend(patches)
            return [0.5]

        mock_engine.classify_patches.side_effect = capture_patches

        pipeline = DetectionPipeline(inference_engine=mock_engine)

        # 创建有明确范围的图像
        new_data = (np.random.rand(100, 100) * 100).astype(np.float32)
        old_data = (np.random.rand(100, 100) * 100).astype(np.float32)

        candidates = [Candidate(x=50, y=50)]

        # 执行
        pipeline._ai_score(candidates, new_data, old_data)

        # 断言：patch 应该在 0-1 范围内
        assert len(captured_patches) == 1
        patch = captured_patches[0]
        assert patch.shape == (3, 224, 224)
        assert np.all(patch >= 0) and np.all(patch <= 1)

    def test_process_pair_calls_ai_scoring(self):
        """测试：process_pair 应该调用 AI 评分"""
        mock_engine = Mock()
        mock_engine.is_ready = True
        mock_engine.classify_patches.return_value = [0.8]

        pipeline = DetectionPipeline(inference_engine=mock_engine)

        new_data = np.random.rand(100, 100).astype(np.float32)
        old_data = np.random.rand(100, 100).astype(np.float32)

        # 模拟一个候选体
        with pytest.MonkeyPatch().context() as m:
            # 模拟 detect_candidates 返回一个候选体
            def mock_detect(new, old, **kwargs):
                return [Candidate(x=50, y=50)]
            m.setattr("scann.services.detection_service.detect_candidates", mock_detect)

            # 执行
            result = pipeline.process_pair("test", new_data, old_data, skip_align=True)

        # 断言：候选体应该有 AI 分数
        assert len(result.candidates) == 1
        assert result.candidates[0].ai_score == pytest.approx(0.8)
