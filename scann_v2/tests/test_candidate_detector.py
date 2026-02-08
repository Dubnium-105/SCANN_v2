"""候选体检测模块单元测试"""

import numpy as np
import pytest


class TestCandidateDetector:
    """测试候选体检测 (从旧版 process_stage_a 移植的逻辑)"""

    def _make_image_pair_with_sources(self, n_sources: int = 3):
        """创建包含模拟星点的新旧图像对"""
        rng = np.random.default_rng(42)
        old_img = rng.normal(loc=100, scale=10, size=(256, 256)).astype(np.float32)
        new_img = old_img.copy()

        y, x = np.mgrid[0:256, 0:256]
        for i in range(n_sources):
            cy, cx = rng.integers(30, 226, size=2)
            flux = rng.uniform(200, 1000)
            star = flux * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 2.5 ** 2))
            new_img += star.astype(np.float32)

        return new_img, old_img

    def test_detect_returns_list(self):
        from scann.core.candidate_detector import detect_candidates

        new_img, old_img = self._make_image_pair_with_sources(5)
        candidates = detect_candidates(new_img, old_img)
        assert isinstance(candidates, list)

    def test_detect_finds_sources(self):
        from scann.core.candidate_detector import DetectionParams, detect_candidates

        new_img, old_img = self._make_image_pair_with_sources(5)
        # 使用宽松参数确保合成数据能通过过滤器
        params = DetectionParams(thresh=50, kill_flat=False, kill_dipole=False)
        candidates = detect_candidates(new_img, old_img, params=params)
        assert len(candidates) >= 1  # 至少检测到一些源

    def test_candidates_have_positions(self):
        from scann.core.candidate_detector import detect_candidates

        new_img, old_img = self._make_image_pair_with_sources(3)
        candidates = detect_candidates(new_img, old_img)
        for c in candidates:
            assert hasattr(c, "x")
            assert hasattr(c, "y")
            assert 0 <= c.x < 256
            assert 0 <= c.y < 256

    def test_candidates_have_features(self):
        from scann.core.candidate_detector import detect_candidates

        new_img, old_img = self._make_image_pair_with_sources(3)
        candidates = detect_candidates(new_img, old_img)
        for c in candidates:
            assert hasattr(c, "features")

    def test_detection_params_threshold(self):
        from scann.core.candidate_detector import DetectionParams, detect_candidates

        new_img, old_img = self._make_image_pair_with_sources(5)

        # 低阈值 -> 更多候选
        params_low = DetectionParams(thresh=20)
        low = detect_candidates(new_img, old_img, params=params_low)

        # 高阈值 -> 更少候选
        params_high = DetectionParams(thresh=200)
        high = detect_candidates(new_img, old_img, params=params_high)

        assert len(low) >= len(high)

    def test_empty_image_returns_empty(self):
        from scann.core.candidate_detector import detect_candidates

        empty = np.zeros((128, 128), dtype=np.float32)
        candidates = detect_candidates(empty, empty)
        assert candidates == []
