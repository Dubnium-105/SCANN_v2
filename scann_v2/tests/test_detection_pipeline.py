"""检测管线单元测试"""

import numpy as np
import pytest


class TestDetectionPipeline:
    """测试完整检测流水线 (对齐 → 检测 → AI 评分 → 排除 → 排序)"""

    def test_pipeline_result_structure(self, synth_image_pair):
        from scann.services.detection_service import DetectionPipeline, PipelineResult

        pipeline = DetectionPipeline()
        new_img, old_img = synth_image_pair
        result = pipeline.process_pair(
            pair_name="test_pair",
            new_data=new_img.astype(np.float32),
            old_data=old_img.astype(np.float32),
        )
        assert isinstance(result, PipelineResult)
        assert isinstance(result.candidates, list)

    def test_pipeline_candidates_sorted_by_score(self, synth_image_pair):
        from scann.services.detection_service import DetectionPipeline

        pipeline = DetectionPipeline()
        new_img, old_img = synth_image_pair
        result = pipeline.process_pair(
            pair_name="test_pair",
            new_data=new_img.astype(np.float32),
            old_data=old_img.astype(np.float32),
        )
        scores = [c.ai_score for c in result.candidates]
        assert scores == sorted(scores, reverse=True)

    def test_pipeline_empty_image(self):
        from scann.services.detection_service import DetectionPipeline

        pipeline = DetectionPipeline()
        empty = np.zeros((128, 128), dtype=np.float32)
        result = pipeline.process_pair(
            pair_name="empty_pair",
            new_data=empty,
            old_data=empty,
        )
        assert result.candidates == []
