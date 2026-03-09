from unittest.mock import Mock, patch

import numpy as np

from scann.core.models import AlignResult, Candidate
from scann.services.detection_service import DetectionPipeline, PipelineResult


class TestDetectionPipelineRegression:
    @patch("scann.services.detection_service.align")
    def test_process_pair_returns_error_when_alignment_fails(self, mock_align):
        pipeline = DetectionPipeline()
        mock_align.return_value = AlignResult(
            aligned_old=None,
            success=False,
            error_message="no stars",
        )

        image = np.zeros((32, 32), dtype=np.float32)
        result = pipeline.process_pair("pair-001", image, image)

        assert isinstance(result, PipelineResult)
        assert result.pair_name == "pair-001"
        assert result.candidates == []
        assert result.error == "对齐失败: no stars"
        assert result.align_result is mock_align.return_value

    @patch("scann.services.detection_service.detect_candidates")
    @patch("scann.services.detection_service.align")
    def test_process_pair_sorts_and_filters_candidates_by_ai_score(
        self,
        mock_align,
        mock_detect,
    ):
        inference_engine = Mock()
        inference_engine.is_ready = True
        inference_engine.threshold = 0.6
        inference_engine.classify_patches.return_value = [0.62, 0.91, 0.31]

        pipeline = DetectionPipeline(inference_engine=inference_engine, patch_size=16)
        mock_align.return_value = AlignResult(
            aligned_old=np.zeros((64, 64), dtype=np.float32),
            dx=0.0,
            dy=0.0,
            success=True,
        )
        mock_detect.return_value = [
            Candidate(x=10, y=20),
            Candidate(x=30, y=40),
            Candidate(x=50, y=60),
        ]

        image = np.ones((64, 64), dtype=np.float32)
        result = pipeline.process_pair("pair-002", image, image)

        assert [candidate.ai_score for candidate in result.candidates] == [0.91, 0.62]
        assert [(candidate.x, candidate.y) for candidate in result.candidates] == [
            (30, 40),
            (10, 20),
        ]
        assert result.error == ""
        mock_detect.assert_called_once()

    @patch("scann.services.detection_service.detect_candidates")
    @patch("scann.services.detection_service.align")
    def test_process_pair_applies_exclusion_service_after_ai_filtering(
        self,
        mock_align,
        mock_detect,
    ):
        inference_engine = Mock()
        inference_engine.is_ready = True
        inference_engine.threshold = 0.5
        inference_engine.classify_patches.return_value = [0.8, 0.7]

        kept_candidate = Candidate(x=11, y=12)
        removed_candidate = Candidate(x=21, y=22, is_known=True)

        pipeline = DetectionPipeline(
            inference_engine=inference_engine,
            exclusion_service=Mock(),
            patch_size=16,
        )
        mock_align.return_value = AlignResult(
            aligned_old=np.zeros((32, 32), dtype=np.float32),
            success=True,
        )
        mock_detect.return_value = [kept_candidate, removed_candidate]

        with patch.object(pipeline, "_exclude_known", wraps=pipeline._exclude_known) as mock_exclude:
            image = np.ones((32, 32), dtype=np.float32)
            result = pipeline.process_pair("pair-003", image, image)

        assert result.candidates == [kept_candidate]
        mock_exclude.assert_called_once()

    @patch("scann.services.detection_service.align")
    def test_process_pair_skip_align_bypasses_alignment(self, mock_align):
        pipeline = DetectionPipeline()
        image = np.zeros((48, 48), dtype=np.float32)

        result = pipeline.process_pair("pair-004", image, image, skip_align=True)

        mock_align.assert_not_called()
        assert isinstance(result, PipelineResult)
        assert result.align_result is None