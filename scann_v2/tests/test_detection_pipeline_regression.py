from unittest.mock import Mock, patch

import numpy as np

from scann.core.models import AlignResult, Candidate, Detection
from scann.core.models import FitsHeader
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
        pipeline.exclusion_service.check_candidates.return_value = [
            kept_candidate,
            removed_candidate,
        ]
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

    @patch("scann.services.detection_service.detect_candidates")
    @patch("scann.services.detection_service.align")
    def test_process_pair_passes_header_and_image_path_to_exclusion_service(
        self,
        mock_align,
        mock_detect,
    ):
        exclusion_service = Mock()
        exclusion_service.check_candidates.side_effect = lambda candidates, header=None, image_path=None: candidates

        pipeline = DetectionPipeline(exclusion_service=exclusion_service)
        mock_align.return_value = AlignResult(
            aligned_old=np.zeros((16, 16), dtype=np.float32),
            success=True,
        )
        mock_detect.return_value = [Candidate(x=5, y=6)]

        image = np.ones((16, 16), dtype=np.float32)
        header = FitsHeader(raw={"RA": 1.0, "DEC": 2.0})
        image_path = "C:/data/example.fit"

        pipeline.process_pair(
            "pair-005",
            image,
            image,
            header=header,
            image_path=image_path,
        )

        exclusion_service.check_candidates.assert_called_once_with(
            mock_detect.return_value,
            header=header,
            image_path=image_path,
        )

    @patch("scann.services.detection_service.align")
    def test_process_pair_skip_align_bypasses_alignment(self, mock_align):
        pipeline = DetectionPipeline()
        image = np.zeros((48, 48), dtype=np.float32)

        result = pipeline.process_pair("pair-004", image, image, skip_align=True)

        mock_align.assert_not_called()
        assert isinstance(result, PipelineResult)
        assert result.align_result is None

    @patch("scann.services.detection_service.align")
    def test_process_pair_uses_auto_alignment_method(self, mock_align):
        pipeline = DetectionPipeline()
        image = np.ones((32, 32), dtype=np.float32)
        mock_align.return_value = AlignResult(
            aligned_old=image.copy(),
            success=True,
        )

        pipeline.process_pair("pair-auto-001", image, image)

        assert mock_align.call_args.kwargs["method"] == "auto"

    @patch("scann.services.detection_service.detect_candidates")
    @patch("scann.services.detection_service.align")
    def test_process_pair_full_image_mode_uses_dense_and_maps_sorted_candidates(
        self,
        mock_align,
        mock_detect,
    ):
        inference_engine = Mock()
        inference_engine.is_ready = True
        inference_engine.threshold = 0.55
        inference_engine.detect_dense_full_image.return_value = [
            Detection(x=21, y=22, width=10, height=8, confidence=0.72),
            Detection(x=11, y=12, width=6, height=6, confidence=0.91),
        ]

        pipeline = DetectionPipeline(
            inference_engine=inference_engine,
            patch_size=16,
            detection_mode="full_image",
        )
        mock_align.return_value = AlignResult(
            aligned_old=np.zeros((64, 64), dtype=np.float32),
            success=True,
        )

        image = np.ones((64, 64), dtype=np.float32)
        result = pipeline.process_pair("pair-full-001", image, image)

        mock_detect.assert_not_called()
        inference_engine.detect_dense_full_image.assert_called_once()
        assert [(candidate.x, candidate.y) for candidate in result.candidates] == [
            (11, 12),
            (21, 22),
        ]
        assert [candidate.ai_score for candidate in result.candidates] == [0.91, 0.72]

    @patch("scann.services.detection_service.detect_candidates")
    @patch("scann.services.detection_service.align")
    def test_process_pair_full_image_mode_still_applies_exclusion_service(
        self,
        mock_align,
        mock_detect,
    ):
        inference_engine = Mock()
        inference_engine.is_ready = True
        inference_engine.threshold = 0.5
        inference_engine.detect_dense_full_image.return_value = [
            Detection(x=31, y=32, width=12, height=10, confidence=0.85),
        ]

        exclusion_service = Mock()
        exclusion_service.check_candidates.side_effect = lambda candidates, header=None, image_path=None: [
            Candidate(
                x=candidates[0].x,
                y=candidates[0].y,
                ai_score=candidates[0].ai_score,
                is_known=True,
            )
        ]

        pipeline = DetectionPipeline(
            inference_engine=inference_engine,
            exclusion_service=exclusion_service,
            detection_mode="full_image",
        )
        mock_align.return_value = AlignResult(
            aligned_old=np.zeros((32, 32), dtype=np.float32),
            success=True,
        )

        image = np.ones((32, 32), dtype=np.float32)
        header = FitsHeader(raw={"RA": 1.0, "DEC": 2.0})
        result = pipeline.process_pair(
            "pair-full-002",
            image,
            image,
            header=header,
            image_path="C:/data/full.fit",
        )

        mock_detect.assert_not_called()
        exclusion_service.check_candidates.assert_called_once()
        assert result.candidates == []

    @patch("scann.services.detection_service.detect_candidates")
    @patch("scann.services.detection_service.align")
    def test_process_pair_hybrid_fallbacks_to_patch_when_full_image_raises(
        self,
        mock_align,
        mock_detect,
    ):
        inference_engine = Mock()
        inference_engine.is_ready = True
        inference_engine.threshold = 0.5
        inference_engine.detect_dense_full_image.side_effect = RuntimeError("dense failed")
        inference_engine.classify_patches.return_value = [0.88]

        pipeline = DetectionPipeline(
            inference_engine=inference_engine,
            detection_mode="hybrid",
            hybrid_primary_mode="full_image",
        )
        mock_align.return_value = AlignResult(
            aligned_old=np.zeros((64, 64), dtype=np.float32),
            success=True,
        )
        mock_detect.return_value = [Candidate(x=25, y=26)]

        image = np.ones((64, 64), dtype=np.float32)
        result = pipeline.process_pair("pair-hybrid-001", image, image)

        inference_engine.detect_dense_full_image.assert_called_once()
        mock_detect.assert_called_once()
        assert len(result.candidates) == 1
        assert (result.candidates[0].x, result.candidates[0].y) == (25, 26)

    @patch("scann.services.detection_service.detect_candidates")
    @patch("scann.services.detection_service.align")
    def test_process_pair_hybrid_fallbacks_to_patch_on_low_dense_confidence(
        self,
        mock_align,
        mock_detect,
    ):
        inference_engine = Mock()
        inference_engine.is_ready = True
        inference_engine.threshold = 0.5
        inference_engine.detect_dense_full_image.return_value = [
            Detection(x=10, y=11, width=6, height=6, confidence=0.61),
        ]
        inference_engine.classify_patches.return_value = [0.93]

        pipeline = DetectionPipeline(
            inference_engine=inference_engine,
            detection_mode="hybrid",
            hybrid_primary_mode="full_image",
            hybrid_low_confidence=0.8,
        )
        mock_align.return_value = AlignResult(
            aligned_old=np.zeros((64, 64), dtype=np.float32),
            success=True,
        )
        mock_detect.return_value = [Candidate(x=30, y=31)]

        image = np.ones((64, 64), dtype=np.float32)
        result = pipeline.process_pair("pair-hybrid-002", image, image)

        inference_engine.detect_dense_full_image.assert_called_once()
        mock_detect.assert_called_once()
        assert len(result.candidates) == 1
        assert (result.candidates[0].x, result.candidates[0].y) == (30, 31)

    @patch("scann.services.detection_service.detect_candidates")
    @patch("scann.services.detection_service.align")
    def test_process_pair_hybrid_patch_first_skips_full_image_when_patch_has_result(
        self,
        mock_align,
        mock_detect,
    ):
        inference_engine = Mock()
        inference_engine.is_ready = True
        inference_engine.threshold = 0.4
        inference_engine.classify_patches.return_value = [0.79]

        pipeline = DetectionPipeline(
            inference_engine=inference_engine,
            detection_mode="hybrid",
            hybrid_primary_mode="patch",
        )
        mock_align.return_value = AlignResult(
            aligned_old=np.zeros((64, 64), dtype=np.float32),
            success=True,
        )
        mock_detect.return_value = [Candidate(x=40, y=41)]

        image = np.ones((64, 64), dtype=np.float32)
        result = pipeline.process_pair("pair-hybrid-003", image, image)

        mock_detect.assert_called_once()
        inference_engine.detect_dense_full_image.assert_not_called()
        assert len(result.candidates) == 1
        assert (result.candidates[0].x, result.candidates[0].y) == (40, 41)
