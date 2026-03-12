"""Detection pipeline and helper orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from scann.core.candidate_detector import DetectionParams, detect_candidates
from scann.core.image_aligner import align
from scann.core.models import AlignResult, Candidate
from scann.services.detection_image_adapter import img_brief_stats, robust_to_uint8
from scann.services.detection_patch_extractor import MODEL_INPUT_SIZE, extract_patch, prepare_triplet_patch
from scann.services.detection_postprocess import nms_candidates
from scann.services.detection_window_scanner import sliding_window_detect

logger = logging.getLogger(__name__)


DEFAULT_PATCH_SIZE = 80


@dataclass
class PipelineResult:
    """管线处理结果。"""

    pair_name: str
    candidates: List[Candidate]
    align_result: Optional[AlignResult] = None
    error: str = ""


class DetectionPipeline:
    """完整检测管线。"""

    def __init__(
        self,
        detection_params: Optional[DetectionParams] = None,
        inference_engine=None,
        exclusion_service=None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        patch_size: int = DEFAULT_PATCH_SIZE,
    ):
        self.detection_params = detection_params or DetectionParams()
        self.inference_engine = inference_engine
        self.exclusion_service = exclusion_service
        self.progress_callback = progress_callback
        self.patch_size = patch_size

    def process_pair(
        self,
        pair_name: str,
        new_data: np.ndarray,
        old_data: np.ndarray,
        skip_align: bool = False,
        header=None,
        image_path: str | None = None,
    ) -> PipelineResult:
        """处理单对图像。"""
        ai_available = (
            self.inference_engine is not None
            and self.inference_engine.is_ready
        )
        is_v1 = self._is_v1_model() if ai_available else False

        if ai_available:
            ch = getattr(self.inference_engine, "_channel_order", (0, 1, 2))
            th = getattr(self.inference_engine, "threshold", 0.5)
            logger.info(
                "AI推理配置: pair=%s, model=%s, threshold=%.4f, patch_size=%d, skip_align=%s, channel_order=%s",
                pair_name,
                "v1" if is_v1 else "v2",
                float(th),
                self.patch_size,
                skip_align,
                ch,
            )

        align_result = None
        aligned_old = old_data
        if not skip_align:
            align_result = self._align_images(new_data, old_data)
            if align_result.success:
                aligned_old = align_result.aligned_old
            else:
                return PipelineResult(
                    pair_name=pair_name,
                    candidates=[],
                    align_result=align_result,
                    error=f"对齐失败: {align_result.error_message}",
                )

        if aligned_old is None:
            aligned_old = old_data

        candidates = self._detect_candidates(
            new_data,
            aligned_old,
            params=self.detection_params,
        )
        logger.info(
            "CV检测 (标准参数): 发现 %d 个候选体 (patch_size=%d)",
            len(candidates),
            self.patch_size,
        )

        if not candidates and ai_available:
            relaxed_params = self._build_relaxed_params()
            candidates = self._detect_candidates(
                new_data,
                aligned_old,
                params=relaxed_params,
            )
            logger.info(
                "CV检测 (放宽参数): 发现 %d 个候选体 (thresh=%d, kill_flat=%s, kill_dipole=%s, topk=%d)",
                len(candidates),
                relaxed_params.thresh,
                relaxed_params.kill_flat,
                relaxed_params.kill_dipole,
                relaxed_params.topk,
            )

        ai_new_data = new_data
        ai_old_data = aligned_old
        if ai_available and is_v1:
            logger.info(
                "v1输入预处理前: %s | %s",
                img_brief_stats("new", new_data),
                img_brief_stats("old", aligned_old),
            )
            ai_new_data = robust_to_uint8(new_data)
            ai_old_data = robust_to_uint8(aligned_old)
            logger.info(
                "v1输入预处理后: %s | %s",
                img_brief_stats("new_u8", ai_new_data),
                img_brief_stats("old_u8", ai_old_data),
            )

        if not candidates and ai_available:
            candidates = self._sliding_window_detect(ai_new_data, ai_old_data)
            logger.info("AI滑动窗口检测: 发现 %d 个候选体", len(candidates))

        if ai_available and candidates:
            engine = self.inference_engine
            if engine is None:
                return PipelineResult(
                    pair_name=pair_name,
                    candidates=candidates,
                    align_result=align_result,
                )

            threshold = engine.threshold
            candidates = self._ai_score(candidates, ai_new_data, ai_old_data)
            candidates = [candidate for candidate in candidates if candidate.ai_score >= threshold]
            logger.info(
                "AI过滤后: %d 个候选体 (阈值=%.4f)",
                len(candidates),
                threshold,
            )

        if self.exclusion_service:
            candidates = self.exclusion_service.check_candidates(
                candidates,
                header=header,
                image_path=image_path,
            )
            candidates = self._exclude_known(candidates)

        candidates.sort(key=lambda candidate: candidate.ai_score, reverse=True)

        return PipelineResult(
            pair_name=pair_name,
            candidates=candidates,
            align_result=align_result,
        )

    def _align_images(self, new_data: np.ndarray, old_data: np.ndarray) -> AlignResult:
        return align(new_data, old_data)

    def _detect_candidates(
        self,
        new_data: np.ndarray,
        old_data: np.ndarray,
        params: DetectionParams,
    ) -> List[Candidate]:
        return detect_candidates(new_data, old_data, params=params)

    def _build_relaxed_params(self) -> DetectionParams:
        return DetectionParams(
            thresh=max(15, self.detection_params.thresh // 3),
            min_area=max(3, self.detection_params.min_area // 2),
            max_area=self.detection_params.max_area * 2,
            sharpness_min=0.5,
            sharpness_max=10.0,
            contrast_min=5,
            edge_margin=self.detection_params.edge_margin,
            dynamic_thresh=False,
            kill_flat=False,
            kill_dipole=False,
            aspect_ratio_max=5.0,
            extent_max=0.95,
            topk=self.detection_params.topk * 3,
        )

    def _sliding_window_detect(
        self,
        new_data: np.ndarray,
        old_data: np.ndarray,
    ) -> List[Candidate]:
        return sliding_window_detect(
            new_data,
            old_data,
            inference_engine=self.inference_engine,
            patch_size=self.patch_size,
            is_v1_model=self._is_v1_model(),
            extract_patch_fn=self._extract_patch,
            prepare_triplet_patch_fn=lambda n, o, x, y, size: self._prepare_triplet_patch(
                n,
                o,
                x,
                y,
                size,
                channel_order=getattr(self.inference_engine, "_channel_order", (0, 1, 2)),
            ),
            nms_candidates_fn=self._nms_candidates,
        )

    def _nms_candidates(
        self,
        candidates: List[Candidate],
        min_dist: int,
    ) -> List[Candidate]:
        return nms_candidates(candidates, min_dist=min_dist)

    def _ai_score(
        self,
        candidates: List[Candidate],
        new_data: np.ndarray,
        old_data: np.ndarray,
    ) -> List[Candidate]:
        if not self.inference_engine or not self.inference_engine.is_ready:
            return candidates

        if not candidates:
            return []

        channel_order = getattr(
            self.inference_engine,
            "_channel_order",
            (0, 1, 2),
        )

        patches = []
        for candidate in candidates:
            patch_3ch = self._prepare_triplet_patch(
                new_data,
                old_data,
                candidate.x,
                candidate.y,
                self.patch_size,
                channel_order=channel_order,
            )
            patches.append(patch_3ch)

        try:
            scores = self.inference_engine.classify_patches(patches)
        except Exception:
            return candidates

        for candidate, score in zip(candidates, scores):
            candidate.ai_score = float(score)

        return candidates

    def _extract_patch(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        size: int,
    ) -> np.ndarray:
        return extract_patch(image, x, y, size)

    def _is_v1_model(self) -> bool:
        if self.inference_engine is None:
            return False
        return getattr(self.inference_engine, "is_v1", False)

    def _prepare_triplet_patch(
        self,
        new_data: np.ndarray,
        old_data: np.ndarray,
        x: int,
        y: int,
        size: int,
        channel_order: tuple = (0, 1, 2),
    ) -> np.ndarray:
        return prepare_triplet_patch(
            new_data,
            old_data,
            x,
            y,
            size,
            is_v1_model=self._is_v1_model(),
            channel_order=channel_order,
            model_input_size=MODEL_INPUT_SIZE,
            extract_patch_fn=self._extract_patch,
        )

    def _exclude_known(self, candidates: List[Candidate]) -> List[Candidate]:
        if not self.exclusion_service:
            return candidates
        return [candidate for candidate in candidates if not candidate.is_known]