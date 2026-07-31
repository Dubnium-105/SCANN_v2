"""Detection pipeline and helper orchestration."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from scann.ai.detection_trace import DetectionTrace, image_statistics
from scann.core.candidate_detector import DetectionParams, detect_candidates
from scann.core.image_aligner import align
from scann.core.models import AlignResult, Candidate, Detection
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
    trace: DetectionTrace | None = None


class DetectionPipeline:
    """完整检测管线。"""

    def __init__(
        self,
        detection_params: Optional[DetectionParams] = None,
        inference_engine=None,
        exclusion_service=None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        patch_size: int = DEFAULT_PATCH_SIZE,
        detection_mode: str = "patch",
        hybrid_primary_mode: str = "full_image",
        hybrid_low_confidence: Optional[float] = None,
    ):
        self.detection_params = detection_params or DetectionParams()
        self.inference_engine = inference_engine
        self.exclusion_service = exclusion_service
        self.progress_callback = progress_callback
        self.patch_size = patch_size
        self.detection_mode = self._normalize_detection_mode(detection_mode)
        self.hybrid_primary_mode = self._normalize_hybrid_primary_mode(hybrid_primary_mode)
        self.hybrid_low_confidence = self._normalize_hybrid_low_confidence(
            hybrid_low_confidence,
        )

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
        trace = DetectionTrace(
            pair_name=pair_name,
            detector_version=str(
                getattr(self.detection_params, "detector", "legacy")
                or "legacy"
            ),
            detection_mode=self.detection_mode,
        )
        trace.image_stats = {
            "new": image_statistics(new_data),
            "old": image_statistics(old_data),
        }
        trace.thresholds = {
            "legacy_threshold": int(self.detection_params.thresh),
            "significance_sigma": float(
                getattr(self.detection_params, "significance_sigma", 5.0)
            ),
            "candidate_topk": int(self.detection_params.topk),
            "raw_candidate_limit": int(
                getattr(self.detection_params, "raw_candidate_limit", 500)
            ),
        }
        ai_available = (
            self.inference_engine is not None
            and self.inference_engine.is_ready
        )
        is_v1 = self._is_v1_model() if ai_available else False
        mode = self.detection_mode

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
            logger.info("检测模式: %s", mode)

        align_result = None
        aligned_old = old_data
        if not skip_align:
            align_result = self._align_images(new_data, old_data)
            if align_result.success:
                aligned_old = align_result.aligned_old
                trace.alignment = {
                    "attempted": True,
                    "success": True,
                    "dx": float(getattr(align_result, "dx", 0.0) or 0.0),
                    "dy": float(getattr(align_result, "dy", 0.0) or 0.0),
                }
            else:
                error = f"对齐失败: {align_result.error_message}"
                trace.alignment = {
                    "attempted": True,
                    "success": False,
                    "error": str(align_result.error_message or ""),
                }
                trace.finish(error=error)
                return PipelineResult(
                    pair_name=pair_name,
                    candidates=[],
                    align_result=align_result,
                    error=error,
                    trace=trace,
                )
        else:
            trace.alignment = {
                "attempted": False,
                "success": True,
                "reason": "skip_align",
            }

        if aligned_old is None:
            aligned_old = old_data

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

        candidates: List[Candidate]
        if mode == "full_image":
            candidates = self._dense_full_image_detect(
                new_data,
                aligned_old,
                trace=trace,
            )
            logger.info("AI全图检测: 发现 %d 个候选体", len(candidates))
        elif mode == "hybrid":
            candidates = self._hybrid_detect(
                new_data,
                aligned_old,
                ai_new_data,
                ai_old_data,
                ai_available=ai_available,
                trace=trace,
            )
        else:
            candidates = self._patch_detect(
                new_data,
                aligned_old,
                ai_new_data,
                ai_old_data,
                ai_available=ai_available,
                trace=trace,
            )

        if self.exclusion_service:
            trace.record_stage("pre_exclusion", len(candidates))
            candidates = self.exclusion_service.check_candidates(
                candidates,
                header=header,
                image_path=image_path,
            )
            candidates = self._exclude_known(candidates)
            trace.record_stage("post_exclusion", len(candidates))

        candidates.sort(key=lambda candidate: candidate.ai_score, reverse=True)
        trace.finish(candidates=candidates)

        return PipelineResult(
            pair_name=pair_name,
            candidates=candidates,
            align_result=align_result,
            trace=trace,
        )

    def _align_images(self, new_data: np.ndarray, old_data: np.ndarray) -> AlignResult:
        return align(new_data, old_data, method="siril")

    def _normalize_detection_mode(self, mode: Optional[str]) -> str:
        if mode in {"patch", "full_image", "hybrid"}:
            return str(mode)
        return "patch"

    def _normalize_hybrid_primary_mode(self, mode: Optional[str]) -> str:
        if mode in {"full_image", "patch"}:
            return str(mode)
        return "full_image"

    def _normalize_hybrid_low_confidence(self, threshold: Optional[float]) -> Optional[float]:
        if threshold is None:
            return None
        try:
            value = float(threshold)
        except (TypeError, ValueError):
            return None
        return max(0.0, min(1.0, value))

    def _hybrid_detect(
        self,
        new_data: np.ndarray,
        old_data: np.ndarray,
        ai_new_data: np.ndarray,
        ai_old_data: np.ndarray,
        *,
        ai_available: bool,
        trace: DetectionTrace | None = None,
    ) -> List[Candidate]:
        order = [self.hybrid_primary_mode, "patch" if self.hybrid_primary_mode == "full_image" else "full_image"]
        fallback_reason = ""

        for stage in order:
            if stage == "full_image":
                dense_candidates, dense_status = self._dense_full_image_detect_with_status(
                    new_data,
                    old_data,
                    trace=trace,
                )
                if dense_status == "ok":
                    if self._is_hybrid_low_confidence(dense_candidates):
                        dense_threshold = self._get_hybrid_low_confidence_threshold()
                        best_score = max(candidate.ai_score for candidate in dense_candidates)
                        fallback_reason = (
                            f"low_confidence(max={best_score:.4f}, threshold={dense_threshold:.4f})"
                        )
                        if trace is not None:
                            trace.record_fallback(fallback_reason)
                        logger.info("hybrid 回退 patch: %s", fallback_reason)
                        continue
                    logger.info(
                        "hybrid 选择 full_image 结果: %d 个候选体 (order=%s)",
                        len(dense_candidates),
                        "->".join(order),
                    )
                    return dense_candidates

                fallback_reason = dense_status
                if trace is not None:
                    trace.record_fallback(dense_status)
                logger.info(
                    "hybrid full_image 阶段未产出结果: status=%s, order=%s",
                    dense_status,
                    "->".join(order),
                )
                continue

            patch_candidates = self._patch_detect(
                new_data,
                old_data,
                ai_new_data,
                ai_old_data,
                ai_available=ai_available,
                trace=trace,
            )
            if patch_candidates:
                logger.info(
                    "hybrid 使用 patch 结果: %d 个候选体 (fallback_reason=%s, order=%s)",
                    len(patch_candidates),
                    fallback_reason or "none",
                    "->".join(order),
                )
                return patch_candidates

            fallback_reason = "patch_empty"
            if trace is not None:
                trace.record_fallback(fallback_reason)
            logger.info(
                "hybrid patch 阶段未产出结果: order=%s",
                "->".join(order),
            )

        logger.info(
            "hybrid 检测结束无候选体: order=%s, fallback_reason=%s",
            "->".join(order),
            fallback_reason or "none",
        )
        return []

    def _patch_detect(
        self,
        new_data: np.ndarray,
        old_data: np.ndarray,
        ai_new_data: np.ndarray,
        ai_old_data: np.ndarray,
        *,
        ai_available: bool,
        trace: DetectionTrace | None = None,
    ) -> List[Candidate]:
        stage_started = time.perf_counter()
        candidates = self._detect_candidates(
            new_data,
            old_data,
            params=self.detection_params,
        )
        if trace is not None:
            trace.record_stage(
                "standard",
                len(candidates),
                duration_ms=(time.perf_counter() - stage_started) * 1000.0,
            )
        logger.info(
            "CV检测 (标准参数): 发现 %d 个候选体 (patch_size=%d)",
            len(candidates),
            self.patch_size,
        )

        if not candidates and ai_available:
            relaxed_params = self._build_relaxed_params()
            stage_started = time.perf_counter()
            candidates = self._detect_candidates(
                new_data,
                old_data,
                params=relaxed_params,
            )
            if trace is not None:
                trace.record_stage(
                    "relaxed",
                    len(candidates),
                    duration_ms=(time.perf_counter() - stage_started) * 1000.0,
                )
                trace.record_fallback("standard_empty")
            logger.info(
                "CV检测 (放宽参数): 发现 %d 个候选体 (thresh=%d, kill_flat=%s, kill_dipole=%s, topk=%d)",
                len(candidates),
                relaxed_params.thresh,
                relaxed_params.kill_flat,
                relaxed_params.kill_dipole,
                relaxed_params.topk,
            )

        if not candidates and ai_available:
            stage_started = time.perf_counter()
            candidates = self._sliding_window_detect(ai_new_data, ai_old_data)
            if trace is not None:
                trace.record_stage(
                    "sliding",
                    len(candidates),
                    duration_ms=(time.perf_counter() - stage_started) * 1000.0,
                )
                trace.record_fallback("relaxed_empty")
            logger.info("AI滑动窗口检测: 发现 %d 个候选体", len(candidates))

        if ai_available and candidates:
            engine = self.inference_engine
            if engine is None:
                return candidates

            threshold = engine.threshold
            before_ai = len(candidates)
            stage_started = time.perf_counter()
            candidates = self._ai_score(candidates, ai_new_data, ai_old_data)
            candidates = [candidate for candidate in candidates if candidate.ai_score >= threshold]
            if trace is not None:
                trace.record_stage("pre_ai", before_ai)
                trace.record_stage(
                    "post_ai",
                    len(candidates),
                    duration_ms=(time.perf_counter() - stage_started) * 1000.0,
                )
                trace.thresholds["ai_confidence"] = float(threshold)
            logger.info(
                "AI过滤后: %d 个候选体 (阈值=%.4f)",
                len(candidates),
                threshold,
            )

        return candidates

    def _get_hybrid_low_confidence_threshold(self) -> float:
        if self.hybrid_low_confidence is not None:
            return self.hybrid_low_confidence
        return float(getattr(self.inference_engine, "threshold", 0.5))

    def _is_hybrid_low_confidence(self, candidates: List[Candidate]) -> bool:
        if not candidates:
            return False
        threshold = self._get_hybrid_low_confidence_threshold()
        best_score = max(candidate.ai_score for candidate in candidates)
        return best_score < threshold

    def _dense_full_image_detect(
        self,
        new_data: np.ndarray,
        old_data: np.ndarray,
        *,
        trace: DetectionTrace | None = None,
    ) -> List[Candidate]:
        candidates, _ = self._dense_full_image_detect_with_status(
            new_data,
            old_data,
            trace=trace,
        )
        return candidates

    def _dense_full_image_detect_with_status(
        self,
        new_data: np.ndarray,
        old_data: np.ndarray,
        *,
        trace: DetectionTrace | None = None,
    ) -> tuple[List[Candidate], str]:
        stage_started = time.perf_counter()
        if not self.inference_engine or not self.inference_engine.is_ready:
            logger.warning("full_image 检测跳过：AI 模型不可用")
            if trace is not None:
                trace.record_stage("dense", 0)
                trace.record_fallback("model_unavailable")
            return [], "model_unavailable"

        detect_fn = getattr(self.inference_engine, "detect_dense_full_image", None)
        if not callable(detect_fn):
            logger.warning("full_image 检测跳过：推理引擎不支持 detect_dense_full_image")
            if trace is not None:
                trace.record_stage("dense", 0)
                trace.record_fallback("unsupported_dense_api")
            return [], "unsupported_dense_api"

        try:
            threshold = float(getattr(self.inference_engine, "threshold", 0.5))
            detections = detect_fn(
                new_data,
                old_data,
                score_threshold=threshold,
            )
        except Exception:
            logger.exception("full_image 检测执行失败")
            if trace is not None:
                trace.record_stage(
                    "dense",
                    0,
                    duration_ms=(time.perf_counter() - stage_started) * 1000.0,
                )
                trace.record_fallback("dense_exception")
            return [], "exception"

        candidates: List[Candidate] = []
        for detection in detections:
            if not isinstance(detection, Detection):
                continue
            candidates.append(
                Candidate(
                    x=int(detection.x),
                    y=int(detection.y),
                    ai_score=float(detection.confidence),
                    bbox_x=int(round(float(detection.x) - float(detection.width) / 2.0)),
                    bbox_y=int(round(float(detection.y) - float(detection.height) / 2.0)),
                    bbox_width=int(detection.width),
                    bbox_height=int(detection.height),
                )
            )
        if not candidates:
            if trace is not None:
                trace.record_stage(
                    "dense",
                    0,
                    duration_ms=(time.perf_counter() - stage_started) * 1000.0,
                )
            return [], "empty"
        if trace is not None:
            trace.record_stage(
                "dense",
                len(candidates),
                duration_ms=(time.perf_counter() - stage_started) * 1000.0,
            )
        return candidates, "ok"

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
            detector=str(
                getattr(self.detection_params, "detector", "legacy")
                or "legacy"
            ),
            significance_sigma=max(
                2.5,
                float(
                    getattr(
                        self.detection_params,
                        "significance_sigma",
                        5.0,
                    )
                )
                * 0.75,
            ),
            significance_morphology=bool(
                getattr(
                    self.detection_params,
                    "significance_morphology",
                    True,
                )
            ),
            raw_candidate_limit=int(
                getattr(
                    self.detection_params,
                    "raw_candidate_limit",
                    500,
                )
            ),
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
            details = self.inference_engine.classify_patches_detailed(
                patches,
                structured_features=[
                    vars(candidate.features)
                    for candidate in candidates
                ],
            )
        except Exception:
            return candidates

        for candidate, detail in zip(candidates, details):
            if isinstance(detail, dict):
                candidate.ai_score = float(detail.get("score", 0.0) or 0.0)
                detail_type = str(detail.get("detail_type") or "").strip().lower()
                if detail_type:
                    setattr(candidate, "detail_type", detail_type)
                label = str(detail.get("label") or "").strip().lower()
                if label:
                    setattr(candidate, "label", label)
                predicted_class = str(detail.get("predicted_class") or "").strip().lower()
                if predicted_class:
                    setattr(candidate, "predicted_label", predicted_class)
            else:
                candidate.ai_score = float(detail)

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
