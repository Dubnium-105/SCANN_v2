"""检测管线服务

职责:
- 编排完整的检测流程:
  对齐 → 候选检测 → AI 评分 → 已知排除 → 排序输出
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from scann.core.candidate_detector import DetectionParams, detect_candidates
from scann.core.image_aligner import align
from scann.core.models import AlignResult, Candidate


@dataclass
class PipelineResult:
    """管线处理结果"""
    pair_name: str
    candidates: List[Candidate]
    align_result: Optional[AlignResult] = None
    error: str = ""


class DetectionPipeline:
    """完整检测管线

    流程:
    1. 对齐新旧图
    2. 检测候选体
    3. AI 评分
    4. 已知天体排除
    5. 排序输出
    """

    def __init__(
        self,
        detection_params: Optional[DetectionParams] = None,
        inference_engine=None,
        exclusion_service=None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ):
        self.detection_params = detection_params or DetectionParams()
        self.inference_engine = inference_engine
        self.exclusion_service = exclusion_service
        self.progress_callback = progress_callback

    def process_pair(
        self,
        pair_name: str,
        new_data: np.ndarray,
        old_data: np.ndarray,
        skip_align: bool = False,
    ) -> PipelineResult:
        """处理单对图像

        Args:
            pair_name: 配对名称
            new_data: 新图数据 (灰度)
            old_data: 旧图数据 (灰度)
            skip_align: 跳过对齐 (已预对齐)

        Returns:
            处理结果
        """
        # 1. 对齐
        align_result = None
        aligned_old = old_data
        if not skip_align:
            align_result = align(new_data, old_data)
            if align_result.success:
                aligned_old = align_result.aligned_old
            else:
                return PipelineResult(
                    pair_name=pair_name,
                    candidates=[],
                    align_result=align_result,
                    error=f"对齐失败: {align_result.error_message}",
                )

        # 2. 候选检测
        candidates = detect_candidates(
            new_data, aligned_old, params=self.detection_params
        )

        # 3. AI 评分 (如果引擎可用)
        if self.inference_engine and self.inference_engine.is_ready and candidates:
            candidates = self._ai_score(candidates, new_data, aligned_old)

        # 4. 已知排除 (如果服务可用)
        if self.exclusion_service:
            candidates = self._exclude_known(candidates)

        # 5. 按 AI 分数排序
        candidates.sort(key=lambda c: c.ai_score, reverse=True)

        return PipelineResult(
            pair_name=pair_name,
            candidates=candidates,
            align_result=align_result,
        )

    def _ai_score(
        self,
        candidates: List[Candidate],
        new_data: np.ndarray,
        old_data: np.ndarray,
    ) -> List[Candidate]:
        """为候选体计算 AI 分数"""
        # TODO: 提取 patch → 推理 → 回填分数
        return candidates

    def _exclude_known(self, candidates: List[Candidate]) -> List[Candidate]:
        """排除已知天体"""
        # TODO: 调用排除服务
        return [c for c in candidates if not c.is_known]
