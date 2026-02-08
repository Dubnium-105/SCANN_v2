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


# 默认 patch 大小（像素）
DEFAULT_PATCH_SIZE = 32
# 模型输入尺寸
MODEL_INPUT_SIZE = 224


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
        """为候选体计算 AI 分数

        流程:
        1. 为每个候选体提取三元组 patch (new + old + diff)
        2. 归一化到 0-1
        3. 批量推理
        4. 回填分数到候选体

        Args:
            candidates: 候选体列表
            new_data: 新图数据
            old_data: 旧图数据

        Returns:
            更新了 AI 分数的候选体列表
        """
        if not self.inference_engine or not self.inference_engine.is_ready:
            return candidates

        if not candidates:
            return []

        # 1. 提取所有 patch
        patches = []
        for candidate in candidates:
            patch_3ch = self._prepare_triplet_patch(
                new_data,
                old_data,
                candidate.x,
                candidate.y,
                self.patch_size
            )
            patches.append(patch_3ch)

        # 2. 批量推理
        try:
            scores = self.inference_engine.classify_patches(patches)
        except Exception:
            # 推理失败，保持原有分数（0）
            return candidates

        # 3. 回填分数
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
        """从图像中提取 patch（带 padding）

        Args:
            image: 输入图像 (H, W) - 注意：numpy 是 (row, col) = (y, x)
            x: 中心 X 坐标（列）
            y: 中心 Y 坐标（行）
            size: patch 边长

        Returns:
            提取的 patch (size, size)
        """
        half = size // 2

        # 计算边界（注意：numpy 是 (row, col) = (y, x)）
        y0 = max(0, y - half)
        y1 = min(image.shape[0], y + half)
        x0 = max(0, x - half)
        x1 = min(image.shape[1], x + half)

        # 提取实际内容
        patch = np.zeros((size, size), dtype=image.dtype)
        patch_height = y1 - y0
        patch_width = x1 - x0

        # 计算在 patch 中的位置
        patch_y0 = half - (y - y0)
        patch_x0 = half - (x - x0)

        # 复制内容
        patch[patch_y0:patch_y0+patch_height, patch_x0:patch_x0+patch_width] = image[y0:y1, x0:x1]

        return patch

    def _prepare_triplet_patch(
        self,
        new_data: np.ndarray,
        old_data: np.ndarray,
        x: int,
        y: int,
        size: int,
    ) -> np.ndarray:
        """准备三元组 patch (new + old + diff)

        格式: (3, H, W), float32, 0~1

        Args:
            new_data: 新图 (H, W)
            old_data: 旧图 (H, W)
            x: 中心 X 坐标
            y: 中心 Y 坐标
            size: patch 边长

        Returns:
            三通道 patch (3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
        """
        # 提取 new 和 old patch
        patch_new = self._extract_patch(new_data, x, y, size)
        patch_old = self._extract_patch(old_data, x, y, size)

        # 计算差分
        patch_diff = patch_new.astype(np.float32) - patch_old.astype(np.float32)

        # 归一化到 0-1（假设图像已经是线性缩放的）
        # 使用全局归一化
        def normalize(img):
            if img.max() > img.min():
                return (img - img.min()) / (img.max() - img.min())
            return img - img.min()

        patch_new_norm = normalize(patch_new)
        patch_old_norm = normalize(patch_old)
        patch_diff_norm = normalize(patch_diff)

        # 拼接成三通道
        patch_3ch = np.stack([patch_new_norm, patch_old_norm, patch_diff_norm], axis=0).astype(np.float32)

        # 调整大小到模型输入尺寸
        if size != MODEL_INPUT_SIZE:
            from skimage.transform import resize
            patch_3ch = resize(
                patch_3ch,
                (3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                order=1,
                preserve_range=True,
                anti_aliasing=False
            )
            patch_3ch = patch_3ch.astype(np.float32)

        return patch_3ch

    def _exclude_known(self, candidates: List[Candidate]) -> List[Candidate]:
        """排除已知天体"""
        if not self.exclusion_service:
            return candidates

        # 这个方法在 process_pair 中已经调用了 exclusion_service.check_candidates
        # 所以这里只需要过滤掉已知的候选体
        return [c for c in candidates if not c.is_known]
