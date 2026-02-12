"""检测管线服务

职责:
- 编排完整的检测流程:
  对齐 → 候选检测 → AI 评分 → 已知排除 → 排序输出
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from scann.core.candidate_detector import DetectionParams, detect_candidates
from scann.core.image_aligner import align
from scann.core.models import AlignResult, Candidate

logger = logging.getLogger(__name__)


# 默认 patch 大小（像素）— V1 训练使用 80x80 切片
DEFAULT_PATCH_SIZE = 80
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

        当 AI 模型可用时，采用宽松 CV + AI 精筛策略:
        - 第一轮: 使用用户参数进行 CV 检测
        - 若无候选且 AI 可用: 放宽 CV 参数重试 (降阈值、关滤波器)
        - 若仍无候选: 使用滑动窗口让 AI 直接扫描全图
        - AI 评分后过滤掉低于阈值的候选体

        Args:
            pair_name: 配对名称
            new_data: 新图数据 (灰度)
            old_data: 旧图数据 (灰度)
            skip_align: 跳过对齐 (已预对齐)

        Returns:
            处理结果
        """
        ai_available = (
            self.inference_engine is not None
            and self.inference_engine.is_ready
        )

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

        # 2. 候选检测 (CV)
        candidates = detect_candidates(
            new_data, aligned_old, params=self.detection_params
        )
        logger.info(
            "CV检测 (标准参数): 发现 %d 个候选体 (patch_size=%d)",
            len(candidates), self.patch_size
        )

        # 2b. 若 CV 无结果且 AI 可用，放宽 CV 参数重试
        if not candidates and ai_available:
            relaxed_params = self._build_relaxed_params()
            candidates = detect_candidates(
                new_data, aligned_old, params=relaxed_params
            )
            logger.info(
                "CV检测 (放宽参数): 发现 %d 个候选体 "
                "(thresh=%d, kill_flat=%s, kill_dipole=%s, topk=%d)",
                len(candidates),
                relaxed_params.thresh,
                relaxed_params.kill_flat,
                relaxed_params.kill_dipole,
                relaxed_params.topk,
            )

        # 2c. 若仍无结果且 AI 可用，使用滑动窗口全图扫描
        if not candidates and ai_available:
            candidates = self._sliding_window_detect(
                new_data, aligned_old
            )
            logger.info(
                "AI滑动窗口检测: 发现 %d 个候选体", len(candidates)
            )

        # 3. AI 评分 + 阈值过滤
        if ai_available and candidates:
            threshold = self.inference_engine.threshold
            candidates = self._ai_score(candidates, new_data, aligned_old)
            # 按阈值过滤，仅保留 AI 认为 "真" 的候选
            candidates = [c for c in candidates if c.ai_score >= threshold]
            logger.info(
                "AI过滤后: %d 个候选体 (阈值=%.4f)",
                len(candidates), threshold,
            )

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

    def _build_relaxed_params(self) -> DetectionParams:
        """构建放宽的 CV 检测参数 (用于 AI 模型兜底)

        策略: 大幅降低阈值、关闭严格过滤器，让更多潜在候选通过，
        交给 AI 做最终判断。
        """
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
        """使用滑动窗口 + AI 在全图上检测候选体

        将图像按 patch_size 大小滑动切分，每个窗口生成三元组 patch
        送入 AI 分类器，将高于阈值的窗口中心作为候选体返回。

        Returns:
            候选体列表 (ai_score 已填入)
        """
        from scann.core.models import CandidateFeatures

        if not self.inference_engine or not self.inference_engine.is_ready:
            return []

        h, w = new_data.shape[:2]
        size = self.patch_size
        stride = max(size // 2, 1)  # 50% 重叠
        threshold = self.inference_engine.threshold
        channel_order = getattr(
            self.inference_engine, '_channel_order', (0, 1, 2)
        )

        # 收集所有窗口中心及对应的 patch
        centers = []
        patches = []
        half = size // 2

        for cy in range(half, h - half, stride):
            for cx in range(half, w - half, stride):
                patch_3ch = self._prepare_triplet_patch(
                    new_data, old_data, cx, cy, size,
                    channel_order=channel_order,
                )
                patches.append(patch_3ch)
                centers.append((cx, cy))

        if not patches:
            return []

        # 批量推理
        try:
            scores = self.inference_engine.classify_patches(patches)
        except Exception as e:
            logger.warning("滑动窗口推理失败: %s", e)
            return []

        # 收集超过阈值的窗口
        candidates = []
        for (cx, cy), score in zip(centers, scores):
            if score >= threshold:
                candidates.append(Candidate(
                    x=cx, y=cy,
                    features=CandidateFeatures(),
                    ai_score=float(score),
                ))

        # 简单 NMS: 合并过于接近的候选体 (保留分数最高的)
        if len(candidates) > 1:
            candidates = self._nms_candidates(candidates, min_dist=size // 2)

        return candidates

    def _nms_candidates(
        self,
        candidates: List[Candidate],
        min_dist: int,
    ) -> List[Candidate]:
        """对候选体做简单的空间 NMS

        按 ai_score 降序，依次保留候选体，移除距离过近的低分候选。
        """
        candidates.sort(key=lambda c: c.ai_score, reverse=True)
        keep = []
        for c in candidates:
            too_close = False
            for k in keep:
                dist = ((c.x - k.x) ** 2 + (c.y - k.y) ** 2) ** 0.5
                if dist < min_dist:
                    too_close = True
                    break
            if not too_close:
                keep.append(c)
        return keep

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

        # 获取通道顺序（V1 模型可能有不同的通道顺序）
        channel_order = getattr(
            self.inference_engine, '_channel_order', (0, 1, 2)
        )

        # 1. 提取所有 patch
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

    def _is_v1_model(self) -> bool:
        """判断当前推理引擎是否使用 V1 模型"""
        if self.inference_engine is None:
            return False
        return getattr(self.inference_engine, 'is_v1', False)

    def _prepare_triplet_patch(
        self,
        new_data: np.ndarray,
        old_data: np.ndarray,
        x: int,
        y: int,
        size: int,
        channel_order: tuple = (0, 1, 2),
    ) -> np.ndarray:
        """准备三元组 patch

        V1 模型:
          - 通道语义: [Diff(A), New(B), Ref(C)]，其中 Ref=Old
          - 归一化: uint8 / 255.0 (简单缩放，保持原始亮度分布)
          - Diff 是 new - old 的 clip(0,255) 结果

        V2 模型:
          - 通道语义: [New, Old, Diff]，按 channel_order 重排
          - 归一化: 每通道独立 min-max 到 [0, 1]

        格式: (3, H, W), float32, 0~1

        Args:
            new_data: 新图 (H, W)
            old_data: 旧图 (H, W)
            x: 中心 X 坐标
            y: 中心 Y 坐标
            size: patch 边长
            channel_order: 通道顺序 (仅 V2 使用，默认 (0,1,2) 即 new/old/diff)

        Returns:
            三通道 patch (3, size, size) 或 (3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
        """
        # 提取 new 和 old patch
        patch_new = self._extract_patch(new_data, x, y, size)
        patch_old = self._extract_patch(old_data, x, y, size)

        if self._is_v1_model():
            # ── V1 兼容模式 ──
            # V1 训练数据的三联图: [左=Diff(A), 中=New(B), 右=Ref(C)]
            # Diff = clip(new - old, 0, 255), New = new, Ref = old
            # 归一化方式: uint8 / 255.0 (与 ToTensor() 行为一致)
            patch_diff = np.clip(
                patch_new.astype(np.float32) - patch_old.astype(np.float32),
                0, 255
            ).astype(np.uint8)

            # 确保 uint8 范围
            p_diff = patch_diff.astype(np.float32) / 255.0
            p_new = patch_new.astype(np.float32) / 255.0
            p_old = patch_old.astype(np.float32) / 255.0  # Ref = Old

            # V1 通道顺序固定: [Diff, New, Ref(Old)]
            patch_3ch = np.stack([p_diff, p_new, p_old], axis=0).astype(np.float32)
        else:
            # ── V2 模式 ──
            # 计算差分
            patch_diff = patch_new.astype(np.float32) - patch_old.astype(np.float32)

            # 每通道独立 min-max 归一化到 [0, 1]
            def normalize(img):
                if img.max() > img.min():
                    return (img - img.min()) / (img.max() - img.min())
                return img - img.min()

            patch_new_norm = normalize(patch_new)
            patch_old_norm = normalize(patch_old)
            patch_diff_norm = normalize(patch_diff)

            # 三个通道: [new, old, diff]，然后按 channel_order 重排
            channels = [patch_new_norm, patch_old_norm, patch_diff_norm]
            ordered_channels = [channels[i] for i in channel_order]
            patch_3ch = np.stack(ordered_channels, axis=0).astype(np.float32)

        # 调整大小到模型输入尺寸（V1 训练时 80→224 通过 Resize）
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
