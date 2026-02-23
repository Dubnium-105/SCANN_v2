"""AI 推理引擎

职责:
- GPU 推理管理
- 显存控制 ≤ 8GB
- CUDA 多线程并行
- 分块推理防止 OOM
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
from torchvision import transforms

from scann.core.models import Candidate, Detection, MarkerType


@dataclass
class InferenceConfig:
    """推理配置"""
    batch_size: int = 64
    max_memory_mb: int = 8000  # 8GB 显存限制
    use_amp: bool = True       # 混合精度
    device: str = "auto"       # "auto", "cuda:0", "cpu"
    model_format: str = "auto" # "auto", "v1_classifier", "v2_classifier"


class InferenceEngine:
    """AI 推理引擎"""

    def __init__(
        self,
        model_path: str,
        config: Optional[InferenceConfig] = None,
    ):
        self.config = config or InferenceConfig()
        self.device = self._resolve_device()
        self.model = None
        self._threshold = 0.5
        self._channel_order = (0, 1, 2)  # 默认通道顺序

        if model_path:
            self._load_model(model_path)

    def _resolve_device(self) -> torch.device:
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            return torch.device("cpu")
        return torch.device(self.config.device)

    def _load_model(self, path: str) -> None:
        """加载模型 (自动检测 v1/v2 格式)"""
        import logging
        _logger = logging.getLogger(__name__)
        from scann.ai.model import ModelFormat, SCANNClassifier

        # 解析模型格式
        try:
            fmt = ModelFormat(self.config.model_format)
        except ValueError:
            fmt = ModelFormat.AUTO

        self.model = SCANNClassifier.load_from_checkpoint(
            path, self.device, model_format=fmt
        )
        self._model_format = fmt

        # 尝试读取保存的阈值和元数据
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            # 确定实际的模型格式
            if "model_format" in ckpt:
                # V2 checkpoint 中有明确的格式元数据
                try:
                    self._model_format = ModelFormat(ckpt["model_format"])
                except ValueError:
                    pass
            elif fmt == ModelFormat.AUTO:
                # 没有元数据，通过 state_dict 键名推断
                state_dict = ckpt.get("state") or ckpt.get("model_state") or {}
                if isinstance(state_dict, dict) and state_dict:
                    from scann.ai.model import detect_model_format
                    self._model_format = detect_model_format(state_dict)

            # 读取阈值：优先 threshold (v2)，其次 t_recall (v1 训练脚本)
            if "threshold" in ckpt and ckpt["threshold"] is not None:
                self._threshold = float(ckpt["threshold"])
            elif "t_recall" in ckpt and ckpt["t_recall"] is not None:
                self._threshold = float(ckpt["t_recall"])
                _logger.info("使用 V1 阈值 (t_recall): %.4f", self._threshold)
            else:
                self._threshold = 0.5

            # 读取 V1 的通道顺序 (order 字段, 如 "0,1,2")
            order_str = ckpt.get("order")
            if order_str and isinstance(order_str, str):
                try:
                    self._channel_order = tuple(
                        int(x.strip()) for x in order_str.split(",")
                    )
                    _logger.info("使用 V1 通道顺序: %s", self._channel_order)
                except (ValueError, TypeError):
                    self._channel_order = (0, 1, 2)
            else:
                self._channel_order = (0, 1, 2)

    # V1 训练时使用的归一化常数 (来自 train_triplet_resnet_augmented.py)
    V1_NORMALIZE_MEAN = (0.2637, 0.2819, 0.2838)
    V1_NORMALIZE_STD = (0.0890, 0.1226, 0.1283)
    # V2 默认归一化常数
    V2_NORMALIZE_MEAN = (0.26, 0.27, 0.27)
    V2_NORMALIZE_STD = (0.09, 0.11, 0.11)

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def model_format(self):
        """当前加载的模型格式"""
        return self._model_format

    @property
    def is_v1(self) -> bool:
        """当前是否为 V1 模型"""
        from scann.ai.model import ModelFormat
        return self._model_format == ModelFormat.V1_CLASSIFIER

    @property
    def channel_order(self) -> tuple:
        """模型训练时使用的通道顺序"""
        return self._channel_order

    @torch.no_grad()
    def classify_patches(
        self,
        patches: List[np.ndarray],
        normalize_mean: Optional[tuple] = None,
        normalize_std: Optional[tuple] = None,
    ) -> List[float]:
        """批量分类裁剪图

        Args:
            patches: 裁剪图列表, 每个 shape=(3, H, W), float32, 0~1
            normalize_mean: 归一化均值 (None=根据模型格式自动选择)
            normalize_std: 归一化标准差 (None=根据模型格式自动选择)

        Returns:
            正类概率列表
        """
        if not self.is_ready:
            raise RuntimeError("模型未加载")
        if not patches:
            return []
        assert self.model is not None

        # 根据模型格式自动选择归一化常数
        if normalize_mean is None or normalize_std is None:
            if self.is_v1:
                normalize_mean = self.V1_NORMALIZE_MEAN
                normalize_std = self.V1_NORMALIZE_STD
            else:
                normalize_mean = self.V2_NORMALIZE_MEAN
                normalize_std = self.V2_NORMALIZE_STD

        norm = transforms.Normalize(list(normalize_mean), list(normalize_std))
        resize = transforms.Resize((224, 224), antialias=True)

        all_probs = []
        batch_size = self.config.batch_size

        for i in range(0, len(patches), batch_size):
            batch_raw = patches[i : i + batch_size]
            tensors = []
            for p in batch_raw:
                t = torch.from_numpy(p).float()
                t = resize(t)
                t = norm(t)
                tensors.append(t)

            stack = torch.stack(tensors).to(self.device)

            if self.config.use_amp and self.device.type == "cuda":
                amp_mod = getattr(torch, "amp", None)
                if amp_mod is not None and hasattr(amp_mod, "autocast"):
                    with amp_mod.autocast("cuda"):
                        logits = self.model(stack)
                else:
                    with torch.cuda.amp.autocast():
                        logits = self.model(stack)
            else:
                logits = self.model(stack)

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())

        return all_probs

    @staticmethod
    def _robust_to_uint8(image: np.ndarray) -> np.ndarray:
        """将任意动态范围图像稳健映射到 uint8。"""
        if image.dtype == np.uint8:
            return image

        img = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if img.size == 0:
            return np.zeros_like(img, dtype=np.uint8)

        p_low = float(np.percentile(img, 1.0))
        p_high = float(np.percentile(img, 99.0))
        if not np.isfinite(p_low) or not np.isfinite(p_high) or p_high <= p_low:
            p_low = float(np.min(img))
            p_high = float(np.max(img))
            if p_high <= p_low:
                return np.zeros_like(img, dtype=np.uint8)

        scaled = (img - p_low) / (p_high - p_low)
        scaled = np.clip(scaled, 0.0, 1.0)
        return (scaled * 255.0).astype(np.uint8)

    @staticmethod
    def _extract_patch_2d(image: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
        """从 2D 图像中按中心点提取 patch，越界补零。"""
        half = size // 2
        y0 = max(0, y - half)
        y1 = min(image.shape[0], y + half)
        x0 = max(0, x - half)
        x1 = min(image.shape[1], x + half)

        patch = np.zeros((size, size), dtype=image.dtype)
        ph = y1 - y0
        pw = x1 - x0
        py0 = half - (y - y0)
        px0 = half - (x - x0)
        patch[py0:py0 + ph, px0:px0 + pw] = image[y0:y1, x0:x1]
        return patch

    def _prepare_v1_triplet_patch(
        self,
        new_u8: np.ndarray,
        old_u8: np.ndarray,
        x: int,
        y: int,
        patch_size: int,
    ) -> np.ndarray:
        """按 v1 训练语义构造单个三联图 patch: [Diff, New, Ref(Old)]。"""
        p_new = self._extract_patch_2d(new_u8, x, y, patch_size)
        p_old = self._extract_patch_2d(old_u8, x, y, patch_size)
        p_diff = np.clip(
            p_new.astype(np.float32) - p_old.astype(np.float32),
            0,
            255,
        ).astype(np.uint8)

        c_diff = p_diff.astype(np.float32) / 255.0
        c_new = p_new.astype(np.float32) / 255.0
        c_old = p_old.astype(np.float32) / 255.0

        order = self._channel_order
        if (
            not isinstance(order, (tuple, list))
            or len(order) != 3
            or sorted(order) != [0, 1, 2]
        ):
            order = (0, 1, 2)

        channels = [c_diff, c_new, c_old]
        ordered = [channels[i] for i in order]
        return np.stack(ordered, axis=0).astype(np.float32)

    @staticmethod
    def _pad_to_min_size(image: np.ndarray, min_h: int, min_w: int) -> np.ndarray:
        """将 2D 图像右下补零到至少 (min_h, min_w)。"""
        h, w = image.shape[:2]
        if h >= min_h and w >= min_w:
            return image
        out = np.zeros((max(h, min_h), max(w, min_w)), dtype=image.dtype)
        out[:h, :w] = image
        return out

    def detect_full_image(
        self,
        image: np.ndarray,
        old_image: Optional[np.ndarray] = None,
        patch_size: int = 224,
        stride: int = 112,
        iou_threshold: float = 0.5,
    ) -> List[Detection]:
        """全图检测 (v2 新功能)

        使用滑动窗口在整幅图像上进行检测，并使用 NMS 合并重叠结果。

        兼容逻辑:
        - v1 模型: 需要 new/old 两张图，按 [Diff, New, Ref] 生成三联图输入。
        - v2 模型: 保持单图滑窗（灰度复制为 3 通道）。

        Args:
            image: 完整天文图像（v1 时表示 new 图，v2 时表示单图输入）
            old_image: v1 模式下的 old 图（可选；缺失时退化为零背景）
            patch_size: 滑动窗口大小（默认 224）
            stride: 滑动步长（默认 112，50% 重叠）
            iou_threshold: NMS IoU 阈值（默认 0.5）

        Returns:
            检测结果列表
        """
        if self.model is None:
            return []

        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        height, width = image.shape[:2]

        # 如果图像小于窗口大小，先进行填充
        if height < patch_size or width < patch_size:
            padded = np.zeros((patch_size, patch_size), dtype=image.dtype)
            padded[:height, :width] = image
            image = padded
            height, width = patch_size, patch_size

        centers = []
        patches: List[np.ndarray] = []

        if self.is_v1:
            if old_image is None:
                old_work = np.zeros_like(image)
            else:
                old_work = np.nan_to_num(old_image, nan=0.0, posinf=0.0, neginf=0.0)

            # v1 也保证两张图都至少达到 patch_size，避免小图没有滑窗
            image = self._pad_to_min_size(image, patch_size, patch_size)
            old_work = self._pad_to_min_size(old_work, patch_size, patch_size)
            height, width = image.shape[:2]

            # 尺寸不一致时裁到公共区域（与对齐后输入语义一致）
            h2, w2 = old_work.shape[:2]
            h = min(height, h2)
            w = min(width, w2)
            new_work = image[:h, :w]
            old_work = old_work[:h, :w]

            new_u8 = self._robust_to_uint8(new_work)
            old_u8 = self._robust_to_uint8(old_work)

            half = patch_size // 2
            for cy in range(half, h - half + 1, stride):
                for cx in range(half, w - half + 1, stride):
                    patch_3ch = self._prepare_v1_triplet_patch(
                        new_u8,
                        old_u8,
                        cx,
                        cy,
                        patch_size,
                    )
                    patches.append(patch_3ch)
                    centers.append((cx, cy))
        else:
            # v2: 单图滑窗，窗口独立 min-max，再复制为 3 通道
            for y in range(0, height - patch_size + 1, stride):
                for x in range(0, width - patch_size + 1, stride):
                    patch = image[y:y + patch_size, x:x + patch_size].astype(np.float32)
                    if patch.max() > patch.min():
                        patch = (patch - patch.min()) / (patch.max() - patch.min())
                    else:
                        patch = patch - patch.min()

                    patch_3ch = np.stack([patch, patch, patch], axis=0).astype(np.float32)
                    patches.append(patch_3ch)
                    centers.append((int(x + patch_size / 2.0), int(y + patch_size / 2.0)))

        if not patches:
            return []

        probs = self.classify_patches(patches)

        # 收集所有窗口的检测结果
        all_detections = []
        for (center_x, center_y), score in zip(centers, probs):
            if score > self._threshold:
                detection = Detection(
                    x=int(center_x),
                    y=int(center_y),
                    width=patch_size,
                    height=patch_size,
                    confidence=float(score),
                    marker_type=MarkerType.BOUNDING_BOX,
                )
                all_detections.append(detection)

        # 应用 NMS 合并重叠检测
        if len(all_detections) > 1:
            all_detections = self._nms(all_detections, iou_threshold)

        return all_detections

    def _nms(self, detections: List[Detection], iou_threshold: float) -> List[Detection]:
        """非极大值抑制（Non-Maximum Suppression）

        合并重叠的检测结果，保留置信度最高的

        Args:
            detections: 检测结果列表
            iou_threshold: IoU 阈值

        Returns:
            合并后的检测结果列表
        """
        if len(detections) == 0:
            return []

        # 按置信度排序（降序）
        sorted_detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

        keep = []
        while len(sorted_detections) > 0:
            # 保留置信度最高的检测
            current = sorted_detections.pop(0)
            keep.append(current)

            # 计算当前检测的边界框（从中心点和宽高计算）
            bbox1 = [
                current.x - current.width // 2,
                current.y - current.height // 2,
                current.x + current.width // 2,
                current.y + current.height // 2
            ]

            # 移除与当前检测重叠的其他检测
            remaining = []
            for d in sorted_detections:
                bbox2 = [
                    d.x - d.width // 2,
                    d.y - d.height // 2,
                    d.x + d.width // 2,
                    d.y + d.height // 2
                ]
                iou = self._calculate_iou(bbox1, bbox2)
                if iou < iou_threshold:
                    remaining.append(d)

            sorted_detections = remaining

        return keep

    def _calculate_iou(self, bbox1: Sequence[float], bbox2: Sequence[float]) -> float:
        """计算两个边界框的 IoU (Intersection over Union)

        Args:
            bbox1: 第一个边界框 [x1, y1, x2, y2]
            bbox2: 第二个边界框 [x1, y1, x2, y2]

        Returns:
            IoU 值 (0-1)
        """
        # 计算交集区域
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)

        # 计算并集区域
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

