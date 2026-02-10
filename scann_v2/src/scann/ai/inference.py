"""AI 推理引擎

职责:
- GPU 推理管理
- 显存控制 ≤ 8GB
- CUDA 多线程并行
- 分块推理防止 OOM
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

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

        # 尝试读取保存的阈值
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            self._threshold = ckpt.get("threshold", 0.5)
            # 如果 checkpoint 中有格式元数据，记录下来
            if "model_format" in ckpt:
                self._model_format = ModelFormat(ckpt["model_format"])

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    @property
    def threshold(self) -> float:
        return self._threshold

    @torch.no_grad()
    def classify_patches(
        self,
        patches: List[np.ndarray],
        normalize_mean: tuple = (0.26, 0.27, 0.27),
        normalize_std: tuple = (0.09, 0.11, 0.11),
    ) -> List[float]:
        """批量分类裁剪图

        Args:
            patches: 裁剪图列表, 每个 shape=(3, H, W), float32, 0~1
            normalize_mean: 归一化均值
            normalize_std: 归一化标准差

        Returns:
            正类概率列表
        """
        if not self.is_ready:
            raise RuntimeError("模型未加载")
        if not patches:
            return []

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
                with torch.amp.autocast("cuda"):
                    logits = self.model(stack)
            else:
                logits = self.model(stack)

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())

        return all_probs

    def detect_full_image(
        self,
        image: np.ndarray,
        patch_size: int = 224,
        stride: int = 112,
        iou_threshold: float = 0.5,
    ) -> List[Detection]:
        """全图检测 (v2 新功能)

        使用滑动窗口在整幅图像上进行检测，并使用 NMS 合并重叠结果

        Args:
            image: 完整天文图像 (H, W)
            patch_size: 滑动窗口大小（默认 224）
            stride: 滑动步长（默认 112，50% 重叠）
            iou_threshold: NMS IoU 阈值（默认 0.5）

        Returns:
            检测结果列表
        """
        from skimage.transform import resize

        if self.model is None:
            return []

        height, width = image.shape[:2]

        # 如果图像小于窗口大小，先进行填充
        if height < patch_size or width < patch_size:
            padded = np.zeros((patch_size, patch_size), dtype=image.dtype)
            padded[:height, :width] = image
            image = padded
            height, width = patch_size, patch_size

        # 收集所有窗口的检测结果
        all_detections = []

        # 滑动窗口遍历
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                # 提取窗口
                patch = image[y:y+patch_size, x:x+patch_size]

                # 归一化
                if patch.max() > patch.min():
                    patch = (patch - patch.min()) / (patch.max() - patch.min())

                # 转换为张量并添加 batch 和 channel 维度
                patch_tensor = torch.from_numpy(patch).float()
                patch_tensor = patch_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

                # 重复为 3 通道（如果模型期望 RGB 输入）
                patch_tensor = patch_tensor.repeat(1, 3, 1, 1)

                # 推理
                with torch.no_grad():
                    patch_tensor = patch_tensor.to(self.device)
                    output = self.model(patch_tensor)

                # 获取概率
                probs = torch.softmax(output, dim=1)[0, 1].cpu().item()

                # 如果置信度超过阈值，添加到结果
                if probs > self._threshold:
                    # 计算窗口中心坐标
                    center_x = int(x + patch_size / 2.0)
                    center_y = int(y + patch_size / 2.0)

                    # 边界框大小
                    bbox_width = patch_size
                    bbox_height = patch_size

                    detection = Detection(
                        x=center_x,
                        y=center_y,
                        width=bbox_width,
                        height=bbox_height,
                        confidence=probs,
                        marker_type=MarkerType.BOUNDING_BOX
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

    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
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

