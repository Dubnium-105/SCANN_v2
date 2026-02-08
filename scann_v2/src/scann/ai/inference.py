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

from scann.core.models import Candidate, Detection


@dataclass
class InferenceConfig:
    """推理配置"""
    batch_size: int = 64
    max_memory_mb: int = 8000  # 8GB 显存限制
    use_amp: bool = True       # 混合精度
    device: str = "auto"       # "auto", "cuda:0", "cpu"


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
        """加载模型"""
        from scann.ai.model import SCANNClassifier
        self.model = SCANNClassifier.load_from_checkpoint(path, self.device)

        # 尝试读取保存的阈值
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            self._threshold = ckpt.get("threshold", 0.5)

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
    ) -> List[Detection]:
        """全图检测 (v2 新功能)

        TODO: 实现滑动窗口 + 检测模型

        Args:
            image: 完整天文图像

        Returns:
            检测结果列表
        """
        # 占位实现：后续替换为真正的全图检测逻辑
        raise NotImplementedError("全图检测功能尚在开发中")
