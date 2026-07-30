"""AI 模型定义模块

职责:
- SCANNDetector: 全图目标检测模型 (新版)
- SCANNClassifier: 兼容 v1 的裁剪图分类器
- 模型格式兼容 (v1/v2 自动检测与转换)
- 显存控制 ≤ 8GB
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

from scann.ai.cache_paths import configure_torch_cache

# 设置PyTorch模型下载路径到项目内（必须在导入torch之前设置）
try:
    model_file = Path(__file__).resolve()
    # model.py 位于 scann_v2/src/scann/ai/model.py，需要向上4级到 scann_v2/
    scann_v2_root = model_file.parent.parent.parent.parent
    model_cache_dir = configure_torch_cache(scann_v2_root)
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置环境变量
    os.environ['TORCH_HOME'] = str(model_cache_dir)
    os.environ['TORCH_HUB_DIR'] = str(model_cache_dir)
except Exception:
    pass

import torch
import torch.nn as nn
from torchvision import models

from scann.ai.device_utils import resolve_device

logger = logging.getLogger(__name__)

# 确认缓存目录
torch.hub.set_dir(str(model_cache_dir))
logger.info(f"PyTorch模型缓存目录: {model_cache_dir}")


# ─────────────────────── 模型格式枚举 ───────────────────────


class ModelFormat(Enum):
    """模型格式标识

    用于区分不同版本/框架的模型文件格式，支持未来扩展。

    - V1_CLASSIFIER: 原始 ResNet18 直接保存 (无 backbone. 前缀)
    - V2_CLASSIFIER: 包装后的 SCANNClassifier (带 backbone. 前缀)
    - AUTO: 自动检测格式
    """
    V1_CLASSIFIER = "v1_classifier"
    V2_CLASSIFIER = "v2_classifier"
    AUTO = "auto"


def detect_model_format(state_dict: Dict[str, torch.Tensor]) -> ModelFormat:
    """自动检测 state_dict 的模型格式

    通过分析键名前缀来判断模型版本:
    - 含 backbone. 前缀 → V2_CLASSIFIER
    - 含 conv1/layer1 等原始 ResNet 键 → V1_CLASSIFIER
    - 空字典 → AUTO (无法判断)

    Args:
        state_dict: 模型参数字典

    Returns:
        检测到的 ModelFormat
    """
    if not state_dict:
        return ModelFormat.AUTO

    keys = list(state_dict.keys())
    # 先清除 module. 前缀再判断
    cleaned = [k[7:] if k.startswith("module.") else k for k in keys]

    has_backbone = any(k.startswith("backbone.") for k in cleaned)
    has_raw_resnet = any(
        k.startswith(("conv1.", "bn1.", "layer1.", "layer2.", "layer3.", "layer4.", "fc."))
        for k in cleaned
    )

    if has_backbone:
        return ModelFormat.V2_CLASSIFIER
    if has_raw_resnet:
        return ModelFormat.V1_CLASSIFIER
    return ModelFormat.AUTO


def convert_state_dict_v1_to_v2(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """将 v1 格式 state_dict 转换为 v2 格式

    v1 键: conv1.weight, layer1.0.conv1.weight, fc.weight, ...
    v2 键: backbone.conv1.weight, backbone.layer1.0.conv1.weight, backbone.fc.weight, ...

    同时处理 module. 前缀 (DataParallel 产生)。

    Args:
        state_dict: v1 格式的参数字典

    Returns:
        v2 格式的参数字典
    """
    converted = {}
    for k, v in state_dict.items():
        # 跳过非权重元数据
        if not isinstance(v, torch.Tensor):
            continue
        # 移除 module. 前缀
        name = k[7:] if k.startswith("module.") else k
        # 添加 backbone. 前缀
        if not name.startswith("backbone."):
            name = f"backbone.{name}"
        converted[name] = v
    return converted


def _convert_state_dict_v2_to_v1(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """将 v2 格式 state_dict 转换为 v1 格式 (用于保存 v1 兼容文件)

    Args:
        state_dict: v2 格式的参数字典

    Returns:
        v1 格式的参数字典 (移除 backbone. 前缀)
    """
    converted = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        if name.startswith("backbone."):
            name = name[len("backbone."):]
        converted[name] = v
    return converted


class SCANNClassifier(nn.Module):
    """v1 兼容分类器 - 基于 ResNet-18 的裁剪图分类

    输入: [B, 3, 224, 224] (三通道: diff/new/ref)
    输出: [B, 2] (二分类 logits)
    """

    SUPPORTED_BACKBONES = ("ResNet18", "ResNet34", "ResNet50", "ViT_B_16")

    @staticmethod
    def _normalize_backbone_name(backbone_name: str) -> str:
        name = str(backbone_name or "").strip()
        if not name or name.lower() == "auto":
            return "ResNet18"

        normalized = name.replace("-", "_").upper()
        mapping = {
            "RESNET18": "ResNet18",
            "RESNET34": "ResNet34",
            "RESNET50": "ResNet50",
            "VIT_B_16": "ViT_B_16",
        }
        return mapping.get(normalized, "ResNet18")

    @staticmethod
    def _detect_backbone_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> str:
        if not state_dict:
            return "ResNet18"

        keys = [k[7:] if k.startswith("module.") else k for k in state_dict.keys()]
        stripped = [k[len("backbone."):] if k.startswith("backbone.") else k for k in keys]

        # torchvision ViT 常见键: conv_proj / encoder.layers / heads.head
        if any(k.startswith(("conv_proj.", "encoder.layers.", "heads.")) for k in stripped):
            return "ViT_B_16"

        return "ResNet18"

    def __init__(self, pretrained: bool = True, backbone_name: str = "ResNet18", num_classes: int = 2):
        super().__init__()
        self.backbone_name = self._normalize_backbone_name(backbone_name)
        self.num_classes = max(2, int(num_classes))

        if self.backbone_name == "ResNet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet34(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, self.num_classes)
        elif self.backbone_name == "ResNet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, self.num_classes)
        elif self.backbone_name == "ViT_B_16":
            weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
            self.backbone = models.vit_b_16(weights=weights)
            num_features = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Linear(num_features, self.num_classes)
        else:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, self.num_classes)

    @staticmethod
    def _infer_num_classes_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> int:
        candidate_keys = (
            "backbone.fc.weight",
            "fc.weight",
            "backbone.heads.head.weight",
            "heads.head.weight",
        )
        for key in candidate_keys:
            tensor = state_dict.get(key)
            if isinstance(tensor, torch.Tensor) and tensor.ndim == 2 and tensor.shape[0] >= 2:
                return int(tensor.shape[0])
        return 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @staticmethod
    def load_from_checkpoint(
        path: str,
        device: Optional[torch.device] = None,
        model_format: ModelFormat = ModelFormat.AUTO,
        backbone_name: str = "auto",
    ) -> "SCANNClassifier":
        """从 checkpoint 加载模型，自动兼容 v1/v2 格式

        Args:
            path: 模型文件路径
            device: 目标设备
            model_format: 模型格式 (AUTO=自动检测, V1_CLASSIFIER, V2_CLASSIFIER)
            backbone_name: 模型主干网络（auto=自动检测/回退）

        Returns:
            加载好的模型 (eval mode)
        """
        if device is None:
            device = resolve_device("auto").resolved

        ckpt = torch.load(path, map_location=device, weights_only=False)

        # 提取 state_dict
        state_dict = None
        if isinstance(ckpt, dict):
            state_dict = ckpt.get("state") or ckpt.get("model_state") or ckpt
        else:
            state_dict = ckpt

        # 确定格式
        if model_format == ModelFormat.AUTO:
            # 先检查 checkpoint 中是否有格式元数据
            if isinstance(ckpt, dict) and "model_format" in ckpt:
                try:
                    model_format = ModelFormat(ckpt["model_format"])
                except ValueError:
                    model_format = detect_model_format(state_dict)
            else:
                model_format = detect_model_format(state_dict)

        logger.info("检测到模型格式: %s", model_format.value)

        resolved_backbone = SCANNClassifier._normalize_backbone_name(backbone_name)
        if str(backbone_name).strip().lower() == "auto":
            if isinstance(ckpt, dict) and isinstance(ckpt.get("backbone"), str):
                resolved_backbone = SCANNClassifier._normalize_backbone_name(ckpt["backbone"])
            else:
                resolved_backbone = SCANNClassifier._detect_backbone_from_state_dict(state_dict)

        # v1 原生格式固定是 ResNet 语义
        if model_format == ModelFormat.V1_CLASSIFIER:
            resolved_backbone = "ResNet18"

        # 清理 state_dict：移除 module. 前缀，过滤非 Tensor 元数据项
        clean_state: Dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            if not isinstance(v, torch.Tensor):
                continue
            name = k[7:] if k.startswith("module.") else k
            clean_state[name] = v

        num_classes = SCANNClassifier._infer_num_classes_from_state_dict(clean_state)

        # v1: 原生加载（不做 key 转换）
        if model_format == ModelFormat.V1_CLASSIFIER:
            model = SCANNClassifier(pretrained=False, backbone_name=resolved_backbone, num_classes=num_classes)
            try:
                model.backbone.load_state_dict(clean_state, strict=True)
                logger.info("使用 v1 原生权重加载（未进行 v1->v2 key 转换）")
            except RuntimeError:
                # 兼容少数“误标 v1 但实际是 v2 键”的文件
                v2_like = {}
                for k, v in clean_state.items():
                    if k.startswith("backbone."):
                        v2_like[k] = v
                if v2_like:
                    model.load_state_dict(v2_like, strict=False)
                    logger.warning("检测到 v1 格式但权重为 v2 键名，已兼容加载")
                else:
                    raise
            model.to(device)
            model.eval()
            return model

        # v2: 正常加载（支持 backbone. 前缀）
        model = SCANNClassifier(pretrained=False, backbone_name=resolved_backbone, num_classes=num_classes)
        model.load_state_dict(clean_state, strict=False)
        model.to(device)
        model.eval()
        return model

    @staticmethod
    def save_checkpoint(
        model: "SCANNClassifier",
        path: str,
        threshold: float = 0.5,
        model_format: ModelFormat = ModelFormat.V2_CLASSIFIER,
        **extra_metadata,
    ) -> None:
        """保存模型 checkpoint，携带格式元数据

        Args:
            model: 要保存的模型
            path: 保存路径
            threshold: 检测阈值
            model_format: 保存的模型格式
            **extra_metadata: 额外元数据 (如 epoch, metrics 等)
        """
        state_dict = model.state_dict()

        # 如果要求保存为 v1 格式，则移除 backbone. 前缀
        if model_format == ModelFormat.V1_CLASSIFIER:
            state_dict = _convert_state_dict_v2_to_v1(state_dict)

        ckpt = {
            "state": state_dict,
            "threshold": threshold,
            "model_format": model_format.value,
            "backbone": getattr(model, "backbone_name", "ResNet18"),
            **extra_metadata,
        }
        torch.save(ckpt, path)
        logger.info("模型已保存: %s (格式=%s)", path, model_format.value)


class SCANNDetector(nn.Module):
    """v2 全图检测模型 (Vision Transformer)

    设计目标:
    - 输入完整 FITS 图像，输出密集检测图
    - 基于 ViT 编码器提取全局上下文
    - 保留 `forward()` 兼容输出: [B, 5] (x, y, w, h, confidence)

    说明:
    - `forward_dense()` 返回 [B, 5, Hp, Wp] 的密集预测图
    - `forward()` 对密集预测做全局池化，保持旧接口可用
    """

    def __init__(
        self,
        in_channels: int = 1,
        pretrained: bool = True,
        patch_size: int = 16,
        embed_dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        _ = pretrained
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)

        self.input_adapter = (
            nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
            if in_channels != 3
            else nn.Identity()
        )

        self.patch_embed = nn.Conv2d(
            3,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=int(num_heads),
            dim_feedforward=int(self.embed_dim * mlp_ratio),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.norm = nn.LayerNorm(self.embed_dim)

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.embed_dim, 1, kernel_size=1),
        )
        self.bbox_head = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.embed_dim, 4, kernel_size=1),
        )

    @staticmethod
    def _build_2d_sincos_pos_embed(
        h: int,
        w: int,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if dim % 4 != 0:
            raise ValueError("embed_dim 必须能被 4 整除")

        quarter_dim = dim // 4
        y = torch.linspace(0.0, 1.0, steps=h, device=device, dtype=dtype)
        x = torch.linspace(0.0, 1.0, steps=w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        freq = torch.arange(quarter_dim, device=device, dtype=dtype)
        freq = 1.0 / (10000 ** (freq / max(1, quarter_dim)))

        y_proj = yy.reshape(-1, 1) * freq.reshape(1, -1)
        x_proj = xx.reshape(-1, 1) * freq.reshape(1, -1)

        pos = torch.cat(
            [
                torch.sin(y_proj),
                torch.cos(y_proj),
                torch.sin(x_proj),
                torch.cos(x_proj),
            ],
            dim=1,
        )
        return pos.unsqueeze(0)

    def forward_dense(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_adapter(x)
        feat = self.patch_embed(x)
        b, c, hp, wp = feat.shape

        tokens = feat.flatten(2).transpose(1, 2)
        pos = self._build_2d_sincos_pos_embed(
            hp,
            wp,
            c,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        tokens = tokens + pos

        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        feat_map = tokens.transpose(1, 2).reshape(b, c, hp, wp)

        heatmap = self.heatmap_head(feat_map)
        bbox = self.bbox_head(feat_map)
        return torch.cat([bbox, heatmap], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dense_pred = self.forward_dense(x)
        pooled = nn.functional.adaptive_avg_pool2d(dense_pred, output_size=1)
        return pooled.flatten(1)

    def estimate_memory_mb(self, input_size: tuple = (1, 3, 1024, 1024)) -> float:
        """估算推理显存占用 (MB, 粗略)。"""
        batch, channels, height, width = input_size
        param_mem = sum(p.numel() * p.element_size() for p in self.parameters())

        hp = max(1, height // self.patch_size)
        wp = max(1, width // self.patch_size)
        token_count = hp * wp

        token_mem = batch * token_count * self.embed_dim * 4
        attn_mem = batch * token_count * token_count * 2
        activation_mem = batch * channels * height * width * 4

        total = (param_mem + token_mem + attn_mem + activation_mem) / 1024 / 1024
        return float(total)
