"""AI 模型定义模块

职责:
- SCANNDetector: 全图目标检测模型 (新版)
- SCANNClassifier: 兼容 v1 的裁剪图分类器
- 显存控制 ≤ 8GB
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


class SCANNClassifier(nn.Module):
    """v1 兼容分类器 - 基于 ResNet-18 的裁剪图分类

    输入: [B, 3, 224, 224] (三通道: diff/new/ref)
    输出: [B, 2] (二分类 logits)
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        if pretrained:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            self.backbone = models.resnet18(weights=None)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @staticmethod
    def load_from_checkpoint(
        path: str,
        device: Optional[torch.device] = None,
    ) -> "SCANNClassifier":
        """从 v1 checkpoint 加载模型

        Args:
            path: 模型文件路径
            device: 目标设备

        Returns:
            加载好的模型 (eval mode)
        """
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = SCANNClassifier(pretrained=False)
        ckpt = torch.load(path, map_location=device, weights_only=False)

        state_dict = None
        if isinstance(ckpt, dict):
            state_dict = ckpt.get("state") or ckpt.get("model_state") or ckpt
        else:
            state_dict = ckpt

        # 清理 'module.' 前缀
        clean_state = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            clean_state[name] = v

        model.load_state_dict(clean_state, strict=True)
        model.to(device)
        model.eval()
        return model


class SCANNDetector(nn.Module):
    """v2 全图检测模型

    设计原则:
    - 显存占用 ≤ 8GB
    - 支持 CUDA 多线程
    - 输入为完整天文图像

    TODO: 具体架构待定，可能基于:
    - YOLOv8-nano (轻量检测)
    - 滑动窗口 + 分类器
    - Feature Pyramid + Attention
    """

    def __init__(self, in_channels: int = 1, pretrained: bool = True):
        super().__init__()
        # 输入通道适配 (天文图像通常为单通道灰度)
        self.input_adapter = (
            nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
            if in_channels != 3
            else nn.Identity()
        )
        # 基于轻量化骨干网络
        if pretrained:
            self.backbone = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.DEFAULT
            )
        else:
            self.backbone = models.mobilenet_v3_small(weights=None)
        # 检测头 (简化版)
        self.detect_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(576, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 5),  # x, y, w, h, confidence
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_adapter(x)
        features = self.backbone.features(x)
        return self.detect_head(features)

    def estimate_memory_mb(self, input_size: tuple = (1, 3, 1024, 1024)) -> float:
        """估算推理时的显存占用 (MB)"""
        param_mem = sum(p.numel() * p.element_size() for p in self.parameters())
        # 粗略估算激活内存
        activation_mem = input_size[0] * input_size[1] * input_size[2] * input_size[3] * 4
        total = (param_mem + activation_mem) / 1024 / 1024
        return total
