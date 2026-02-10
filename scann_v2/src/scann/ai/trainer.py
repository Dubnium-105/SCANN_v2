"""AI 训练管线

职责:
- 完整的模型训练流程
- 数据增强
- 评估指标 (Recall, Precision, F2, PR-AUC)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class TrainConfig:
    """训练配置"""
    data_dir: str = "dataset"
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 2e-4
    weight_decay: float = 1e-3
    pos_weight: float = 1.0
    focal_gamma: float = 2.0
    target_recall: float = 0.98
    val_ratio: float = 0.2
    seed: int = 42
    save_path: str = "best_model.pth"
    channel_order: Tuple[int, int, int] = (0, 1, 2)


@dataclass
class TrainMetrics:
    """训练指标"""
    epoch: int = 0
    train_loss: float = 0.0
    pr_auc: float = 0.0
    threshold: float = 0.5
    precision: float = 0.0
    recall: float = 0.0
    f2_score: float = 0.0


class FocalLoss(nn.Module):
    """Focal Loss 实现

    参考: https://arxiv.org/abs/1708.02002
    用于处理类别不平衡问题

    Args:
        gamma: 聚焦参数，越大越关注难分类样本
        alpha: 类别权重，可以是 float（所有样本相同权重）或 list/tensor（每个类别不同权重）
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        logp = torch.log_softmax(logits, dim=1)
        p = torch.softmax(logits, dim=1)
        t = targets.view(-1, 1)
        logp_t = logp.gather(1, t).squeeze(1)
        p_t = p.gather(1, t).squeeze(1)
        loss = -(1 - p_t) ** self.gamma * logp_t

        if self.alpha is not None:
            # 如果 alpha 是列表或 tensor，根据样本的标签选择对应的权重
            if isinstance(self.alpha, (list, tuple)):
                alpha_t = torch.tensor(self.alpha, dtype=logits.dtype, device=logits.device)
                # 选择每个样本对应类别的权重
                alpha_per_sample = alpha_t[t]
                loss = alpha_per_sample * loss
            elif isinstance(self.alpha, torch.Tensor):
                # 如果是 tensor，直接使用（确保在正确的设备上）
                alpha_t = self.alpha.to(dtype=logits.dtype, device=logits.device)
                alpha_per_sample = alpha_t[t]
                loss = alpha_per_sample * loss
            else:
                # alpha 是 float，对所有样本使用相同权重
                loss = self.alpha * loss

        return loss.mean()


def compute_confusion_matrix(
    y_true,
    y_pred,
    threshold: Optional[float] = None,
) -> np.ndarray:
    """计算混淆矩阵

    Args:
        y_true: 真实标签 (0/1)
        y_pred: 预测标签或概率
        threshold: 如果 y_pred 是概率，用此阈值二值化

    Returns:
        2x2 numpy array: [[TN, FP], [FN, TP]]
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if threshold is not None:
        preds = (y_pred >= threshold).astype(np.int32)
    else:
        preds = y_pred.astype(np.int32)

    tp = int(((preds == 1) & (y_true == 1)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    return np.array([[tn, fp], [fn, tp]], dtype=np.int32)


def compute_metrics(
    y_true,
    y_pred,
    threshold: Optional[float] = None,
    beta: float = 1.0,
) -> Dict[str, float]:
    """计算精度指标"""
    cm = compute_confusion_matrix(y_true, y_pred, threshold)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    total = tn + fp + fn + tp
    eps = 1e-12

    accuracy = (tp + tn) / (total + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f_beta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + eps)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f_beta) if beta == 1.0 else float(f_beta),
        f"f{beta:.0f}": float(f_beta),
    }


def find_threshold_for_recall(
    probs,
    labels,
    target_recall: float = 0.98,
) -> float:
    """找到满足目标 Recall 的最佳阈值"""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    thresholds = np.unique(probs)
    thresholds = np.sort(thresholds)[::-1]

    best_threshold = 0.0

    for t in thresholds:
        metrics = compute_metrics(probs, labels, float(t))
        if metrics["recall"] >= target_recall - 1e-6:
            best_threshold = float(t)
            break

    return best_threshold
