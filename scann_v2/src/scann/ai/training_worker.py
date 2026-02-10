"""AI 训练工作线程

职责:
- QThread 子类，在后台执行训练
- 通过信号实时报告训练进度
- 支持被外部中断
"""

from __future__ import annotations

import logging
import random
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import pyqtSignal, QObject, QThread

# 设置PyTorch模型下载路径到项目内（必须在导入torch之前设置）
try:
    model_file = Path(__file__).resolve()
    # training_worker.py 位于 scann_v2/src/scann/ai/training_worker.py，需要向上4级到 scann_v2/
    scann_v2_root = model_file.parent.parent.parent.parent
    model_cache_dir = scann_v2_root / "models" / "torch_cache"
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置环境变量
    os.environ['TORCH_HOME'] = str(model_cache_dir)
    os.environ['TORCH_HUB_DIR'] = str(model_cache_dir)
except Exception:
    pass

import torch  # 现在导入torch

from scann.ai.model import ModelFormat, SCANNClassifier
from scann.ai.trainer import TrainConfig, compute_metrics, find_threshold_for_recall

logger = logging.getLogger(__name__)

# 确认缓存目录
torch.hub.set_dir(str(model_cache_dir))
logger.info(f"PyTorch模型缓存目录: {model_cache_dir}")


class TrainingWorker(QThread):
    """训练工作线程

    信号:
        progress(epoch, total, loss, val_loss): 每个 epoch 的进度
        finished(model_path, metrics): 训练完成
        error(message): 训练出错
    """

    progress = pyqtSignal(int, int, float, float)  # epoch, total, loss, val_loss
    finished = pyqtSignal(str, dict)  # model_path, metrics
    error = pyqtSignal(str)

    def __init__(
        self,
        params: dict,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self._params = params
        self._should_stop = False

    def run(self) -> None:
        """执行训练流程"""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from sklearn.metrics import average_precision_score
            from torch.utils.data import DataLoader, WeightedRandomSampler
            from torchvision import models, transforms

            # 解析参数
            pos_dir = self._params["pos_dir"]
            neg_dir = self._params["neg_dir"]
            epochs = self._params["epochs"]
            batch_size = self._params["batch_size"]
            lr = self._params["lr"]
            backbone_name = self._params["backbone"]
            save_format = self._params.get("save_format", "v2_classifier")
            val_split = self._params.get("val_split", 0.2)
            augment = self._params.get("augment", True)

            # 设备
            requested_device = str(self._params.get("device", "auto")).strip().lower()
            if requested_device in {"auto", ""}:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            elif requested_device in {"cuda", "cuda:0", "gpu", "nvidia"}:
                if torch.cuda.is_available():
                    device = torch.device("cuda:0")
                else:
                    logger.warning("请求使用CUDA但当前环境不可用，回退到CPU")
                    device = torch.device("cpu")
            elif requested_device in {"cpu"}:
                device = torch.device("cpu")
            else:
                # 允许传入诸如 "cuda:1" / "mps" 等自定义字符串；失败则回退cpu
                try:
                    device = torch.device(requested_device)
                except Exception:
                    logger.warning(f"无法解析训练设备: {requested_device}，回退到CPU")
                    device = torch.device("cpu")

            logger.info(f"训练设备: {device}")

            # === 1. 数据集加载 ===

            # 收集所有样本
            all_samples = []
            for dir_path, label in [(pos_dir, 1), (neg_dir, 0)]:
                if not os.path.isdir(dir_path):
                    raise ValueError(f"目录不存在: {dir_path}")
                for fn in os.listdir(dir_path):
                    if fn.lower().endswith((".png", ".fts", ".fit")):
                        all_samples.append((os.path.join(dir_path, fn), label))

            if not all_samples:
                raise ValueError("未找到任何样本文件")

            # 划分训练集/验证集
            n = len(all_samples)
            idx = np.arange(n)
            np.random.shuffle(idx)
            split = int((1.0 - val_split) * n)
            train_idx = idx[:split].tolist()
            val_idx = idx[split:].tolist()

            train_samples = [all_samples[i] for i in train_idx]
            val_samples = [all_samples[i] for i in val_idx]

            logger.info(f"训练集: {len(train_samples)}, 验证集: {len(val_samples)}")

            # 创建数据集（内联实现以避免复杂依赖）
            # 直接传入预构建的样本列表，避免 TripletPNGDataset 重复收集
            from scann.ai.dataset import TripletPNGDataset

            train_set = TripletPNGDataset(
                samples=train_samples,
                split="train",
                resize=224,
                augment=augment,
            )
            val_set = TripletPNGDataset(
                samples=val_samples,
                split="val",
                resize=224,
                augment=False,
            )

            # 类别平衡采样
            train_labels = [all_samples[i][1] for i in train_idx]
            count_neg = train_labels.count(0)
            count_pos = train_labels.count(1)
            weight_class = [1.0 / max(count_neg, 1), 1.0 / max(count_pos, 1)]
            samples_weight = [weight_class[y] for y in train_labels]
            samples_weight = torch.tensor(samples_weight, dtype=torch.double)

            sampler = WeightedRandomSampler(
                samples_weight, num_samples=len(train_set), replacement=True
            )

            train_loader = DataLoader(
                train_set, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=False
            )
            val_loader = DataLoader(
                val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
            )

            # === 2. 模型 ===
            backbone_cls = {
                "ResNet18": models.resnet18,
                "ResNet34": models.resnet34,
                "ResNet50": models.resnet50,
            }.get(backbone_name, models.resnet18)

            if backbone_name == "ResNet18":
                weights = models.ResNet18_Weights.DEFAULT
            elif backbone_name == "ResNet34":
                weights = models.ResNet34_Weights.DEFAULT
            else:
                weights = models.ResNet50_Weights.DEFAULT

            model = backbone_cls(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, 2)
            model = model.to(device)

            # === 3. 损失和优化器 ===
            from scann.ai.trainer import FocalLoss

            criterion = FocalLoss(gamma=2.0, alpha=[1.0, 1.5]).to(device)

            optimizer_name = self._params.get("optimizer", "Adam")
            if optimizer_name == "AdamW":
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
            elif optimizer_name == "SGD":
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)
            else:  # Adam
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

            # === 4. 训练循环 ===
            best_f2 = -1.0
            best_threshold = 0.5
            best_epoch = 0
            save_path = "best_model.pth"

            for epoch in range(epochs):
                if self._should_stop:
                    logger.info("训练被中断")
                    break

                model.train()
                total_loss = 0.0
                seen = 0

                for x, y in train_loader:
                    if self._should_stop:
                        break

                    x, y = x.to(device), y.to(device)

                    optimizer.zero_grad()
                    logits = model(x)
                    loss = criterion(logits, y)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item() * x.size(0)
                    seen += x.size(0)

                train_loss = total_loss / max(seen, 1)

                # 验证
                model.eval()
                all_probs, all_labels = [], []

                with torch.no_grad():
                    for x, y in val_loader:
                        if self._should_stop:
                            break
                        x, y = x.to(device), y.to(device)
                        logits = model(x)
                        probs = torch.softmax(logits, dim=1)[:, 1]
                        all_probs.append(probs.cpu().numpy())
                        all_labels.append(y.cpu().numpy())

                if not all_probs:
                    break

                probs = np.concatenate(all_probs)
                labels = np.concatenate(all_labels)

                # 计算 PR-AUC
                try:
                    pr_auc = float(average_precision_score(labels, probs))
                except Exception:
                    pr_auc = 0.0

                # 寻找阈值 (目标 recall=0.98)
                threshold = find_threshold_for_recall(probs, labels, target_recall=0.98)
                metrics = compute_metrics(probs, labels, threshold, beta=2.0)

                # 发射进度
                self.progress.emit(epoch + 1, epochs, train_loss, pr_auc)

                # 保存最佳模型
                if metrics["f2"] > best_f2 + 0.001:
                    best_f2 = metrics["f2"]
                    best_threshold = threshold
                    best_epoch = epoch

                    # 确定保存格式
                    model_format = ModelFormat.V1_CLASSIFIER if save_format == "v1_classifier" else ModelFormat.V2_CLASSIFIER

                    # 使用 SCANNClassifier 包装保存
                    wrapper = SCANNClassifier(pretrained=False)
                    wrapper.backbone = model
                    SCANNClassifier.save_checkpoint(
                        wrapper, save_path, threshold=best_threshold, model_format=model_format
                    )
                    logger.info(f"保存最佳模型 (epoch={epoch+1}, F2={best_f2:.4f})")

            # 训练完成
            final_metrics = {
                "best_f2": best_f2,
                "best_threshold": best_threshold,
                "best_epoch": best_epoch,
            }
            self.finished.emit(save_path, final_metrics)

        except Exception as e:
            logger.exception("训练失败")
            self.error.emit(f"训练失败: {e}")

    def stop(self) -> None:
        """请求停止训练"""
        self._should_stop = True
