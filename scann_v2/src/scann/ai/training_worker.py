"""AI 训练工作线程

职责:
- QThread 子类，在后台执行训练
- 通过信号实时报告训练进度
- 支持被外部中断
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Optional

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

from scann.ai.class_balance import (
    DETAIL_TYPE_CLASS_ORDER,
    DETAIL_TYPE_TO_CLASS_INDEX,
    build_class_audit,
    compute_class_balanced_weights,
    compute_multiclass_metrics,
    merge_imbalance_config,
    normalize_detail_type,
    sampler_weights_from_class_weights,
    stratified_group_train_val_split,
)
from scann.ai.device_utils import resolve_device
from scann.ai.model import ModelFormat, SCANNClassifier
from scann.ai.trainer import TrainConfig
from scann.core.fits_annotation_storage import load_v2_annotation_document
from scann.core.fits_io import read_fits
from scann.data.file_manager import FitsImagePair, match_new_old_pairs
from scann.services.detection_image_adapter import robust_to_uint8

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

    def _resolve_dataset_format(self) -> str:
        dataset_format = str(self._params.get("dataset_format", "")).strip().lower()
        if dataset_format in {"v1", "v1_triplet", "triplet"}:
            return "v1"
        if dataset_format in {"v2", "v2_fits", "fits"}:
            return "v2"
        if self._params.get("pos_dir") and self._params.get("neg_dir"):
            return "v1"

        save_format = str(self._params.get("save_format", "")).strip().lower()
        if save_format == "v1_classifier":
            return "v1"
        return "v2"

    def _resolve_task_type(self) -> str:
        task_type = str(self._params.get("task_type", "classification")).strip().lower()
        if task_type != "classification":
            logger.warning("训练任务已统一为11类细分类，忽略 task_type=%s，强制使用 classification", task_type)
        return "classification"

    def _collect_v1_samples_from_dirs(
        self,
        pos_dir: str | Path,
        neg_dir: str | Path,
    ) -> list[tuple[str, int]]:
        all_samples: list[tuple[str, int]] = []
        supported_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

        for dir_path, label in [(Path(pos_dir), 1), (Path(neg_dir), 0)]:
            if not dir_path.is_dir():
                raise ValueError(f"目录不存在: {dir_path}")
            for fn in sorted(dir_path.iterdir()):
                if fn.is_file() and fn.suffix.lower() in supported_exts:
                    all_samples.append((str(fn), label))

        if not all_samples:
            raise ValueError("未找到任何 v1 三联图样本，请检查 positive/negative 子目录")
        return all_samples

    def _collect_v1_samples_from_root(self, dataset_root: Path) -> list[tuple[str, int]]:
        return self._collect_v1_samples_from_dirs(
            dataset_root / "positive",
            dataset_root / "negative",
        )

    def _normalize_dataset_key(self, value: str | None) -> str:
        if not value:
            return ""

        key = Path(str(value)).stem
        if key.lower().endswith("__aligned_crop"):
            key = key[: -len("__aligned_crop")]
        for prefix in ("FW_", "fw_", "Fw_"):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        return key.strip().lower()

    def _build_pair_lookup(self, pairs: list[FitsImagePair]) -> dict[str, FitsImagePair]:
        lookup: dict[str, FitsImagePair] = {}
        for pair in pairs:
            candidates = {
                pair.name,
                pair.new_path.stem,
                pair.new_path.name,
                pair.old_path.stem,
                pair.old_path.name,
            }
            for candidate in candidates:
                key = self._normalize_dataset_key(candidate)
                if key:
                    lookup[key] = pair
        return lookup

    def _resolve_pair_from_paths(self, dataset_root: Path, image_info: dict[str, Any]) -> FitsImagePair | None:
        paths = image_info.get("paths")
        if not isinstance(paths, dict):
            metadata = image_info.get("metadata")
            if isinstance(metadata, dict):
                paths = metadata.get("paths")
        if not isinstance(paths, dict):
            return None

        new_rel = str(paths.get("new") or "").strip()
        old_rel = str(paths.get("old") or "").strip()
        if not new_rel or not old_rel:
            return None

        new_path = dataset_root / new_rel
        old_path = dataset_root / old_rel
        if not new_path.is_file() or not old_path.is_file():
            return None

        return FitsImagePair(
            name=str(image_info.get("id") or image_info.get("file_name") or new_path.stem),
            new_path=new_path,
            old_path=old_path,
        )

    def _resolve_annotation_detail_type(self, ann: dict[str, Any], image_info: dict[str, Any]) -> str | None:
        detail_type = str(ann.get("detail_type") or image_info.get("detail_type") or "").strip()
        return normalize_detail_type(detail_type)

    def _ensure_2d_image(self, data: np.ndarray, path: Path) -> np.ndarray:
        arr = np.asarray(data)
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"FITS 图像维度不受支持: {path} -> {arr.shape}")
        return arr

    def _load_pair_uint8(self, pair: FitsImagePair) -> tuple[np.ndarray, np.ndarray]:
        new_data = self._ensure_2d_image(read_fits(pair.new_path).data, pair.new_path)
        old_data = self._ensure_2d_image(read_fits(pair.old_path).data, pair.old_path)
        new_u8 = robust_to_uint8(new_data)
        old_u8 = robust_to_uint8(old_data)
        height = min(new_u8.shape[0], old_u8.shape[0])
        width = min(new_u8.shape[1], old_u8.shape[1])
        if height <= 0 or width <= 0:
            raise ValueError(f"图像尺寸无效: {pair.new_path.name} / {pair.old_path.name}")
        return new_u8[:height, :width], old_u8[:height, :width]

    def _extract_center_patch(self, image: np.ndarray, center_x: float, center_y: float, size: int) -> np.ndarray:
        patch = np.zeros((size, size), dtype=np.uint8)
        left = int(round(center_x - size / 2.0))
        top = int(round(center_y - size / 2.0))
        right = left + size
        bottom = top + size

        src_left = max(left, 0)
        src_top = max(top, 0)
        src_right = min(right, image.shape[1])
        src_bottom = min(bottom, image.shape[0])
        if src_left >= src_right or src_top >= src_bottom:
            return patch

        dst_left = src_left - left
        dst_top = src_top - top
        dst_right = dst_left + (src_right - src_left)
        dst_bottom = dst_top + (src_bottom - src_top)
        patch[dst_top:dst_bottom, dst_left:dst_right] = image[src_top:src_bottom, src_left:src_right]
        return patch

    def _build_triplet_patch(
        self,
        new_u8: np.ndarray,
        old_u8: np.ndarray,
        center_x: float,
        center_y: float,
        size: int,
    ) -> np.ndarray:
        patch_new = self._extract_center_patch(new_u8, center_x, center_y, size)
        patch_old = self._extract_center_patch(old_u8, center_x, center_y, size)
        patch_diff = np.clip(
            patch_new.astype(np.float32) - patch_old.astype(np.float32),
            -255.0,
            255.0,
        )
        channels = [
            (patch_diff + 255.0) / 510.0,
            patch_new.astype(np.float32) / 255.0,
            patch_old.astype(np.float32) / 255.0,
        ]
        return np.stack(channels, axis=0).astype(np.float32)

    @staticmethod
    def _resolve_manual_crop_bounds(image_info: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int] | None:
        metadata = image_info.get("metadata")
        if not isinstance(metadata, dict):
            return None
        crop = metadata.get("manual_crop")
        if not isinstance(crop, dict):
            return None

        try:
            x = float(crop.get("x"))
            y = float(crop.get("y"))
            crop_w = float(crop.get("width"))
            crop_h = float(crop.get("height"))
        except (TypeError, ValueError):
            return None

        if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(crop_w) or not np.isfinite(crop_h):
            return None
        if crop_w <= 1 or crop_h <= 1 or width <= 0 or height <= 0:
            return None

        x0 = max(0, min(width - 1, int(round(x))))
        y0 = max(0, min(height - 1, int(round(y))))
        x1 = max(x0 + 1, min(width, int(round(x + crop_w))))
        y1 = max(y0 + 1, min(height, int(round(y + crop_h))))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    def _collect_v2_sample_records_from_root(self, dataset_root: Path) -> list[dict[str, Any]]:
        new_dir = dataset_root / "new"
        old_dir = dataset_root / "old"
        ann_path = dataset_root / "annotations.json"
        ann_db_path = dataset_root / "annotations.db"
        dataset_db_path = dataset_root / "scann_dataset.db"
        if not new_dir.is_dir() or not old_dir.is_dir():
            raise ValueError("v2 数据集目录下必须包含 new 和 old 子目录")
        if not ann_path.is_file() and not ann_db_path.is_file() and not dataset_db_path.is_file():
            raise ValueError("v2 数据集目录下缺少标注文件（annotations.json 或 annotations.db）")

        pairs, _only_new, _only_old = match_new_old_pairs(str(new_dir), str(old_dir))

        annotations_doc = self._load_v2_annotations_document(dataset_root)

        images = annotations_doc.get("images", [])
        if not images:
            raise ValueError("标注文档中没有 images 条目")

        pair_lookup = self._build_pair_lookup(pairs)
        pair_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        all_samples: list[dict[str, Any]] = []

        for image_info in images:
            annotations = image_info.get("annotations") or []
            if not annotations:
                continue

            pair = self._resolve_pair_from_paths(dataset_root, image_info)
            if pair is None:
                for candidate in (
                    image_info.get("id"),
                    image_info.get("file_name"),
                    image_info.get("file"),
                ):
                    key = self._normalize_dataset_key(candidate)
                    if key and key in pair_lookup:
                        pair = pair_lookup[key]
                        break
            if pair is None:
                continue

            pair_key = self._normalize_dataset_key(pair.name)
            if pair_key not in pair_cache:
                pair_cache[pair_key] = self._load_pair_uint8(pair)
            new_u8, old_u8 = pair_cache[pair_key]

            h, w = new_u8.shape[:2]
            crop_bounds = self._resolve_manual_crop_bounds(image_info, w, h)
            if crop_bounds is not None:
                x0, y0, x1, y1 = crop_bounds
                new_u8_view = new_u8[y0:y1, x0:x1]
                old_u8_view = old_u8[y0:y1, x0:x1]
            else:
                x0, y0 = 0, 0
                new_u8_view = new_u8
                old_u8_view = old_u8

            image_id = str(image_info.get("id") or image_info.get("file_name") or image_info.get("file") or pair.name)
            for ann_index, ann in enumerate(annotations):
                detail_type = self._resolve_annotation_detail_type(ann, image_info)
                if detail_type is None:
                    continue
                center_x = float(ann.get("x", 0)) + float(ann.get("width", 0)) / 2.0
                center_y = float(ann.get("y", 0)) + float(ann.get("height", 0)) / 2.0
                if crop_bounds is not None:
                    if center_x < x0 or center_x > x1 or center_y < y0 or center_y > y1:
                        continue
                    center_x -= x0
                    center_y -= y0
                patch_size = max(
                    80,
                    int(round(max(float(ann.get("width", 0) or 0), float(ann.get("height", 0) or 0), 0.0))),
                )
                triplet = self._build_triplet_patch(new_u8_view, old_u8_view, center_x, center_y, patch_size)
                class_index = DETAIL_TYPE_TO_CLASS_INDEX.get(detail_type)
                if class_index is None:
                    continue
                all_samples.append(
                    {
                        "data": triplet,
                        "label": int(class_index),
                        "detail_type": detail_type,
                        "task_id": image_id,
                        "annotation_index": int(ann_index),
                    }
                )

        if not all_samples:
            raise ValueError("v2 数据集里未找到可训练的已标注样本")
        return all_samples

    def _collect_v2_samples_from_root(self, dataset_root: Path) -> list[tuple[np.ndarray, int]]:
        records = self._collect_v2_sample_records_from_root(dataset_root)
        return [(record["data"], int(record["label"])) for record in records]

    def _load_v2_annotations_document(self, dataset_root: Path) -> dict[str, Any]:
        snapshot_path_raw = str(self._params.get("annotations_document_path", "")).strip()
        if snapshot_path_raw:
            snapshot_path = Path(snapshot_path_raw)
            if not snapshot_path.is_file():
                raise ValueError(f"快照标注文档不存在: {snapshot_path}")
            payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not isinstance(payload.get("images"), list):
                raise ValueError("快照标注文档格式无效")
            return payload
        return load_v2_annotation_document(dataset_root)

    def _build_sample_pool(self) -> tuple[list[dict[str, Any]], str]:
        dataset_dir = Path(str(self._params.get("dataset_dir", "")).strip())
        if not str(dataset_dir):
            raise ValueError("未设置数据集目录")

        dataset_format = self._resolve_dataset_format()
        if dataset_format != "v2":
            raise ValueError("当前训练链路已统一为11类细分类，仅支持 v2 FITS 标注数据集")
        return self._collect_v2_sample_records_from_root(dataset_dir), "array"

    @staticmethod
    def _compute_dense_detection_loss(
        pred_dense: torch.Tensor,
        target_heatmap: torch.Tensor,
        target_bbox: torch.Tensor,
        target_bbox_mask: torch.Tensor,
        heatmap_pos_weight: float,
        bbox_loss_weight: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_bbox = pred_dense[:, :4, :, :]
        pred_heatmap = pred_dense[:, 4:5, :, :]

        heatmap_loss_raw = torch.nn.functional.binary_cross_entropy_with_logits(
            pred_heatmap,
            target_heatmap,
            reduction="none",
        )
        heatmap_weights = torch.ones_like(heatmap_loss_raw)
        heatmap_weights = torch.where(
            target_heatmap > 0.5,
            torch.full_like(heatmap_weights, float(heatmap_pos_weight)),
            heatmap_weights,
        )
        heatmap_loss = (heatmap_loss_raw * heatmap_weights).mean()

        bbox_loss_raw = torch.nn.functional.smooth_l1_loss(
            pred_bbox,
            target_bbox,
            reduction="none",
        )
        bbox_mask_expand = target_bbox_mask.expand_as(bbox_loss_raw)
        bbox_denom = torch.clamp(bbox_mask_expand.sum(), min=1.0)
        bbox_loss = (bbox_loss_raw * bbox_mask_expand).sum() / bbox_denom

        total_loss = heatmap_loss + float(bbox_loss_weight) * bbox_loss
        return total_loss, heatmap_loss, bbox_loss

    @staticmethod
    def _save_detection_checkpoint(
        model: torch.nn.Module,
        save_path: str,
        best_epoch: int,
        best_val_loss: float,
        heatmap_threshold: float,
        bbox_loss_weight: float,
        heatmap_pos_weight: float,
        patch_size: int,
    ) -> None:
        checkpoint = {
            "state": model.state_dict(),
            "model_format": "v2_detector",
            "task_type": "detection",
            "backbone": "SCANNDetector",
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "heatmap_threshold": float(heatmap_threshold),
            "bbox_loss_weight": float(bbox_loss_weight),
            "heatmap_pos_weight": float(heatmap_pos_weight),
            "patch_size": int(patch_size),
        }
        torch.save(checkpoint, save_path)

    def run(self) -> None:
        """执行训练流程"""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
            from torchvision import models

            # 解析参数
            epochs = self._params["epochs"]
            batch_size = self._params["batch_size"]
            lr = self._params["lr"]
            backbone_name = self._params["backbone"]
            save_format = self._params.get("save_format", "v2_classifier")
            val_split = self._params.get("val_split", 0.2)
            augment = self._params.get("augment", True)
            dataset_format = self._resolve_dataset_format()
            task_type = self._resolve_task_type()
            save_path = str(self._params.get("save_path", "best_model.pth"))

            # 设备
            requested_device = str(self._params.get("device", "auto")).strip()
            resolved_device = resolve_device(requested_device)
            device = resolved_device.resolved
            if resolved_device.used_fallback:
                logger.warning("Training device fallback: %s", resolved_device.message)
            logger.info("Training device: requested=%s resolved=%s", requested_device, resolved_device.message)

            if task_type == "detection":
                if dataset_format != "v2":
                    raise ValueError("detection 任务仅支持 v2 FITS 配对数据集")

                from scann.ai.dataset import FitsDenseDetectionDataset
                from scann.ai.model import SCANNDetector

                patch_size = int(self._params.get("dense_patch_size", 16))
                heatmap_pos_weight = float(self._params.get("heatmap_pos_weight", 4.0))
                bbox_loss_weight = float(self._params.get("bbox_loss_weight", 2.0))
                heatmap_threshold = float(self._params.get("heatmap_threshold", 0.35))

                dataset_dir = Path(str(self._params.get("dataset_dir", "")).strip())
                if not str(dataset_dir):
                    raise ValueError("未设置数据集目录")

                dense_dataset = FitsDenseDetectionDataset(
                    dataset_root=str(dataset_dir),
                    annotation_file=str(self._params.get("annotations_document_path") or "") or None,
                    patch_size=patch_size,
                )
                if len(dense_dataset) < 2:
                    raise ValueError("dense 检测训练至少需要 2 个样本")

                n = len(dense_dataset)
                idx = np.arange(n)
                np.random.shuffle(idx)
                split = int((1.0 - val_split) * n)
                split = min(max(split, 1), n - 1)
                train_idx = idx[:split].tolist()
                val_idx = idx[split:].tolist()

                train_set = Subset(dense_dataset, train_idx)
                val_set = Subset(dense_dataset, val_idx)
                train_loader = DataLoader(
                    train_set,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=False,
                )
                val_loader = DataLoader(
                    val_set,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                )

                model = SCANNDetector(
                    in_channels=3,
                    pretrained=False,
                    patch_size=patch_size,
                ).to(device)

                optimizer_name = self._params.get("optimizer", "Adam")
                if optimizer_name == "AdamW":
                    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
                elif optimizer_name == "SGD":
                    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)
                else:
                    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

                best_val_loss = float("inf")
                best_epoch = -1

                for epoch in range(epochs):
                    if self._should_stop:
                        logger.info("训练被中断")
                        break

                    model.train()
                    train_loss_sum = 0.0
                    train_count = 0

                    for x, targets in train_loader:
                        if self._should_stop:
                            break

                        x = x.to(device)
                        target_heatmap = torch.as_tensor(targets["heatmap"], dtype=torch.float32, device=device)
                        target_bbox = torch.as_tensor(targets["bbox"], dtype=torch.float32, device=device)
                        target_bbox_mask = torch.as_tensor(targets["bbox_mask"], dtype=torch.float32, device=device)

                        optimizer.zero_grad()
                        pred_dense = model.forward_dense(x)
                        loss, _heat_loss, _bbox_loss = self._compute_dense_detection_loss(
                            pred_dense,
                            target_heatmap,
                            target_bbox,
                            target_bbox_mask,
                            heatmap_pos_weight=heatmap_pos_weight,
                            bbox_loss_weight=bbox_loss_weight,
                        )
                        loss.backward()
                        optimizer.step()

                        cur_batch = x.size(0)
                        train_loss_sum += float(loss.item()) * cur_batch
                        train_count += cur_batch

                    train_loss = train_loss_sum / max(train_count, 1)

                    model.eval()
                    val_loss_sum = 0.0
                    val_count = 0
                    with torch.no_grad():
                        for x, targets in val_loader:
                            if self._should_stop:
                                break

                            x = x.to(device)
                            target_heatmap = torch.as_tensor(targets["heatmap"], dtype=torch.float32, device=device)
                            target_bbox = torch.as_tensor(targets["bbox"], dtype=torch.float32, device=device)
                            target_bbox_mask = torch.as_tensor(targets["bbox_mask"], dtype=torch.float32, device=device)

                            pred_dense = model.forward_dense(x)
                            loss, _heat_loss, _bbox_loss = self._compute_dense_detection_loss(
                                pred_dense,
                                target_heatmap,
                                target_bbox,
                                target_bbox_mask,
                                heatmap_pos_weight=heatmap_pos_weight,
                                bbox_loss_weight=bbox_loss_weight,
                            )

                            cur_batch = x.size(0)
                            val_loss_sum += float(loss.item()) * cur_batch
                            val_count += cur_batch

                    if val_count == 0:
                        break

                    val_loss = val_loss_sum / val_count
                    self.progress.emit(epoch + 1, epochs, train_loss, val_loss)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_epoch = epoch
                        self._save_detection_checkpoint(
                            model=model,
                            save_path=save_path,
                            best_epoch=best_epoch,
                            best_val_loss=best_val_loss,
                            heatmap_threshold=heatmap_threshold,
                            bbox_loss_weight=bbox_loss_weight,
                            heatmap_pos_weight=heatmap_pos_weight,
                            patch_size=patch_size,
                        )
                        logger.info("保存最佳 dense 模型 (epoch=%d, val_loss=%.6f)", epoch + 1, best_val_loss)

                if best_epoch < 0:
                    raise ValueError("dense 检测训练未产生有效验证结果")

                final_metrics = {
                    "task_type": "detection",
                    "best_epoch": best_epoch,
                    "best_val_loss": float(best_val_loss),
                    "best_f2": 0.0,
                    "best_threshold": float(heatmap_threshold),
                }
                self.finished.emit(save_path, final_metrics)
                return

            # === 1. 数据集加载 ===
            all_samples, sample_kind = self._build_sample_pool()
            if len(all_samples) < 2:
                raise ValueError("训练至少需要 2 个样本")

            logger.info(
                "训练数据集: format=%s, kind=%s, samples=%d",
                dataset_format,
                sample_kind,
                len(all_samples),
            )

            # 划分训练集/验证集
            imbalance_config = merge_imbalance_config(self._params)
            seed = int(self._params.get("seed", 42))
            train_idx, val_idx, split_support = stratified_group_train_val_split(
                all_samples,
                val_split=val_split,
                seed=seed,
            )

            train_records = [all_samples[i] for i in train_idx]
            val_records = [all_samples[i] for i in val_idx]
            train_samples = [(record["data"], int(record["label"])) for record in train_records]
            val_samples = [(record["data"], int(record["label"])) for record in val_records]

            logger.info(f"训练集: {len(train_samples)}, 验证集: {len(val_samples)}")

            # 创建数据集（内联实现以避免复杂依赖）
            # 直接传入预构建的样本列表，避免 TripletPNGDataset 重复收集
            from scann.ai.dataset import TripletArrayDataset, TripletPNGDataset

            dataset_cls = TripletArrayDataset if sample_kind == "array" else TripletPNGDataset
            train_set = dataset_cls(
                samples=train_samples,
                split="train",
                resize=224,
                augment=augment,
            )
            val_set = dataset_cls(
                samples=val_samples,
                split="val",
                resize=224,
                augment=False,
            )

            # 类别平衡采样（11类）
            train_labels = [int(record["label"]) for record in train_records]
            class_count = len(DETAIL_TYPE_CLASS_ORDER)
            weight_class = compute_class_balanced_weights(
                train_labels,
                beta=float(imbalance_config["class_weight_beta"]),
                clip=imbalance_config["class_weight_clip"],
                class_count=class_count,
            )
            sample_weight_values = sampler_weights_from_class_weights(
                train_labels,
                weight_class,
                power=float(imbalance_config["sampler_power"]),
                max_ratio=float(imbalance_config["sampler_max_ratio"]),
            )
            samples_weight = torch.tensor(sample_weight_values, dtype=torch.double)
            class_support = build_class_audit(
                all_samples,
                split_support=split_support,
                min_train_support=int(imbalance_config["min_train_support_warning"]),
                min_val_support=int(imbalance_config["min_val_support_warning"]),
            )
            untrained_classes = [
                detail_type
                for detail_type, count in class_support["split_support"].get("train", {}).items()
                if int(count) <= 0
            ]
            unverifiable_classes = [
                detail_type
                for detail_type, count in class_support["split_support"].get("val", {}).items()
                if int(count) <= 0
            ]
            unreliable_classes = list(
                dict.fromkeys(
                    list(class_support.get("missing_classes", []))
                    + list(class_support.get("low_sample_classes", []))
                    + unverifiable_classes
                )
            )

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
            if backbone_name == "ResNet34":
                model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
                model.fc = nn.Linear(model.fc.in_features, class_count)
            elif backbone_name == "ResNet50":
                model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                model.fc = nn.Linear(model.fc.in_features, class_count)
            elif backbone_name == "ViT_B_16":
                model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
                model.heads.head = nn.Linear(model.heads.head.in_features, class_count)
            else:
                model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                model.fc = nn.Linear(model.fc.in_features, class_count)
            model = model.to(device)

            # === 3. 损失和优化器 ===
            from scann.ai.trainer import FocalLoss

            focal_gamma = float(self._params.get("focal_gamma", 2.0))
            criterion = FocalLoss(gamma=focal_gamma, alpha=weight_class).to(device)

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
            best_accuracy = 0.0
            best_epoch = 0
            best_metrics: dict[str, Any] = {}

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
                all_probs, all_labels, all_preds = [], [], []

                with torch.no_grad():
                    for x, y in val_loader:
                        if self._should_stop:
                            break
                        x, y = x.to(device), y.to(device)
                        logits = model(x)
                        probs = torch.softmax(logits, dim=1)
                        all_probs.append(probs.cpu().numpy())
                        all_preds.append(torch.argmax(probs, dim=1).cpu().numpy())
                        all_labels.append(y.cpu().numpy())

                if all_probs:
                    labels = np.concatenate(all_labels)
                    preds = np.concatenate(all_preds)
                else:
                    labels = np.asarray([], dtype=np.int64)
                    preds = np.asarray([], dtype=np.int64)

                # 多分类验证指标
                metrics = compute_multiclass_metrics(labels, preds, class_count=class_count)
                macro_f1 = float(metrics["macro_f1_supported"])
                accuracy = float(metrics["accuracy"])

                # 发射进度
                self.progress.emit(epoch + 1, epochs, train_loss, macro_f1)

                # 保存最佳模型
                if macro_f1 > best_f2 + 0.001:
                    best_f2 = macro_f1
                    best_accuracy = accuracy
                    best_epoch = epoch
                    best_metrics = dict(metrics)

                    # 确定保存格式
                    model_format = ModelFormat.V1_CLASSIFIER if save_format == "v1_classifier" else ModelFormat.V2_CLASSIFIER
                    if backbone_name == "ViT_B_16" and model_format == ModelFormat.V1_CLASSIFIER:
                        logger.warning("ViT_B_16 不支持保存为 v1_classifier，已自动切换为 v2_classifier")
                        model_format = ModelFormat.V2_CLASSIFIER

                    # 使用 SCANNClassifier 包装保存
                    wrapper = SCANNClassifier(pretrained=False, backbone_name=backbone_name, num_classes=class_count)
                    wrapper.backbone = model
                    wrapper.backbone_name = backbone_name
                    SCANNClassifier.save_checkpoint(
                        wrapper,
                        save_path,
                        threshold=best_threshold,
                        model_format=model_format,
                        backbone=backbone_name,
                        task_type="classification",
                        class_names=list(DETAIL_TYPE_CLASS_ORDER),
                        num_classes=class_count,
                        classification_mode="detail_type_11",
                        class_support=class_support,
                        class_weights=weight_class,
                        sampler_weight_range=[
                            float(min(sample_weight_values)) if sample_weight_values else 0.0,
                            float(max(sample_weight_values)) if sample_weight_values else 0.0,
                        ],
                        imbalance_config=imbalance_config,
                        untrained_classes=untrained_classes,
                        unverifiable_classes=unverifiable_classes,
                        unreliable_classes=unreliable_classes,
                        selection_metric=str(imbalance_config["selection_metric"]),
                        best_metrics=best_metrics,
                    )
                    logger.info(
                        "保存最佳11类模型 (epoch=%s, macro_f1=%.4f, acc=%.4f)",
                        epoch + 1,
                        best_f2,
                        best_accuracy,
                    )

            # 训练完成
            final_metrics = {
                "best_f2": best_f2,
                "best_macro_f1": best_f2,
                "macro_f1_supported": best_f2,
                "best_accuracy": best_accuracy,
                "best_threshold": best_threshold,
                "best_epoch": best_epoch,
                "num_classes": len(DETAIL_TYPE_CLASS_ORDER),
                "class_names": list(DETAIL_TYPE_CLASS_ORDER),
                "selection_metric": str(imbalance_config["selection_metric"]),
                "class_support": class_support,
                "class_weights": weight_class,
                "sampler_weight_range": [
                    float(min(sample_weight_values)) if sample_weight_values else 0.0,
                    float(max(sample_weight_values)) if sample_weight_values else 0.0,
                ],
                "untrained_classes": untrained_classes,
                "unverifiable_classes": unverifiable_classes,
                "unreliable_classes": unreliable_classes,
                "promotion_warnings": class_support.get("promotion_warnings", []),
            }
            final_metrics.update(best_metrics)
            self.finished.emit(save_path, final_metrics)

        except Exception as e:
            logger.exception("训练失败")
            self.error.emit(f"训练失败: {e}")

    def stop(self) -> None:
        """请求停止训练"""
        self._should_stop = True
