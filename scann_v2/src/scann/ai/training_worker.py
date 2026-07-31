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
import hashlib
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PyQt5.QtCore import pyqtSignal, QObject, QThread
from scann.ai.cache_paths import configure_torch_cache

# 设置PyTorch模型下载路径到项目内（必须在导入torch之前设置）
try:
    model_file = Path(__file__).resolve()
    # training_worker.py 位于 scann_v2/src/scann/ai/training_worker.py，需要向上4级到 scann_v2/
    scann_v2_root = model_file.parent.parent.parent.parent
    model_cache_dir = configure_torch_cache(scann_v2_root)
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置环境变量
    os.environ['TORCH_HOME'] = str(model_cache_dir)
    os.environ['TORCH_HUB_DIR'] = str(model_cache_dir)
except Exception:
    pass

import torch  # 现在导入torch
import torch.nn.functional as F

from scann.ai.class_balance import (
    DETAIL_TYPE_CLASS_ORDER,
    DETAIL_TYPE_TO_CLASS_INDEX,
    build_class_audit,
    build_class_log_prior,
    compute_class_balanced_weights,
    compute_multiclass_metrics,
    compute_sample_quality,
    generate_variance_transfer_features,
    merge_imbalance_config,
    normalize_detail_type,
    sampler_weights_from_class_weights,
    stratified_group_train_val_split,
)
from scann.ai.device_utils import resolve_device
from scann.ai.feature_classifier import (
    FeatureHeadClassifier,
    apply_prior_logit_correction,
    feature_encoder_spec,
    forward_feature_encoder,
    load_feature_encoder,
    preprocess_feature_batch,
)
from scann.ai.hierarchical_classifier import (
    ACTION_CLASSES,
    DETAIL_CLASSES,
    FAMILY_CLASSES,
    HIERARCHICAL_MODEL_FORMAT,
    HierarchicalHeads,
    calibration_metrics,
    fit_temperature_scaling,
    hierarchical_loss,
    taxonomy_target_indices,
)
from scann.ai.model import ModelFormat, SCANNClassifier
from scann.ai.taxonomy import TAXONOMY_VERSION
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
                edge_margin = min(
                    float(center_x),
                    float(center_y),
                    max(0.0, float(new_u8_view.shape[1]) - float(center_x)),
                    max(0.0, float(new_u8_view.shape[0]) - float(center_y)),
                )
                record = {
                    "data": triplet,
                    "label": int(class_index),
                    "detail_type": detail_type,
                    "task_id": image_id,
                    "annotation_index": int(ann_index),
                    "bbox_width": float(ann.get("width", 0) or 0),
                    "bbox_height": float(ann.get("height", 0) or 0),
                    "confidence": float(ann.get("confidence", 1.0) or 1.0),
                    "patch_size": int(patch_size),
                    "edge_margin": float(edge_margin),
                }
                record["quality_score"] = compute_sample_quality(record)
                all_samples.append(
                    record
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

    def _resolve_training_mode(self, imbalance_config: dict[str, Any]) -> str:
        raw = self._params.get("training_mode", imbalance_config.get("training_mode", "end_to_end"))
        normalized = str(raw or "").strip().lower()
        if normalized in {
            "hierarchical_frozen",
            "hierarchical_frozen_feature",
            "hierarchical_v1",
        }:
            return "hierarchical_frozen"
        if normalized in {"frozen_feature_classifier", "frozen_features", "feature_head"}:
            return "frozen_feature_classifier"
        return "end_to_end"

    def _snapshot_cache_id(self) -> str:
        for key in ("snapshot_id", "annotations_document_path", "snapshot_document_relpath"):
            raw = str(self._params.get(key, "") or "").strip()
            if raw:
                stem = Path(raw).stem
                if stem:
                    return stem
        digest = hashlib.sha1(json.dumps(self._params, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        return f"live-{digest[:12]}"

    def _feature_cache_path(self, dataset_root: Path, feature_encoder: str) -> Path:
        safe_encoder = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "-" for ch in feature_encoder)
        return dataset_root / ".scann_control" / "feature_cache" / self._snapshot_cache_id() / f"{safe_encoder}.npz"

    @staticmethod
    def _feature_record_key(record: dict[str, Any]) -> str:
        return "|".join(
            [
                str(record.get("task_id", "")),
                str(record.get("annotation_index", "")),
                str(record.get("label", "")),
                str(record.get("detail_type", "")),
            ]
        )

    def _load_feature_cache(
        self,
        cache_path: Path,
        records: list[dict[str, Any]],
        *,
        feature_encoder: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if not cache_path.is_file():
            return None
        try:
            payload = np.load(cache_path, allow_pickle=False)
            if str(payload["feature_encoder"].item()) != feature_encoder:
                return None
            cached_keys = [str(item) for item in payload["keys"].tolist()]
            expected_keys = [self._feature_record_key(record) for record in records]
            if cached_keys != expected_keys:
                return None
            features = np.asarray(payload["features"], dtype=np.float32)
            labels = np.asarray(payload["labels"], dtype=np.int64)
            qualities = np.asarray(payload["qualities"], dtype=np.float32)
            if features.shape[0] != len(records) or labels.shape[0] != len(records) or qualities.shape[0] != len(records):
                return None
            logger.info("Feature cache hit: %s", cache_path)
            return features, labels, qualities
        except Exception:
            logger.warning("Ignoring invalid feature cache: %s", cache_path, exc_info=True)
            return None

    def _save_feature_cache(
        self,
        cache_path: Path,
        records: list[dict[str, Any]],
        *,
        feature_encoder: str,
        features: np.ndarray,
        labels: np.ndarray,
        qualities: np.ndarray,
    ) -> None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                feature_encoder=np.asarray(feature_encoder),
                keys=np.asarray([self._feature_record_key(record) for record in records]),
                features=np.asarray(features, dtype=np.float32),
                labels=np.asarray(labels, dtype=np.int64),
                qualities=np.asarray(qualities, dtype=np.float32),
            )
            logger.info("Feature cache saved: %s", cache_path)
        except Exception:
            logger.warning("Failed to save feature cache: %s", cache_path, exc_info=True)

    def _extract_features_for_records(
        self,
        records: list[dict[str, Any]],
        *,
        feature_encoder: str,
        batch_size: int,
        device: torch.device,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        encoder, spec = load_feature_encoder(feature_encoder, device=device)
        features: list[np.ndarray] = []
        labels: list[int] = []
        qualities: list[float] = []

        with torch.no_grad():
            for start in range(0, len(records), max(1, int(batch_size))):
                batch_records = records[start : start + max(1, int(batch_size))]
                tensors: list[torch.Tensor] = []
                for record in batch_records:
                    data = np.asarray(record["data"], dtype=np.float32)
                    if data.ndim != 3:
                        raise ValueError("feature records must contain CHW patch data")
                    tensor = torch.from_numpy(data).float().unsqueeze(0).to(device)
                    if tensor.shape[-2:] != (int(spec.input_size), int(spec.input_size)):
                        tensor = F.interpolate(
                            tensor,
                            size=(int(spec.input_size), int(spec.input_size)),
                            mode="bilinear",
                            align_corners=False,
                        )
                    tensors.append(tensor.squeeze(0))
                tensor = torch.stack(tensors, dim=0)
                prepared = preprocess_feature_batch(tensor, input_size=spec.input_size)
                encoded = forward_feature_encoder(encoder, prepared, family=spec.family)
                features.append(encoded.detach().cpu().numpy().astype(np.float32))
                labels.extend(int(record["label"]) for record in batch_records)
                qualities.extend(float(record.get("quality_score", compute_sample_quality(record))) for record in batch_records)

        return (
            np.concatenate(features, axis=0).astype(np.float32) if features else np.empty((0, spec.feature_dim), dtype=np.float32),
            np.asarray(labels, dtype=np.int64),
            np.asarray(qualities, dtype=np.float32),
        )

    def _extract_or_load_features(
        self,
        dataset_root: Path,
        records: list[dict[str, Any]],
        *,
        feature_encoder: str,
        batch_size: int,
        device: torch.device,
        cache_enabled: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path | None, bool]:
        cache_path = self._feature_cache_path(dataset_root, feature_encoder) if cache_enabled else None
        if cache_path is not None:
            cached = self._load_feature_cache(cache_path, records, feature_encoder=feature_encoder)
            if cached is not None:
                features, labels, qualities = cached
                return features, labels, qualities, cache_path, True

        features, labels, qualities = self._extract_features_for_records(
            records,
            feature_encoder=feature_encoder,
            batch_size=batch_size,
            device=device,
        )
        if cache_path is not None:
            self._save_feature_cache(
                cache_path,
                records,
                feature_encoder=feature_encoder,
                features=features,
                labels=labels,
                qualities=qualities,
            )
        return features, labels, qualities, cache_path, False

    @staticmethod
    def _feature_dbl_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        qualities: torch.Tensor,
        *,
        class_weights: list[float],
        dbl_config: dict[str, Any],
    ) -> torch.Tensor:
        logp = torch.log_softmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        target_index = targets.view(-1, 1)
        logp_t = logp.gather(1, target_index).squeeze(1)
        p_t = probs.gather(1, target_index).squeeze(1)

        gamma = float(dbl_config.get("focal_gamma", 2.0) or 0.0)
        base_loss = -logp_t * ((1.0 - p_t).clamp_min(0.0) ** gamma)

        class_weight_tensor = torch.as_tensor(class_weights, dtype=logits.dtype, device=logits.device)
        safe_targets = targets.view(-1).clamp(0, class_weight_tensor.numel() - 1)
        class_weight = class_weight_tensor.gather(0, safe_targets)

        if bool(dbl_config.get("enabled", True)):
            q_min = float(dbl_config.get("quality_min_weight", 0.35) or 0.0)
            q_max = float(dbl_config.get("quality_max_weight", 1.0) or 1.0)
            quality = qualities.to(dtype=logits.dtype, device=logits.device).clamp(0.0, 1.0)
            quality_weight = q_min + (q_max - q_min) * quality
        else:
            quality_weight = torch.ones_like(base_loss)

        weights = class_weight * quality_weight
        return (base_loss * weights).sum() / weights.sum().clamp_min(1e-12)

    @staticmethod
    def _attach_long_tail_score(metrics: dict[str, Any], imbalance_config: dict[str, Any]) -> float:
        raw_weights = imbalance_config.get("selection_metric_weights")
        weights = raw_weights if isinstance(raw_weights, dict) else {}
        default_weights = {
            "macro_f1_supported": 0.5,
            "tail_recall@1": 0.3,
            "macro_ap": 0.2,
        }
        score = 0.0
        for key, default_weight in default_weights.items():
            try:
                weight = float(weights.get(key, default_weight))
            except (TypeError, ValueError):
                weight = default_weight
            try:
                value = float(metrics.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            if not np.isfinite(weight) or not np.isfinite(value):
                continue
            score += weight * value
        metrics["long_tail_score"] = float(score)
        return float(score)

    @classmethod
    def _feature_selection_value(
        cls,
        metrics: dict[str, Any],
        metric_name: str,
        imbalance_config: dict[str, Any],
    ) -> float:
        normalized = str(metric_name or "macro_f1_supported").strip()
        if normalized in {"long_tail_score", "balanced_long_tail_score"}:
            raw_score = cls._attach_long_tail_score(metrics, imbalance_config)
        else:
            if "long_tail_score" not in metrics:
                cls._attach_long_tail_score(metrics, imbalance_config)
            try:
                raw_score = float(metrics.get(normalized, metrics.get("macro_f1_supported", 0.0)) or 0.0)
            except (TypeError, ValueError):
                raw_score = 0.0

        penalty = 0.0
        constraints = imbalance_config.get("selection_constraints")
        if isinstance(constraints, dict):
            for key, raw_floor in constraints.items():
                try:
                    floor = float(raw_floor)
                    value = float(metrics.get(str(key), 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(floor) or not np.isfinite(value):
                    continue
                if value < floor:
                    penalty += floor - value
        metrics["selection_raw_score"] = float(raw_score)
        metrics["selection_constraint_penalty"] = float(penalty)
        metrics["selection_constraints_met"] = bool(penalty <= 1e-12)
        if penalty > 0.0:
            return float(raw_score - 1.0 - penalty)
        return float(raw_score)

    @staticmethod
    def _format_feature_metric_summary(
        metrics: dict[str, Any],
        *,
        leading: tuple[str, float] | None = None,
    ) -> str:
        parts: list[str] = []
        seen: set[str] = set()

        def add(key: str, value: Any) -> None:
            if key in seen:
                return
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return
            if not np.isfinite(numeric):
                return
            seen.add(key)
            parts.append(f"{key}={numeric:.4f}")

        if leading is not None:
            add(str(leading[0]), leading[1])
        for key in ("macro_f1_supported", "tail_recall@1", "macro_ap", "long_tail_score", "selection_score"):
            if key in metrics:
                add(key, metrics[key])
        return ", ".join(parts)

    @staticmethod
    def _evaluate_feature_head(
        head: torch.nn.Module,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        device: torch.device,
        batch_size: int,
        class_log_prior: list[float],
        prior_config: dict[str, Any],
        class_count: int,
        tail_max_support: int,
    ) -> dict[str, Any]:
        head.eval()
        all_probs: list[np.ndarray] = []
        all_preds: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(labels), max(1, int(batch_size))):
                batch = torch.from_numpy(features[start : start + max(1, int(batch_size))]).float().to(device)
                logits = head(batch)
                if bool(prior_config.get("enabled", True)):
                    logits = apply_prior_logit_correction(
                        logits,
                        class_log_prior,
                        tau=float(prior_config.get("tau", 1.0) or 0.0),
                    )
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.detach().cpu().numpy())
                all_preds.append(torch.argmax(probs, dim=1).detach().cpu().numpy())
        probs_np = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, class_count), dtype=np.float32)
        preds_np = np.concatenate(all_preds, axis=0) if all_preds else np.empty((0,), dtype=np.int64)
        return compute_multiclass_metrics(
            labels,
            preds_np,
            probs=probs_np,
            class_count=class_count,
            tail_max_support=tail_max_support,
        )

    def _run_frozen_feature_training(
        self,
        *,
        dataset_root: Path,
        all_samples: list[dict[str, Any]],
        train_idx: list[int],
        val_idx: list[int],
        split_support: dict[str, list[int]],
        class_support: dict[str, Any],
        imbalance_config: dict[str, Any],
        device: torch.device,
        epochs: int,
        batch_size: int,
        lr: float,
        save_path: str,
        backbone_name: str,
    ) -> None:
        from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
        import torch.optim as optim

        class_count = len(DETAIL_TYPE_CLASS_ORDER)
        feature_encoder_raw = str(self._params.get("feature_encoder") or imbalance_config.get("feature_encoder") or "auto")
        spec = feature_encoder_spec(feature_encoder_raw, device=device)
        cache_enabled = bool(self._params.get("feature_cache_enabled", imbalance_config.get("feature_cache_enabled", True)))

        features, labels, qualities, cache_path, cache_hit = self._extract_or_load_features(
            dataset_root,
            all_samples,
            feature_encoder=spec.name,
            batch_size=batch_size,
            device=device,
            cache_enabled=cache_enabled,
        )
        if features.shape[0] != len(all_samples):
            raise ValueError("feature extraction returned an unexpected number of rows")

        train_features = features[train_idx]
        train_labels = labels[train_idx]
        train_qualities = qualities[train_idx]
        val_features = features[val_idx]
        val_labels = labels[val_idx]

        weight_class = compute_class_balanced_weights(
            train_labels.tolist(),
            beta=float(imbalance_config["class_weight_beta"]),
            clip=imbalance_config["class_weight_clip"],
            class_count=class_count,
        )
        prior_config = imbalance_config.get("prior_logit_correction", {})
        class_log_prior = build_class_log_prior(
            train_labels.tolist(),
            class_count=class_count,
            smoothing=float(prior_config.get("smoothing", 1.0) or 0.0),
        )

        vt_features, vt_labels, vt_summary = generate_variance_transfer_features(
            train_features,
            train_labels,
            class_count=class_count,
            config=imbalance_config.get("variance_transfer", {}),
            seed=int(self._params.get("seed", 42)),
        )
        if vt_features.size:
            train_features_aug = np.concatenate([train_features, vt_features], axis=0).astype(np.float32)
            train_labels_aug = np.concatenate([train_labels, vt_labels], axis=0).astype(np.int64)
            train_qualities_aug = np.concatenate(
                [train_qualities, np.ones((vt_labels.shape[0],), dtype=np.float32)],
                axis=0,
            ).astype(np.float32)
        else:
            train_features_aug = train_features.astype(np.float32)
            train_labels_aug = train_labels.astype(np.int64)
            train_qualities_aug = train_qualities.astype(np.float32)

        sample_weight_values = sampler_weights_from_class_weights(
            train_labels_aug.tolist(),
            weight_class,
            power=float(imbalance_config["sampler_power"]),
            max_ratio=float(imbalance_config["sampler_max_ratio"]),
        )
        sampler = WeightedRandomSampler(
            torch.tensor(sample_weight_values, dtype=torch.double),
            num_samples=len(train_labels_aug),
            replacement=True,
        )
        train_dataset = TensorDataset(
            torch.from_numpy(train_features_aug).float(),
            torch.from_numpy(train_labels_aug).long(),
            torch.from_numpy(train_qualities_aug).float(),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)

        head_config = self._params.get("feature_head") if isinstance(self._params.get("feature_head"), dict) else {}
        head = FeatureHeadClassifier(
            feature_dim=int(train_features_aug.shape[1]),
            num_classes=class_count,
            hidden_dim=int(head_config.get("hidden_dim") or 0),
            dropout=float(head_config.get("dropout") or 0.0),
        ).to(device)
        optimizer = optim.AdamW(
            head.parameters(),
            lr=float(self._params.get("feature_head_lr", lr) or lr),
            weight_decay=float(self._params.get("weight_decay", 1e-3) or 1e-3),
        )

        best_metric = -1.0
        best_metrics: dict[str, Any] = {}
        best_epoch = -1
        tail_max_support = int(imbalance_config.get("tail_recall_max_support", 20) or 20)

        for epoch in range(int(epochs)):
            if self._should_stop:
                break
            head.train()
            loss_sum = 0.0
            seen = 0
            for xb, yb, qb in train_loader:
                if self._should_stop:
                    break
                xb = xb.to(device)
                yb = yb.to(device)
                qb = qb.to(device)
                optimizer.zero_grad()
                logits = head(xb)
                loss = self._feature_dbl_loss(
                    logits,
                    yb,
                    qb,
                    class_weights=weight_class,
                    dbl_config=imbalance_config.get("dbl", {}),
                )
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.item()) * xb.size(0)
                seen += xb.size(0)

            train_loss = loss_sum / max(seen, 1)
            metrics = self._evaluate_feature_head(
                head,
                val_features.astype(np.float32),
                val_labels.astype(np.int64),
                device=device,
                batch_size=batch_size,
                class_log_prior=class_log_prior,
                prior_config=prior_config,
                class_count=class_count,
                tail_max_support=tail_max_support,
            )
            metric_name = str(imbalance_config.get("selection_metric") or "macro_f1_supported")
            metric_value = self._feature_selection_value(metrics, metric_name, imbalance_config)
            metrics["selection_score"] = float(metric_value)
            metric_summary = self._format_feature_metric_summary(metrics)
            self.progress.emit(epoch + 1, int(epochs), train_loss, metric_value)
            logger.info(
                "Frozen-feature epoch %d/%d: loss=%.4f, %s",
                epoch + 1,
                int(epochs),
                train_loss,
                metric_summary,
            )

            if metric_value > best_metric + 0.001:
                best_metric = metric_value
                best_epoch = epoch
                best_metrics = dict(metrics)
                ckpt = {
                    "state": {"head": head.state_dict()},
                    "head_state": head.state_dict(),
                    "model_format": "frozen_feature_classifier",
                    "task_type": "classification",
                    "training_mode": "frozen_feature_classifier",
                    "backbone": backbone_name,
                    "feature_encoder": spec.name,
                    "feature_dim": int(train_features_aug.shape[1]),
                    "feature_head_config": {
                        "hidden_dim": int(head_config.get("hidden_dim") or 0),
                        "dropout": float(head_config.get("dropout") or 0.0),
                    },
                    "class_names": list(DETAIL_TYPE_CLASS_ORDER),
                    "num_classes": class_count,
                    "class_counts": {
                        DETAIL_TYPE_CLASS_ORDER[index]: int((train_labels == index).sum())
                        for index in range(class_count)
                    },
                    "class_log_prior": class_log_prior,
                    "prior_correction_tau": float(prior_config.get("tau", 1.0) if prior_config.get("enabled", True) else 0.0),
                    "prior_logit_correction": prior_config,
                    "variance_transfer_config": imbalance_config.get("variance_transfer", {}),
                    "variance_transfer_summary": vt_summary,
                    "dbl_config": imbalance_config.get("dbl", {}),
                    "classification_mode": "detail_type_11",
                    "class_support": class_support,
                    "class_weights": weight_class,
                    "selection_metric": metric_name,
                    "selection_metric_weights": imbalance_config.get("selection_metric_weights", {}),
                    "selection_constraints": imbalance_config.get("selection_constraints", {}),
                    "selection_score": float(metric_value),
                    "best_epoch": int(best_epoch),
                    "best_metrics": best_metrics,
                    "feature_cache_path": str(cache_path) if cache_path is not None else "",
                    "feature_cache_hit": bool(cache_hit),
                }
                torch.save(ckpt, save_path)
                logger.info(
                    "Saved frozen-feature classifier (epoch=%d, %s, encoder=%s)",
                    epoch + 1,
                    self._format_feature_metric_summary(best_metrics, leading=("selection_score", metric_value)),
                    spec.name,
                )

        if best_epoch < 0:
            raise ValueError("frozen feature training did not produce a valid checkpoint")

        final_selection_metric = str(imbalance_config.get("selection_metric") or "macro_f1_supported")
        final_selection_score = self._feature_selection_value(best_metrics, final_selection_metric, imbalance_config)
        final_metrics = {
            "training_mode": "frozen_feature_classifier",
            "feature_encoder": spec.name,
            "feature_dim": int(train_features_aug.shape[1]),
            "best_epoch": best_epoch,
            "best_macro_f1": float(best_metrics.get("macro_f1_supported", 0.0)),
            "macro_f1_supported": float(best_metrics.get("macro_f1_supported", 0.0)),
            "macro_ap": float(best_metrics.get("macro_ap", 0.0)),
            "tail_recall@1": float(best_metrics.get("tail_recall@1", 0.0)),
            "long_tail_score": float(best_metrics.get("long_tail_score", 0.0)),
            "num_classes": class_count,
            "class_names": list(DETAIL_TYPE_CLASS_ORDER),
            "class_support": class_support,
            "class_weights": weight_class,
            "class_log_prior": class_log_prior,
            "selection_metric": final_selection_metric,
            "selection_metric_weights": imbalance_config.get("selection_metric_weights", {}),
            "selection_constraints": imbalance_config.get("selection_constraints", {}),
            "selection_score": float(final_selection_score),
            "variance_transfer_summary": vt_summary,
            "feature_cache_path": str(cache_path) if cache_path is not None else "",
            "feature_cache_hit": bool(cache_hit),
            "split_support": split_support,
            "promotion_warnings": class_support.get("promotion_warnings", []),
            "untrained_classes": class_support.get("untrained_classes", []),
            "unverifiable_classes": class_support.get("unverifiable_classes", []),
            "unreliable_classes": list(
                dict.fromkeys(
                    list(class_support.get("missing_classes", []))
                    + list(class_support.get("low_sample_classes", []))
                    + list(class_support.get("unverifiable_classes", []))
                )
            ),
        }
        final_metrics.update(best_metrics)
        logger.info(
            "Frozen-feature training best metrics: epoch=%d, %s",
            int(best_epoch) + 1,
            self._format_feature_metric_summary(final_metrics),
        )
        self.finished.emit(save_path, final_metrics)

    @staticmethod
    def _hierarchical_validation_metrics(
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for prefix, logits_key, target_key, class_names in (
            (
                "action",
                "review_action_logits",
                "review_action",
                ACTION_CLASSES,
            ),
            (
                "family",
                "phenomenon_family_logits",
                "phenomenon_family",
                FAMILY_CLASSES,
            ),
            (
                "detail",
                "detail_type_logits",
                "detail_type",
                DETAIL_CLASSES,
            ),
        ):
            class_count = len(class_names)
            target = targets[target_key]
            mask = target >= 0
            if not bool(mask.any()):
                metrics[f"{prefix}_support"] = 0
                continue
            probabilities = torch.softmax(
                outputs[logits_key][mask],
                dim=1,
            )
            predictions = torch.argmax(probabilities, dim=1)
            labels_np = target[mask].detach().cpu().numpy()
            predictions_np = predictions.detach().cpu().numpy()
            probabilities_np = probabilities.detach().cpu().numpy()
            per_class: dict[str, Any] = {}
            f1_values: list[float] = []
            ap_values: list[float] = []
            for class_index, class_name in enumerate(class_names):
                actual = labels_np == class_index
                predicted = predictions_np == class_index
                support = int(actual.sum())
                true_positive = int((actual & predicted).sum())
                false_positive = int((~actual & predicted).sum())
                false_negative = int((actual & ~predicted).sum())
                precision = (
                    true_positive / (true_positive + false_positive)
                    if true_positive + false_positive
                    else 0.0
                )
                recall = (
                    true_positive / (true_positive + false_negative)
                    if support
                    else 0.0
                )
                f1 = (
                    2.0 * precision * recall / (precision + recall)
                    if precision + recall
                    else 0.0
                )
                average_precision = 0.0
                if support:
                    try:
                        from sklearn.metrics import average_precision_score

                        average_precision = float(
                            average_precision_score(
                                actual.astype(np.int32),
                                probabilities_np[:, class_index],
                            )
                        )
                    except Exception:
                        average_precision = 0.0
                    f1_values.append(f1)
                    ap_values.append(average_precision)
                per_class[str(class_name)] = {
                    "support": support,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "ap": average_precision,
                }
            metrics[f"{prefix}_support"] = int(mask.sum().item())
            metrics[f"{prefix}_accuracy"] = float(
                (labels_np == predictions_np).mean()
            )
            metrics[f"{prefix}_macro_f1"] = (
                float(sum(f1_values) / len(f1_values))
                if f1_values
                else 0.0
            )
            metrics[f"{prefix}_macro_ap"] = (
                float(sum(ap_values) / len(ap_values))
                if ap_values
                else 0.0
            )
            metrics[f"{prefix}_per_class"] = per_class

        action_target = targets["review_action"]
        action_mask = action_target >= 0
        if bool(action_mask.any()):
            action_probabilities = torch.softmax(
                outputs["review_action_logits"][action_mask],
                dim=1,
            )
            action_predictions = torch.argmax(
                action_probabilities,
                dim=1,
            )
            keep_index = ACTION_CLASSES.index("keep")
            reject_index = ACTION_CLASSES.index("reject")
            actual_keep = action_target[action_mask] == keep_index
            predicted_keep = action_predictions == keep_index
            actual_reject = action_target[action_mask] == reject_index
            predicted_reject = action_predictions == reject_index
            keep_true_positive = int(
                (actual_keep & predicted_keep).sum().item()
            )
            metrics["keep_recall"] = (
                keep_true_positive / int(actual_keep.sum().item())
                if int(actual_keep.sum().item())
                else None
            )
            metrics["keep_precision"] = (
                keep_true_positive / int(predicted_keep.sum().item())
                if int(predicted_keep.sum().item())
                else None
            )
            reject_true_positive = int(
                (actual_reject & predicted_reject).sum().item()
            )
            metrics["reject_recall"] = (
                reject_true_positive / int(actual_reject.sum().item())
                if int(actual_reject.sum().item())
                else None
            )
        return metrics

    def _run_hierarchical_feature_training(
        self,
        *,
        dataset_root: Path,
        all_samples: list[dict[str, Any]],
        train_idx: list[int],
        val_idx: list[int],
        split_support: dict[str, list[int]],
        class_support: dict[str, Any],
        imbalance_config: dict[str, Any],
        device: torch.device,
        epochs: int,
        batch_size: int,
        lr: float,
        save_path: str,
        backbone_name: str,
    ) -> None:
        from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
        import torch.optim as optim

        feature_encoder_raw = str(
            self._params.get("feature_encoder")
            or imbalance_config.get("feature_encoder")
            or "auto"
        )
        spec = feature_encoder_spec(feature_encoder_raw, device=device)
        cache_enabled = bool(
            self._params.get(
                "feature_cache_enabled",
                imbalance_config.get("feature_cache_enabled", True),
            )
        )
        features, _flat_labels, qualities, cache_path, cache_hit = (
            self._extract_or_load_features(
                dataset_root,
                all_samples,
                feature_encoder=spec.name,
                batch_size=batch_size,
                device=device,
                cache_enabled=cache_enabled,
            )
        )
        target_rows = [
            taxonomy_target_indices(record.get("detail_type"))
            for record in all_samples
        ]
        targets_np = {
            key: np.asarray(
                [row[key] for row in target_rows],
                dtype=np.int64,
            )
            for key in (
                "review_action",
                "phenomenon_family",
                "detail_type",
            )
        }

        def class_weights_for(
            values: np.ndarray,
            class_count: int,
        ) -> list[float]:
            observed = [
                int(value)
                for value in values.tolist()
                if int(value) >= 0
            ]
            return compute_class_balanced_weights(
                observed,
                beta=float(imbalance_config["class_weight_beta"]),
                clip=imbalance_config["class_weight_clip"],
                class_count=class_count,
            )

        train_targets = {
            key: values[train_idx]
            for key, values in targets_np.items()
        }
        val_targets = {
            key: torch.from_numpy(values[val_idx]).long().to(device)
            for key, values in targets_np.items()
        }
        weight_values = {
            "review_action": class_weights_for(
                train_targets["review_action"],
                len(ACTION_CLASSES),
            ),
            "phenomenon_family": class_weights_for(
                train_targets["phenomenon_family"],
                len(FAMILY_CLASSES),
            ),
            "detail_type": class_weights_for(
                train_targets["detail_type"],
                len(DETAIL_CLASSES),
            ),
        }
        class_weight_tensors = {
            key: torch.as_tensor(
                value,
                dtype=torch.float32,
                device=device,
            )
            for key, value in weight_values.items()
        }
        detail_sample_weights = sampler_weights_from_class_weights(
            train_targets["detail_type"].tolist(),
            weight_values["detail_type"],
            power=float(imbalance_config["sampler_power"]),
            max_ratio=float(imbalance_config["sampler_max_ratio"]),
        )
        train_dataset = TensorDataset(
            torch.from_numpy(features[train_idx]).float(),
            torch.from_numpy(train_targets["review_action"]).long(),
            torch.from_numpy(train_targets["phenomenon_family"]).long(),
            torch.from_numpy(train_targets["detail_type"]).long(),
            torch.from_numpy(qualities[train_idx]).float(),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=WeightedRandomSampler(
                torch.as_tensor(
                    detail_sample_weights,
                    dtype=torch.double,
                ),
                num_samples=len(train_dataset),
                replacement=True,
            ),
            num_workers=0,
        )
        head_config = (
            self._params.get("hierarchical_head")
            if isinstance(self._params.get("hierarchical_head"), dict)
            else {}
        )
        hidden_dim = int(head_config.get("hidden_dim") or 256)
        dropout = float(head_config.get("dropout") or 0.1)
        heads = HierarchicalHeads(
            int(features.shape[1]),
            hidden_dim=hidden_dim,
            dropout=dropout,
        ).to(device)
        optimizer = optim.AdamW(
            heads.parameters(),
            lr=float(self._params.get("hierarchical_head_lr", lr) or lr),
            weight_decay=float(
                self._params.get("weight_decay", 1e-3) or 1e-3
            ),
        )
        focal_gamma = float(
            self._params.get("focal_gamma", 2.0) or 2.0
        )
        val_features = torch.from_numpy(
            features[val_idx]
        ).float().to(device)
        best_score = -float("inf")
        best_metrics: dict[str, Any] = {}
        best_epoch = -1

        for epoch in range(int(epochs)):
            if self._should_stop:
                break
            heads.train()
            total_loss = 0.0
            seen = 0
            for xb, action_y, family_y, detail_y, _quality in train_loader:
                xb = xb.to(device)
                batch_targets = {
                    "review_action": action_y.to(device),
                    "phenomenon_family": family_y.to(device),
                    "detail_type": detail_y.to(device),
                }
                optimizer.zero_grad()
                outputs = heads(xb)
                losses = hierarchical_loss(
                    outputs,
                    batch_targets,
                    focal_gamma=focal_gamma,
                    class_weights=class_weight_tensors,
                )
                losses["loss"].backward()
                optimizer.step()
                total_loss += float(losses["loss"].item()) * xb.shape[0]
                seen += xb.shape[0]
            train_loss = total_loss / max(seen, 1)

            heads.eval()
            with torch.no_grad():
                val_outputs = heads(val_features)
                val_losses = hierarchical_loss(
                    val_outputs,
                    val_targets,
                    focal_gamma=focal_gamma,
                    class_weights=class_weight_tensors,
                )
            metrics = self._hierarchical_validation_metrics(
                val_outputs,
                val_targets,
            )
            selection_score = (
                0.4 * float(metrics.get("action_macro_f1", 0.0))
                + 0.2 * float(metrics.get("family_macro_f1", 0.0))
                + 0.4 * float(metrics.get("detail_macro_f1", 0.0))
            )
            metrics["selection_metric"] = "hierarchical_composite"
            metrics["selection_score"] = selection_score
            metrics["val_loss"] = float(val_losses["loss"].item())
            self.progress.emit(
                epoch + 1,
                int(epochs),
                train_loss,
                selection_score,
            )
            if selection_score <= best_score + 0.0001:
                continue

            temperatures = {
                "review_action": fit_temperature_scaling(
                    val_outputs["review_action_logits"],
                    val_targets["review_action"],
                ),
                "phenomenon_family": fit_temperature_scaling(
                    val_outputs["phenomenon_family_logits"],
                    val_targets["phenomenon_family"],
                ),
                "detail_type": fit_temperature_scaling(
                    val_outputs["detail_type_logits"],
                    val_targets["detail_type"],
                ),
            }
            calibrated = {
                "review_action": calibration_metrics(
                    torch.softmax(
                        val_outputs["review_action_logits"]
                        / temperatures["review_action"],
                        dim=1,
                    ),
                    val_targets["review_action"],
                ),
                "phenomenon_family": calibration_metrics(
                    torch.softmax(
                        val_outputs["phenomenon_family_logits"]
                        / temperatures["phenomenon_family"],
                        dim=1,
                    ),
                    val_targets["phenomenon_family"],
                ),
                "detail_type": calibration_metrics(
                    torch.softmax(
                        val_outputs["detail_type_logits"]
                        / temperatures["detail_type"],
                        dim=1,
                    ),
                    val_targets["detail_type"],
                ),
            }
            best_score = selection_score
            best_epoch = epoch
            best_metrics = {
                **metrics,
                "calibration": calibrated,
            }
            checkpoint = {
                "model_format": HIERARCHICAL_MODEL_FORMAT,
                "task_type": "classification",
                "training_mode": "hierarchical_frozen",
                "backbone": backbone_name,
                "feature_encoder": spec.name,
                "feature_dim": int(features.shape[1]),
                "input_size": int(spec.input_size),
                "feature_version": str(
                    self._params.get(
                        "feature_version",
                        "frozen-image-feature-v1",
                    )
                ),
                "taxonomy_version": str(
                    self._params.get("taxonomy_version")
                    or TAXONOMY_VERSION
                ),
                "partition_id": str(
                    self._params.get("partition_id") or ""
                ),
                "partition_manifest_sha256": str(
                    self._params.get("partition_manifest_sha256")
                    or ""
                ),
                "head_config": {
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                },
                "head_states": {
                    "review_action_head": (
                        heads.review_action_head.state_dict()
                    ),
                    "phenomenon_family_head": (
                        heads.phenomenon_family_head.state_dict()
                    ),
                    "detail_type_head": (
                        heads.detail_type_head.state_dict()
                    ),
                },
                "classes": {
                    "review_action": list(ACTION_CLASSES),
                    "phenomenon_family": list(FAMILY_CLASSES),
                    "detail_type": list(DETAIL_CLASSES),
                },
                "class_names": list(DETAIL_CLASSES),
                "temperatures": temperatures,
                "class_weights": weight_values,
                "class_support": class_support,
                "split_support": split_support,
                "best_epoch": best_epoch,
                "best_metrics": best_metrics,
                "selection_metric": "hierarchical_composite",
                "selection_score": selection_score,
                "calibration_source": "validation",
                "gold_test_used_for_selection": False,
                "feature_cache_path": (
                    str(cache_path) if cache_path is not None else ""
                ),
                "feature_cache_hit": bool(cache_hit),
            }
            torch.save(checkpoint, save_path)

        if best_epoch < 0:
            raise ValueError(
                "hierarchical training did not produce a valid checkpoint"
            )
        final_metrics = {
            **best_metrics,
            "training_mode": "hierarchical_frozen",
            "best_epoch": best_epoch,
            "feature_encoder": spec.name,
            "taxonomy_version": str(
                self._params.get("taxonomy_version")
                or TAXONOMY_VERSION
            ),
            "partition_id": str(
                self._params.get("partition_id") or ""
            ),
            "gold_test_used_for_selection": False,
            "promotion_warnings": class_support.get(
                "promotion_warnings",
                [],
            ),
        }
        self.finished.emit(save_path, final_metrics)

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
            dataset_root = Path(str(self._params.get("dataset_dir", "")).strip())
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

            training_mode = self._resolve_training_mode(imbalance_config)
            if training_mode == "hierarchical_frozen":
                self._run_hierarchical_feature_training(
                    dataset_root=dataset_root,
                    all_samples=all_samples,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    split_support=split_support,
                    class_support=class_support,
                    imbalance_config=imbalance_config,
                    device=device,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    save_path=save_path,
                    backbone_name=backbone_name,
                )
                return

            if training_mode == "frozen_feature_classifier":
                self._run_frozen_feature_training(
                    dataset_root=dataset_root,
                    all_samples=all_samples,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    split_support=split_support,
                    class_support=class_support,
                    imbalance_config=imbalance_config,
                    device=device,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    save_path=save_path,
                    backbone_name=backbone_name,
                )
                return

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
                    probs_np = np.concatenate(all_probs)
                else:
                    labels = np.asarray([], dtype=np.int64)
                    preds = np.asarray([], dtype=np.int64)
                    probs_np = np.empty((0, class_count), dtype=np.float32)

                # 多分类验证指标
                metrics = compute_multiclass_metrics(labels, preds, probs=probs_np, class_count=class_count)
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
