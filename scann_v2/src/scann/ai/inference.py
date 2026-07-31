"""AI 推理引擎

职责:
- GPU 推理管理
- 显存控制 ≤ 8GB
- CUDA 多线程并行
- 分块推理防止 OOM
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from scann.ai.device_utils import get_mixed_precision_context, resolve_device
from scann.ai.feature_classifier import FrozenFeaturePatchClassifier
from scann.ai.hierarchical_classifier import (
    HIERARCHICAL_MODEL_FORMAT,
    FrozenFeatureHierarchicalClassifier,
    hierarchical_predictions,
)
from scann.ai.multimodal_classifier import (
    MULTIMODAL_MODEL_FORMAT,
    SharedEncoderLateFusionClassifier,
    build_structured_feature_matrix,
)
from scann.core.annotation_models import DETAIL_TYPE_TO_LABEL, DetailType
from scann.core.models import Candidate, Detection, MarkerType


logger = logging.getLogger(__name__)


DETAIL_TYPE_CLASS_ORDER: tuple[str, ...] = (
    DetailType.ASTEROID.value,
    DetailType.SUPERNOVA.value,
    DetailType.VARIABLE_STAR.value,
    DetailType.SATELLITE_TRAIL.value,
    DetailType.NOISE.value,
    DetailType.DIFFRACTION_SPIKE.value,
    DetailType.CMOS_CONDENSATION.value,
    DetailType.CORRESPONDING.value,
    DetailType.DISAPPEARED_ASTEROID.value,
    DetailType.DISAPPEARED_STAR.value,
    DetailType.DISAPPEARED_GALAXY.value,
)


@dataclass
class InferenceConfig:
    """推理配置"""
    batch_size: int = 64
    max_memory_mb: int = 8000  # 8GB 显存限制
    use_amp: bool = True       # 混合精度
    device: str = "auto"       # "auto", "cpu", "cuda[:idx]", "npu[:idx]", "mlu[:idx]", ...
    model_format: str = "auto" # "auto", "v1_classifier", "v2_classifier"
    model_backbone: str = "auto"  # "auto", "ResNet18", "ResNet34", "ResNet50", "ViT_B_16"


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
        self._class_names: list[str] | None = None

        if model_path:
            self._load_model(model_path)

    def _resolve_device(self) -> torch.device:
        resolved = resolve_device(self.config.device)
        self._resolved_device = resolved
        logger.info("推理设备: requested=%s resolved=%s", self.config.device, resolved.message)
        return resolved.resolved

    def _load_model(self, path: str) -> None:
        """加载模型 (自动检测 v1/v2 格式)

        v1: 使用原生 ResNet18 权重加载（不做 key 格式转换）
        v2: 使用 SCANNClassifier 兼容加载
        """
        import logging
        _logger = logging.getLogger(__name__)
        from scann.ai.model import ModelFormat, SCANNClassifier

        # 解析模型格式
        try:
            fmt = ModelFormat(self.config.model_format)
        except ValueError:
            fmt = ModelFormat.AUTO

        # 先读取 checkpoint 元数据，确定真实格式
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self._model_format = fmt
        self._model_backbone = str(getattr(self.config, "model_backbone", "auto") or "auto")

        # 默认阈值/通道顺序（后续可被 checkpoint 覆盖）
        self._threshold = 0.5
        self._channel_order = (0, 1, 2)

        if isinstance(ckpt, dict):
            raw_class_names = ckpt.get("class_names")
            if isinstance(raw_class_names, (list, tuple)):
                normalized_class_names = [str(item).strip().lower() for item in raw_class_names if str(item).strip()]
                if normalized_class_names:
                    self._class_names = normalized_class_names

            if (
                str(ckpt.get("model_format") or "").strip().lower() == "frozen_feature_classifier"
                or str(ckpt.get("training_mode") or "").strip().lower() == "frozen_feature_classifier"
            ):
                self._model_format = "frozen_feature_classifier"
                self._model_backbone = str(ckpt.get("backbone") or ckpt.get("feature_encoder") or "frozen_feature_classifier")
                if ckpt.get("threshold") is not None:
                    self.threshold = float(ckpt["threshold"])
                self.model = FrozenFeaturePatchClassifier.from_checkpoint(ckpt, device=self.device)
                self.model.eval()
                _logger.info("浣跨敤 frozen feature checkpoint 鍔犺浇: encoder=%s", ckpt.get("feature_encoder"))
                return

            if (
                str(ckpt.get("model_format") or "").strip().lower()
                == HIERARCHICAL_MODEL_FORMAT
            ):
                self._model_format = HIERARCHICAL_MODEL_FORMAT
                self._model_backbone = str(
                    ckpt.get("feature_encoder")
                    or "hierarchical_frozen_encoder"
                )
                if ckpt.get("threshold") is not None:
                    self.threshold = float(ckpt["threshold"])
                self.model = FrozenFeatureHierarchicalClassifier.from_checkpoint(
                    ckpt,
                    device=self.device,
                )
                self.model.eval()
                _logger.info(
                    "使用 hierarchical checkpoint 加载: encoder=%s",
                    ckpt.get("feature_encoder"),
                )
                return

            if (
                str(ckpt.get("model_format") or "").strip().lower()
                == MULTIMODAL_MODEL_FORMAT
            ):
                self._model_format = MULTIMODAL_MODEL_FORMAT
                self._model_backbone = str(
                    ckpt.get("feature_encoder")
                    or "multimodal_frozen_encoder"
                )
                self.model = (
                    SharedEncoderLateFusionClassifier.from_checkpoint(
                        ckpt,
                        device=self.device,
                    )
                )
                self.model.eval()
                _logger.info(
                    "使用 multimodal hierarchical checkpoint 加载: encoder=%s",
                    ckpt.get("feature_encoder"),
                )
                return

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

            # 若推理配置未显式指定 backbone，则优先使用 checkpoint 的 backbone 元数据
            ckpt_backbone = ckpt.get("backbone")
            if self._model_backbone.lower() == "auto" and isinstance(ckpt_backbone, str):
                self._model_backbone = ckpt_backbone

            # 读取阈值：优先 threshold (v2)，其次 t_recall (v1 训练脚本)
            if ckpt.get("threshold") is not None:
                self.threshold = float(ckpt["threshold"])
            elif ckpt.get("t_recall") is not None:
                self.threshold = float(ckpt["t_recall"])
                _logger.info("使用 V1 阈值 (t_recall): %.4f", self._threshold)

            # 读取 V1 的通道顺序 (order 字段, 如 "0,1,2")
            order_str = ckpt.get("order")
            if order_str and isinstance(order_str, str):
                try:
                    order = tuple(int(x.strip()) for x in order_str.split(","))
                    if len(order) == 3 and sorted(order) == [0, 1, 2]:
                        self._channel_order = order
                        _logger.info("使用 V1 通道顺序: %s", self._channel_order)
                except (ValueError, TypeError):
                    pass

        # 再按真实格式加载模型
        if self._model_format == ModelFormat.V1_CLASSIFIER:
            from torchvision import models as tv_models

            # 提取并清理 state_dict（仅去 module. 前缀，不做 v1->v2 转换）
            state_dict = ckpt.get("state") or ckpt.get("model_state") or ckpt if isinstance(ckpt, dict) else ckpt
            clean_state = {}
            if isinstance(state_dict, dict):
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith("module.") else k
                    # 兼容少数带 backbone. 前缀的权重
                    if name.startswith("backbone."):
                        name = name[len("backbone."):]
                    clean_state[name] = v

            has_weight_keys = any(
                k.startswith(("conv1.", "bn1.", "layer1.", "layer2.", "layer3.", "layer4.", "fc."))
                for k in clean_state.keys()
            )

            if has_weight_keys:
                model = tv_models.resnet18(weights=None)
                model.fc = torch.nn.Linear(model.fc.in_features, 2)
                model.load_state_dict(clean_state, strict=True)
                model.to(self.device)
                model.eval()
                self.model = model
                _logger.info("使用 v1 原生权重加载（未进行 v1->v2 key 转换）")
            else:
                # 仅元数据或无法识别权重时，回退兼容加载路径
                _logger.warning("v1 checkpoint 未找到可识别权重键，回退兼容加载")
                self.model = SCANNClassifier.load_from_checkpoint(
                    path,
                    self.device,
                    model_format=self._model_format,
                    backbone_name=self._model_backbone,
                )
        else:
            self.model = SCANNClassifier.load_from_checkpoint(
                path,
                self.device,
                model_format=self._model_format,
                backbone_name=self._model_backbone,
            )

    # V1 归一化常数（按 SCANN.py 迁移）
    V1_NORMALIZE_MEAN = (0.2601623164967817, 0.2682929013103806, 0.26861570225529907)
    V1_NORMALIZE_STD = (0.09133092247248126, 0.10773878132887775, 0.10867911864809723)
    # V2 默认归一化常数
    V2_NORMALIZE_MEAN = (0.26, 0.27, 0.27)
    V2_NORMALIZE_STD = (0.09, 0.11, 0.11)

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        """设置推理阈值（自动夹紧到 [0, 1]）。"""
        v = float(value)
        if not np.isfinite(v):
            return
        self._threshold = max(0.0, min(1.0, v))

    @property
    def model_format(self):
        """当前加载的模型格式"""
        return self._model_format

    @property
    def model_backbone(self) -> str:
        """当前加载模型的骨干网络提示。"""
        return getattr(self, "_model_backbone", "auto")

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
        details = self.classify_patches_detailed(
            patches,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )
        return [float(item.get("score", 0.0) or 0.0) for item in details]

    def _resolve_class_names(self, class_count: int) -> list[str]:
        if self._class_names and len(self._class_names) == class_count:
            return list(self._class_names)
        if class_count == len(DETAIL_TYPE_CLASS_ORDER):
            return list(DETAIL_TYPE_CLASS_ORDER)
        if class_count == 2:
            return ["bogus", "real"]
        return [f"class_{index}" for index in range(class_count)]

    @staticmethod
    def _normalize_label_from_class_name(class_name: str) -> str | None:
        normalized = str(class_name or "").strip().lower()
        if normalized in {"real", "bogus"}:
            return normalized
        try:
            detail_type = DetailType(normalized)
            return DETAIL_TYPE_TO_LABEL[detail_type].value
        except Exception:
            return None

    @staticmethod
    def _normalize_detail_type_from_class_name(class_name: str) -> str | None:
        normalized = str(class_name or "").strip().lower()
        try:
            return DetailType(normalized).value
        except Exception:
            return None

    @torch.no_grad()
    def classify_patches_detailed(
        self,
        patches: List[np.ndarray],
        normalize_mean: Optional[tuple] = None,
        normalize_std: Optional[tuple] = None,
        structured_features: Optional[
            Sequence[Mapping[str, Any]]
        ] = None,
    ) -> List[dict[str, object]]:
        if not self.is_ready:
            raise RuntimeError("模型未加载")
        if not patches:
            return []
        assert self.model is not None

        if normalize_mean is None or normalize_std is None:
            if self.is_v1:
                normalize_mean = self.V1_NORMALIZE_MEAN
                normalize_std = self.V1_NORMALIZE_STD
            else:
                normalize_mean = self.V2_NORMALIZE_MEAN
                normalize_std = self.V2_NORMALIZE_STD

        norm = transforms.Normalize(list(normalize_mean), list(normalize_std))
        resize = transforms.Resize((224, 224), antialias=True)
        internal_preprocessing_flag = getattr(
            self.model,
            "uses_internal_preprocessing",
            False,
        )
        uses_internal_preprocessing = (
            bool(internal_preprocessing_flag)
            if isinstance(internal_preprocessing_flag, (bool, np.bool_))
            else False
        )
        multimodal_flag = getattr(self.model, "is_multimodal", False)
        is_multimodal = (
            bool(multimodal_flag)
            if isinstance(multimodal_flag, (bool, np.bool_))
            else False
        )

        all_details: list[dict[str, object]] = []
        batch_size = self.config.batch_size

        for i in range(0, len(patches), batch_size):
            batch_raw = patches[i : i + batch_size]
            if is_multimodal:
                views: list[torch.Tensor] = []
                for patch in batch_raw:
                    tensor = torch.from_numpy(patch).float()
                    if tensor.ndim != 3 or tensor.shape[0] != 3:
                        raise ValueError(
                            "multimodal classifier expects "
                            "[new,old,difference] patch channels"
                        )
                    views.append(
                        torch.stack(
                            [
                                tensor[index].unsqueeze(0).repeat(3, 1, 1)
                                for index in range(3)
                            ],
                            dim=0,
                        )
                    )
                batch_records = (
                    list(structured_features[i : i + len(batch_raw)])
                    if structured_features is not None
                    else [{} for _ in batch_raw]
                )
                feature_names = tuple(
                    self.model.feature_normalization.feature_names
                )
                structured_values, structured_mask = (
                    build_structured_feature_matrix(
                        batch_records,
                        feature_names=feature_names,
                    )
                )
                logits = self.model(
                    torch.stack(views).to(self.device),
                    torch.from_numpy(structured_values)
                    .float()
                    .to(self.device),
                    torch.from_numpy(structured_mask)
                    .bool()
                    .to(self.device),
                )
                all_details.extend(hierarchical_predictions(logits))
                continue

            tensors = []
            for p in batch_raw:
                t = torch.from_numpy(p).float()
                if uses_internal_preprocessing:
                    target_size = int(getattr(getattr(self.model, "spec", None), "input_size", 224) or 224)
                    if t.ndim != 3:
                        raise ValueError("frozen feature classifier expects CHW patch data")
                    if t.shape[-2:] != (target_size, target_size):
                        t = F.interpolate(
                            t.unsqueeze(0),
                            size=(target_size, target_size),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0)
                    tensors.append(t)
                    continue
                if self.is_v1:
                    t = t.unsqueeze(0)
                    t = F.interpolate(
                        t,
                        size=(224, 224),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                    t = norm(t)
                else:
                    t = resize(t)
                    t = norm(t)
                tensors.append(t)

            stack = torch.stack(tensors).to(self.device)
            if self.config.use_amp:
                with get_mixed_precision_context(self.device, enabled=True):
                    logits = self.model(stack)
            else:
                logits = self.model(stack)

            if isinstance(logits, dict) and {
                "review_action_logits",
                "phenomenon_family_logits",
                "detail_type_logits",
            }.issubset(logits):
                all_details.extend(hierarchical_predictions(logits))
                continue

            probs = torch.softmax(logits, dim=1)
            class_count = int(probs.shape[1]) if probs.ndim == 2 else 2
            class_names = self._resolve_class_names(class_count)
            probs_np = probs.detach().cpu().numpy()

            for row in probs_np:
                if row.size == 0:
                    all_details.append({"score": 0.0, "label": None, "detail_type": None})
                    continue

                top_index = int(np.argmax(row))
                top_confidence = float(row[top_index])
                predicted_class = class_names[top_index] if top_index < len(class_names) else f"class_{top_index}"

                if class_count == 2:
                    score = float(row[1])
                else:
                    score = float(
                        sum(
                            float(prob)
                            for index, prob in enumerate(row)
                            if self._normalize_label_from_class_name(class_names[index]) == "real"
                        )
                    )

                all_details.append(
                    {
                        "score": score,
                        "predicted_class": predicted_class,
                        "predicted_confidence": top_confidence,
                        "label": self._normalize_label_from_class_name(predicted_class),
                        "detail_type": self._normalize_detail_type_from_class_name(predicted_class),
                    }
                )

        return all_details

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

    @staticmethod
    def _normalize_dense_channel(image: np.ndarray) -> np.ndarray:
        """将单通道图像归一化到 [0, 1]，用于 dense 输入。"""
        img = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if img.size == 0:
            return img

        v_min = float(np.min(img))
        v_max = float(np.max(img))
        if not np.isfinite(v_min) or not np.isfinite(v_max) or v_max <= v_min:
            return np.zeros_like(img, dtype=np.float32)

        img = (img - v_min) / (v_max - v_min)
        return np.clip(img, 0.0, 1.0).astype(np.float32)

    @torch.no_grad()
    def detect_dense_full_image(
        self,
        new_image: np.ndarray,
        old_image: Optional[np.ndarray] = None,
        score_threshold: float = 0.5,
        top_k: int = 200,
        iou_threshold: float = 0.5,
    ) -> List[Detection]:
        """全图 dense 检测入口。

        说明:
        - 输入构造为三通道: [diff, new, old]
        - 调用检测模型 `forward_dense()` 执行前向
        - 解码输出并执行 NMS 合并
        """
        if self.model is None:
            logger.warning("dense 检测不可用：模型未加载")
            return []

        if new_image is None or np.size(new_image) == 0:
            logger.warning("dense 检测跳过：new_image 为空")
            return []

        new_data = np.asarray(new_image)
        if new_data.ndim < 2:
            logger.warning("dense 检测跳过：new_image 维度无效 (ndim=%s)", new_data.ndim)
            return []
        if new_data.ndim > 2:
            new_data = new_data.squeeze()
            if new_data.ndim != 2:
                logger.warning("dense 检测跳过：new_image 无法转换为 2D")
                return []

        if old_image is None:
            old_data = np.zeros_like(new_data)
        else:
            old_data = np.asarray(old_image)
            if old_data.ndim < 2:
                logger.warning("dense 检测跳过：old_image 维度无效 (ndim=%s)", old_data.ndim)
                return []
            if old_data.ndim > 2:
                old_data = old_data.squeeze()
                if old_data.ndim != 2:
                    logger.warning("dense 检测跳过：old_image 无法转换为 2D")
                    return []

        height = min(int(new_data.shape[0]), int(old_data.shape[0]))
        width = min(int(new_data.shape[1]), int(old_data.shape[1]))
        if height <= 0 or width <= 0:
            logger.warning("dense 检测跳过：输入尺寸无效 new=%s old=%s", new_data.shape, old_data.shape)
            return []

        new_data = new_data[:height, :width]
        old_data = old_data[:height, :width]

        new_norm = self._normalize_dense_channel(new_data)
        old_norm = self._normalize_dense_channel(old_data)
        diff = np.abs(new_norm - old_norm).astype(np.float32)

        dense_input = np.stack([diff, new_norm, old_norm], axis=0).astype(np.float32)
        input_tensor = torch.from_numpy(dense_input).unsqueeze(0).to(self.device)

        dense_forward = getattr(self.model, "forward_dense", None)
        if not callable(dense_forward):
            logger.warning("dense 检测不可用：当前模型不支持 forward_dense，返回空结果")
            return []

        try:
            if self.config.use_amp:
                with get_mixed_precision_context(self.device, enabled=True):
                    dense_output = dense_forward(input_tensor)
            else:
                dense_output = dense_forward(input_tensor)
        except Exception:
            logger.exception("dense 检测前向失败，返回空结果")
            return []

        if not isinstance(dense_output, torch.Tensor):
            logger.warning("dense 检测前向输出类型无效: %s", type(dense_output).__name__)
            return []

        if dense_output.ndim != 4 or dense_output.shape[1] < 5:
            logger.warning("dense 检测前向输出形状无效: %s", tuple(dense_output.shape))
            return []

        detections = self._decode_dense_predictions(
            dense_output=dense_output,
            image_height=height,
            image_width=width,
            score_threshold=score_threshold,
            top_k=top_k,
        )

        if len(detections) > 1:
            detections = self._nms(detections, iou_threshold=iou_threshold)

        logger.info(
            "dense 检测执行完成: input=%s output=%s threshold=%.3f top_k=%d iou=%.3f detections=%d",
            tuple(input_tensor.shape),
            tuple(dense_output.shape),
            float(score_threshold),
            int(top_k),
            float(iou_threshold),
            len(detections),
        )
        return detections

    def _decode_dense_predictions(
        self,
        dense_output: torch.Tensor,
        image_height: int,
        image_width: int,
        score_threshold: float,
        top_k: int,
    ) -> List[Detection]:
        """将 dense 输出 [B,5,Hp,Wp] 解码为 Detection 列表。"""
        if dense_output.ndim != 4 or dense_output.shape[1] < 5:
            return []

        pred = dense_output[0]
        hp = int(pred.shape[1])
        wp = int(pred.shape[2])
        if hp <= 0 or wp <= 0:
            return []

        heatmap_logits = pred[4]
        score_map = torch.sigmoid(heatmap_logits)
        flat_scores = score_map.reshape(-1)
        if flat_scores.numel() == 0:
            return []

        k = max(1, min(int(top_k), int(flat_scores.numel())))
        top_scores, top_indices = torch.topk(flat_scores, k=k)
        keep_mask = top_scores > float(score_threshold)
        if not bool(torch.any(keep_mask)):
            return []

        top_scores = top_scores[keep_mask]
        top_indices = top_indices[keep_mask]

        stride_x = float(image_width) / float(wp)
        stride_y = float(image_height) / float(hp)

        bbox = pred[0:4]
        detections: List[Detection] = []
        for score, flat_idx in zip(top_scores, top_indices):
            idx = int(flat_idx.item())
            row = idx // wp
            col = idx % wp

            dx = float(torch.tanh(bbox[0, row, col]).item()) * 0.5
            dy = float(torch.tanh(bbox[1, row, col]).item()) * 0.5
            width_scale = float(torch.exp(torch.clamp(bbox[2, row, col], min=-2.0, max=2.0)).item())
            height_scale = float(torch.exp(torch.clamp(bbox[3, row, col], min=-2.0, max=2.0)).item())

            center_x = (float(col) + 0.5 + dx) * stride_x
            center_y = (float(row) + 0.5 + dy) * stride_y
            center_x = min(max(center_x, 0.0), max(float(image_width - 1), 0.0))
            center_y = min(max(center_y, 0.0), max(float(image_height - 1), 0.0))

            box_width = max(1.0, width_scale * stride_x)
            box_height = max(1.0, height_scale * stride_y)
            box_width = min(box_width, float(max(1, image_width)))
            box_height = min(box_height, float(max(1, image_height)))

            detections.append(
                Detection(
                    x=int(round(center_x)),
                    y=int(round(center_y)),
                    width=int(round(box_width)),
                    height=int(round(box_height)),
                    confidence=float(score.item()),
                    marker_type=MarkerType.BOUNDING_BOX,
                )
            )

        return detections

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

