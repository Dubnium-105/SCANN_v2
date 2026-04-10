"""AI 训练数据集"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from torchvision import transforms

from scann.core.fits_annotation_storage import load_v2_annotation_document
from scann.data.file_manager import FitsImagePair


logger = logging.getLogger(__name__)


class TripletDataset:
    """三联图数据集 (兼容 v1 PNG 格式)

    读取 80x240 PNG 三联图，切分为三通道输入。
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        indices: Optional[List[int]] = None,
        channel_order: Tuple[int, int, int] = (0, 1, 2),
        resize: int = 224,
        mean: Tuple[float, ...] = (0.264, 0.282, 0.284),
        std: Tuple[float, ...] = (0.089, 0.123, 0.128),
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.channel_order = channel_order
        self.resize = resize
        self.mean = mean
        self.std = std

        # 收集样本
        self.samples: List[Tuple[str, int]] = []
        for label_name, y in [("negative", 0), ("positive", 1)]:
            folder = self.root_dir / label_name
            if not folder.is_dir():
                continue
            for fn in sorted(folder.iterdir()):
                if fn.suffix.lower() == ".png":
                    self.samples.append((str(fn), y))

        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        parts = self._read_triplet(path)
        # 后续处理由 PyTorch DataLoader 端完成
        return parts, y

    def _read_triplet(self, path: str) -> List[np.ndarray]:
        """读取三联图并切分"""
        from PIL import Image

        im = Image.open(path).convert("L")
        w, h = im.size
        if w < 240 or h < 80:
            raise ValueError(f"尺寸不符: {w}x{h} for {path}")

        parts = [
            np.array(im.crop((0, 0, 80, 80)), dtype=np.float32) / 255.0,
            np.array(im.crop((80, 0, 160, 80)), dtype=np.float32) / 255.0,
            np.array(im.crop((160, 0, 240, 80)), dtype=np.float32) / 255.0,
        ]
        return [parts[i] for i in self.channel_order]

    def get_label_counts(self) -> dict:
        """统计各类别数量"""
        counts = {0: 0, 1: 0}
        for _, y in self.samples:
            counts[y] = counts.get(y, 0) + 1
        return counts


class TripletPNGDataset:
    """三联图 PyTorch 数据集 (支持 transform 和 tensor 返回)

    与 TripletDataset 类似，但返回 PyTorch tensor 格式，
    并支持数据增强和归一化。
    """

    def __init__(
        self,
        root_dir: str = "",  # 仅用于兼容，实际使用 samples
        split: str = "train",
        indices: Optional[List[int]] = None,
        channel_order: Tuple[int, int, int] = (0, 1, 2),
        resize: int = 224,
        mean: Tuple[float, ...] = (0.264, 0.282, 0.284),
        std: Tuple[float, ...] = (0.089, 0.123, 0.128),
        augment: bool = True,
        samples: Optional[List[Tuple[str, int]]] = None,  # 新增：支持传入预构建的样本列表
    ):
        self.root_dir = Path(root_dir) if root_dir else Path("")
        self.split = split
        self.channel_order = channel_order
        self.resize = resize
        self.mean = mean
        self.std = std
        self.augment = augment and (split == "train")

        # 收集样本（优先使用传入的 samples）
        self.samples: List[Tuple[str, int]] = []
        if samples is not None:
            self.samples = samples
        else:
            # 兼容旧逻辑：从目录收集
            for label_name, y in [("negative", 0), ("positive", 1)]:
                folder = self.root_dir / label_name
                if not folder.is_dir():
                    continue
                for fn in sorted(folder.iterdir()):
                    if fn.suffix.lower() == ".png":
                        self.samples.append((str(fn), y))

        if indices is not None:
            # 应用索引筛选
            self.samples = [self.samples[i] for i in indices]

        # 基础 transform
        self.base_transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
        ])
        self.normalize = transforms.Normalize(list(mean), list(std))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import torch
        from torchvision.transforms import functional as TF

        path, y = self.samples[idx]
        parts = self._read_triplet_images(path)

        # 转换为 tensor
        tensors = [self.base_transform(p) for p in parts]
        x = torch.cat(tensors, dim=0)  # [3, H, W]

        # 数据增强
        if self.augment:
            if random.random() < 0.5:
                x = TF.hflip(x)
            if random.random() < 0.5:
                x = TF.vflip(x)
            k = random.randint(0, 3)
            if k > 0:
                x = torch.rot90(x, k, dims=[1, 2])

        # 归一化
        x = self.normalize(x)

        return x, torch.tensor(y, dtype=torch.long)

    def _read_triplet_images(self, path: str) -> List:
        """读取三联图并切分（返回 PIL Image）"""
        from PIL import Image

        im = Image.open(path).convert("L")
        w, h = im.size
        if w < 240 or h < 80:
            raise ValueError(f"尺寸不符: {w}x{h} for {path}")

        parts = [
            im.crop((0, 0, 80, 80)),
            im.crop((80, 0, 160, 80)),
            im.crop((160, 0, 240, 80)),
        ]
        return [parts[i] for i in self.channel_order]

    def get_label_counts(self) -> dict:
        """统计各类别数量"""
        counts = {0: 0, 1: 0}
        for _, y in self.samples:
            counts[y] = counts.get(y, 0) + 1
        return counts


class TripletArrayDataset:
    """内存三通道 patch 数据集。

    用于承载从 v2 new/old + annotations.json 动态提取的三通道样本，
    保持与 TripletPNGDataset 相同的增强和归一化流程。
    """

    def __init__(
        self,
        samples: List[Tuple[np.ndarray, int]],
        split: str = "train",
        resize: int = 224,
        mean: Tuple[float, ...] = (0.264, 0.282, 0.284),
        std: Tuple[float, ...] = (0.089, 0.123, 0.128),
        augment: bool = True,
    ):
        self.samples = samples
        self.split = split
        self.resize = resize
        self.mean = mean
        self.std = std
        self.augment = augment and (split == "train")
        self.normalize = transforms.Normalize(list(mean), list(std))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import torch
        from torchvision.transforms import functional as TF

        triplet, y = self.samples[idx]
        x = torch.tensor(triplet, dtype=torch.float32)
        x = TF.resize(x, [self.resize, self.resize])

        if self.augment:
            if random.random() < 0.5:
                x = TF.hflip(x)
            if random.random() < 0.5:
                x = TF.vflip(x)
            k = random.randint(0, 3)
            if k > 0:
                x = torch.rot90(x, k, dims=[1, 2])

        x = self.normalize(x)
        return x, torch.tensor(y, dtype=torch.long)

    def get_label_counts(self) -> dict:
        """统计各类别数量"""
        counts = {0: 0, 1: 0}
        for _, y in self.samples:
            counts[y] = counts.get(y, 0) + 1
        return counts


class FitsDetectionDataset:
    """FITS 全图检测数据集 (v2 新格式)

    支持从 FITS 图像中提取滑动窗口 patches 用于目标检测训练。
    """

    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        patch_size: int = 224,
        stride: int = 112,
        label_map: Optional[dict] = None,
    ):
        """
        Args:
            image_dir: FITS 图像目录
            annotation_file: JSON 格式的标注文件
            patch_size: 提取的 patch 大小
            stride: 滑动窗口步长
            label_map: 标签映射字典，如 {"real": 1, "bogus": 0}
        """
        self.image_dir = Path(image_dir)
        self.annotation_file = Path(annotation_file)
        self.patch_size = patch_size
        self.stride = stride
        self.label_map = label_map or {"real": 1, "bogus": 0}

        # 加载标注
        self.samples = self._load_annotations()

    def _load_annotations(self) -> list:
        """从 JSON 文件加载标注"""
        import json

        with open(self.annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples = []
        for img_info in data.get("images", []):
            img_path = self.image_dir / img_info["file"]
            if not img_path.exists():
                continue

            samples.append({
                "image": str(img_path),
                "width": img_info["width"],
                "height": img_info["height"],
                "annotations": img_info.get("annotations", []),
            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """获取指定索引的样本

        Returns:
            (patch, targets) tuple
            patch: (3, patch_size, patch_size) numpy array
            targets: list of [x_center, y_center, width, height, class_id]
        """
        from scann.core.fits_io import read_fits

        sample = self.samples[idx]

        # 读取 FITS 图像
        fits_image = read_fits(sample["image"])
        image = fits_image.data

        # 默认提取中心区域的 patch
        crop_box = self._get_center_crop_box(sample["width"], sample["height"])
        patch = self._extract_patch(image, crop_box)

        # 转换标注到 patch 坐标系
        targets = self._annotations_to_targets(sample["annotations"], crop_box)

        return patch, targets

    def get_crop(self, idx: int, x: int, y: int, size: int):
        """获取指定位置的 crop

        Args:
            idx: 图像索引
            x: 左上角 x 坐标
            y: 左上角 y 坐标
            size: crop 大小

        Returns:
            (patch, targets) tuple
        """
        from scann.core.fits_io import read_fits

        sample = self.samples[idx]
        fits_image = read_fits(sample["image"])
        image = fits_image.data

        crop_box = (x, y, x + size, y + size)
        patch = self._extract_patch(image, crop_box)
        targets = self._annotations_to_targets(sample["annotations"], crop_box)

        return patch, targets

    def iter_patches(self, idx: int):
        """迭代图像的所有滑动窗口 patches

        Args:
            idx: 图像索引

        Yields:
            (patch, targets) tuples
        """
        from scann.core.fits_io import read_fits

        sample = self.samples[idx]
        fits_image = read_fits(sample["image"])
        image = fits_image.data

        # 计算所有可能的 crop 位置
        for y in range(0, sample["height"] - self.patch_size + 1, self.stride):
            for x in range(0, sample["width"] - self.patch_size + 1, self.stride):
                crop_box = (x, y, x + self.patch_size, y + self.patch_size)
                patch = self._extract_patch(image, crop_box)
                targets = self._annotations_to_targets(sample["annotations"], crop_box)
                yield patch, targets

    def _get_center_crop_box(self, width: int, height: int) -> tuple:
        """获取中心区域的 crop box"""
        x0 = max(0, (width - self.patch_size) // 2)
        y0 = max(0, (height - self.patch_size) // 2)
        x1 = min(width, x0 + self.patch_size)
        y1 = min(height, y0 + self.patch_size)
        return (x0, y0, x1, y1)

    def _extract_patch(self, image: np.ndarray, crop_box: tuple) -> np.ndarray:
        """从图像中提取 patch 并归一化

        Args:
            image: (H, W) 输入图像
            crop_box: (x0, y0, x1, y1) crop 区域

        Returns:
            (3, patch_size, patch_size) 归一化的 patch
        """
        x0, y0, x1, y1 = crop_box

        # 裁剪
        patch = image[y0:y1, x0:x1].astype(np.float32)

        # 如果裁剪尺寸不足，使用 padding
        if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
            padded = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
            h, w = patch.shape
            padded[:h, :w] = patch
            patch = padded

        # 归一化到 0-1
        if patch.max() > patch.min():
            patch = (patch - patch.min()) / (patch.max() - patch.min())

        # 调整大小到目标尺寸
        if patch.shape != (self.patch_size, self.patch_size):
            from skimage.transform import resize
            patch = resize(
                patch,
                (self.patch_size, self.patch_size),
                order=1,
                preserve_range=True,
                anti_aliasing=False
            )
            patch = patch.astype(np.float32)

        # 扩展为三通道（如果需要多通道输入）
        # 这里简单复制为三通道
        patch_3ch = np.stack([patch, patch, patch], axis=0)

        return patch_3ch

    def _annotations_to_targets(
        self,
        annotations: list,
        crop_box: tuple,
    ) -> list:
        """将标注转换为训练目标格式

        Args:
            annotations: 原始标注列表
            crop_box: (x0, y0, x1, y1) crop 区域

        Returns:
            list of [x_center, y_center, width, height, class_id]
            坐标已归一化到 [0, 1]
        """
        x0, y0, x1, y1 = crop_box
        crop_width = x1 - x0
        crop_height = y1 - y0

        targets = []
        for ann in annotations:
            # 检查标注是否在 crop 区域内
            ann_x = ann["x"]
            ann_y = ann["y"]
            ann_w = ann["width"]
            ann_h = ann["height"]

            # 计算重叠
            # 简化处理：如果中心点在 crop 区域内
            center_x = ann_x + ann_w / 2
            center_y = ann_y + ann_h / 2

            if center_x < x0 or center_x > x1 or center_y < y0 or center_y > y1:
                continue  # 不在 crop 区域内

            # 转换到 crop 坐标系
            rel_x = (center_x - x0) / crop_width
            rel_y = (center_y - y0) / crop_height
            rel_w = ann_w / crop_width
            rel_h = ann_h / crop_height

            # 映射标签
            label = ann.get("label", "real")
            class_id = self.label_map.get(label, 0)

            targets.append([rel_x, rel_y, rel_w, rel_h, class_id])

        return targets

    def get_label_counts(self) -> dict:
        """统计各类别的标注数量"""
        counts = {}
        for sample in self.samples:
            for ann in sample["annotations"]:
                label = ann.get("label", "real")
                counts[label] = counts.get(label, 0) + 1
        return counts


class FitsDenseDetectionDataset:
    """v2 全图 dense 检测训练数据集。

    输入输出约定:
    - 输入: 3 通道 [diff/new/old]，shape=(3, H, W)
    - 目标: heatmap=(1, Hp, Wp), bbox=(4, Hp, Wp), bbox_mask=(1, Hp, Wp)
    - Hp/Wp 与模型 `forward_dense()` 输出空间对齐: H//patch_size, W//patch_size
    """

    def __init__(
        self,
        dataset_root: str,
        annotation_file: str | None = None,
        patch_size: int = 16,
    ):
        self.dataset_root = Path(dataset_root)
        self.new_dir = self.dataset_root / "new"
        self.old_dir = self.dataset_root / "old"
        self.annotation_file = Path(annotation_file) if annotation_file else (self.dataset_root / "annotations.json")
        self.dataset_db_file = self.dataset_root / "scann_dataset.db"
        self.patch_size = max(1, int(patch_size))

        if not self.new_dir.is_dir() or not self.old_dir.is_dir():
            raise ValueError("v2 dense 数据集目录下必须包含 new 和 old 子目录")
        if (
            not self.annotation_file.is_file()
            and not (self.dataset_root / "annotations.db").is_file()
            and not self.dataset_db_file.is_file()
        ):
            raise ValueError("v2 dense 数据集缺少标注文件（annotations.json 或 annotations.db）")

        self.samples = self._load_samples()

    @staticmethod
    def _normalize_dataset_key(value: str | None) -> str:
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

    def _build_pair_lookup(self) -> Dict[str, Any]:
        from scann.data.file_manager import match_new_old_pairs

        pairs, _only_new, _only_old = match_new_old_pairs(str(self.new_dir), str(self.old_dir))
        lookup: Dict[str, Any] = {}
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

    def _resolve_pair_from_paths(self, image_info: dict[str, Any]) -> FitsImagePair | None:
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

        new_path = self.dataset_root / new_rel
        old_path = self.dataset_root / old_rel
        if not new_path.is_file() or not old_path.is_file():
            return None

        return FitsImagePair(
            name=str(image_info.get("id") or image_info.get("file_name") or new_path.stem),
            new_path=new_path,
            old_path=old_path,
        )

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(number):
            return None
        return number

    @staticmethod
    def _resolve_annotation_label(ann: dict[str, Any], image_info: dict[str, Any]) -> str | None:
        label = str(ann.get("label") or "").strip().lower()
        if label in {"real", "bogus"}:
            return label

        detail_type = str(ann.get("detail_type") or image_info.get("detail_type") or "").strip()
        if detail_type:
            try:
                from scann.core.annotation_models import DETAIL_TYPE_TO_LABEL, DetailType

                mapped = DETAIL_TYPE_TO_LABEL[DetailType(detail_type)]
                return mapped.value
            except Exception:
                detail_lower = detail_type.lower()
                if detail_lower in {"real", "bogus"}:
                    return detail_lower

        image_label = str(image_info.get("label") or "").strip().lower()
        if image_label in {"real", "bogus"}:
            return image_label

        return None

    def _parse_positive_annotation(
        self,
        ann: dict[str, Any],
        image_info: dict[str, Any],
    ) -> tuple[float, float, float, float] | None:
        label = self._resolve_annotation_label(ann, image_info)
        if label != "real":
            return None

        x = self._safe_float(ann.get("x"))
        y = self._safe_float(ann.get("y"))
        w = self._safe_float(ann.get("width"))
        h = self._safe_float(ann.get("height"))
        if x is None or y is None or w is None or h is None or w <= 0 or h <= 0:
            return None

        center_x = x + w / 2.0
        center_y = y + h / 2.0
        return center_x, center_y, w, h

    @staticmethod
    def _parse_manual_crop(image_info: dict[str, Any]) -> dict[str, float] | None:
        metadata = image_info.get("metadata")
        if not isinstance(metadata, dict):
            return None
        manual_crop = metadata.get("manual_crop")
        if not isinstance(manual_crop, dict):
            return None

        x = FitsDenseDetectionDataset._safe_float(manual_crop.get("x"))
        y = FitsDenseDetectionDataset._safe_float(manual_crop.get("y"))
        width = FitsDenseDetectionDataset._safe_float(manual_crop.get("width"))
        height = FitsDenseDetectionDataset._safe_float(manual_crop.get("height"))
        if x is None or y is None or width is None or height is None:
            return None
        if width <= 1 or height <= 1:
            return None
        return {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        }

    @staticmethod
    def _resolve_manual_crop_bounds(
        manual_crop: dict[str, float] | None,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int] | None:
        if manual_crop is None or width <= 0 or height <= 0:
            return None

        x = float(manual_crop.get("x", 0.0))
        y = float(manual_crop.get("y", 0.0))
        crop_w = float(manual_crop.get("width", 0.0))
        crop_h = float(manual_crop.get("height", 0.0))
        if crop_w <= 1 or crop_h <= 1:
            return None

        x0 = max(0, min(width - 1, int(round(x))))
        y0 = max(0, min(height - 1, int(round(y))))
        x1 = max(x0 + 1, min(width, int(round(x + crop_w))))
        y1 = max(y0 + 1, min(height, int(round(y + crop_h))))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    def _load_samples(self) -> List[Dict[str, Any]]:
        if self.annotation_file.name == "annotations.json":
            annotations_doc = load_v2_annotation_document(self.dataset_root)
        else:
            import json

            try:
                with open(self.annotation_file, "r", encoding="utf-8") as file_obj:
                    annotations_doc = json.load(file_obj)
            except json.JSONDecodeError as exc:
                raise ValueError(f"annotations.json 无法解析: {exc}") from exc

        images = annotations_doc.get("images", [])
        if not isinstance(images, list):
            raise ValueError("annotations.json 中 images 字段格式无效")

        pair_lookup = self._build_pair_lookup()
        if not pair_lookup:
            raise ValueError("v2 dense 数据集未找到可配对的 new/old 图像")

        samples: List[Dict[str, Any]] = []
        for image_info in images:
            if not isinstance(image_info, dict):
                continue

            pair = self._resolve_pair_from_paths(image_info)
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
                logger.warning("dense 数据集跳过样本：找不到匹配 new/old 图像，image=%s", image_info.get("id"))
                continue

            parsed_annotations: List[tuple[float, float, float, float]] = []
            for ann in image_info.get("annotations", []) or []:
                if not isinstance(ann, dict):
                    logger.warning("dense 数据集跳过异常标注：annotation 非字典，image=%s", image_info.get("id"))
                    continue
                parsed = self._parse_positive_annotation(ann, image_info)
                if parsed is None and (ann.get("label") is not None or ann.get("detail_type") is not None):
                    logger.warning("dense 数据集跳过异常标注：字段缺失或无效，image=%s ann=%s", image_info.get("id"), ann)
                if parsed is not None:
                    parsed_annotations.append(parsed)

            samples.append({
                "image_id": image_info.get("id") or image_info.get("file_name") or pair.name,
                "pair": pair,
                "annotations": parsed_annotations,
                "manual_crop": self._parse_manual_crop(image_info),
            })

        if not samples:
            raise ValueError("v2 dense 数据集里没有可用样本")
        return samples

    @staticmethod
    def _to_2d_float(image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"FITS 图像维度不受支持: {arr.shape}")
        return np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _normalize_channel(image: np.ndarray) -> np.ndarray:
        v_min = float(np.min(image))
        v_max = float(np.max(image))
        if not np.isfinite(v_min) or not np.isfinite(v_max) or v_max <= v_min:
            return np.zeros_like(image, dtype=np.float32)
        return np.clip((image - v_min) / (v_max - v_min), 0.0, 1.0).astype(np.float32)

    def _encode_dense_targets(
        self,
        image_height: int,
        image_width: int,
        annotations: List[tuple[float, float, float, float]],
    ) -> dict[str, np.ndarray]:
        hp = max(1, image_height // self.patch_size)
        wp = max(1, image_width // self.patch_size)
        stride_x = float(image_width) / float(wp)
        stride_y = float(image_height) / float(hp)

        heatmap = np.zeros((1, hp, wp), dtype=np.float32)
        bbox = np.zeros((4, hp, wp), dtype=np.float32)
        bbox_mask = np.zeros((1, hp, wp), dtype=np.float32)

        for center_x, center_y, box_w, box_h in annotations:
            col = int(center_x / stride_x)
            row = int(center_y / stride_y)
            col = max(0, min(wp - 1, col))
            row = max(0, min(hp - 1, row))

            offset_x = (center_x / stride_x) - (col + 0.5)
            offset_y = (center_y / stride_y) - (row + 0.5)
            offset_x = float(np.clip(offset_x, -0.499, 0.499))
            offset_y = float(np.clip(offset_y, -0.499, 0.499))

            width_scale = max(float(box_w) / stride_x, 1e-6)
            height_scale = max(float(box_h) / stride_y, 1e-6)

            bbox[0, row, col] = np.arctanh(np.clip(offset_x * 2.0, -0.999, 0.999))
            bbox[1, row, col] = np.arctanh(np.clip(offset_y * 2.0, -0.999, 0.999))
            bbox[2, row, col] = float(np.clip(np.log(width_scale), -2.0, 2.0))
            bbox[3, row, col] = float(np.clip(np.log(height_scale), -2.0, 2.0))

            heatmap[0, row, col] = 1.0
            bbox_mask[0, row, col] = 1.0

        return {
            "heatmap": heatmap,
            "bbox": bbox,
            "bbox_mask": bbox_mask,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from scann.core.fits_io import read_fits

        sample = self.samples[idx]
        pair = sample["pair"]

        new_raw = self._to_2d_float(read_fits(pair.new_path).data)
        old_raw = self._to_2d_float(read_fits(pair.old_path).data)

        height = min(new_raw.shape[0], old_raw.shape[0])
        width = min(new_raw.shape[1], old_raw.shape[1])
        if height <= 0 or width <= 0:
            raise ValueError(f"图像尺寸无效: {pair.new_path.name} / {pair.old_path.name}")

        crop_bounds = self._resolve_manual_crop_bounds(sample.get("manual_crop"), width, height)
        if crop_bounds is not None:
            x0, y0, x1, y1 = crop_bounds
            new_raw = new_raw[y0:y1, x0:x1]
            old_raw = old_raw[y0:y1, x0:x1]
            annotations = [
                (center_x - x0, center_y - y0, box_w, box_h)
                for center_x, center_y, box_w, box_h in sample["annotations"]
                if x0 <= center_x <= x1 and y0 <= center_y <= y1
            ]
        else:
            annotations = list(sample["annotations"])

        new_img = self._normalize_channel(new_raw)
        old_img = self._normalize_channel(old_raw)
        diff_img = np.abs(new_img - old_img).astype(np.float32)
        input_image = np.stack([diff_img, new_img, old_img], axis=0).astype(np.float32)

        targets = self._encode_dense_targets(new_img.shape[0], new_img.shape[1], annotations)
        targets["image_id"] = sample["image_id"]
        return input_image, targets
