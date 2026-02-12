"""v1 三联图分类标注后端

将 80×240 PNG 三联图按 real/bogus 分类，通过文件夹归类持久化标注结果。
支持子类型细分标注 (asteroid/supernova/... 等)。
"""

from __future__ import annotations

import csv
import logging
import math
import os
import shutil
from pathlib import Path
from typing import Optional

from PIL import Image

from scann.core.annotation_backend import AnnotationBackend, ImageData
from scann.core.annotation_models import (
    AnnotationAction,
    AnnotationSample,
    AnnotationStats,
    BBox,
    DETAIL_TYPE_TO_LABEL,
    DetailType,
    ExportResult,
)

logger = logging.getLogger(__name__)

# 支持的图像后缀
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class TripletAnnotationBackend(AnnotationBackend):
    """v1 三联图分类标注后端

    - 输入: PNG 三联图文件夹 (positive/negative/unlabeled)
    - 标注方式: 二分类 + 细分类型
    - 持久化: 文件移动到 positive/ 或 negative/ 目录
    - supports_bbox: False (仅分类)
    """

    def __init__(self) -> None:
        super().__init__()
        self._dataset_root: Optional[Path] = None
        # 内部元数据: sample_id → {detail_type, original_folder}
        self._meta: dict[str, dict] = {}

    # ─── 抽象方法实现 ───

    def load_samples(self, path: str, filter: str = "all") -> list[AnnotationSample]:
        """扫描数据集目录，加载三联图样本"""
        root = Path(path)
        if not root.exists():
            raise FileNotFoundError(f"数据集路径不存在: {path}")

        self._dataset_root = root
        self._samples.clear()
        self._meta.clear()

        # 确保目录结构存在
        positive_dir = root / "positive"
        negative_dir = root / "negative"
        unlabeled_dir = root / "unlabeled"

        # 扫描各子目录中的图像
        found: dict[str, AnnotationSample] = {}

        for subdir, label in [
            (positive_dir, "real"),
            (negative_dir, "bogus"),
            (unlabeled_dir, None),
        ]:
            if not subdir.is_dir():
                continue
            for f in sorted(subdir.iterdir()):
                if f.suffix.lower() not in _IMAGE_EXTS:
                    continue
                sample_id = f.stem
                # 去重: 如果同名文件已在 positive/negative 中, 跳过 unlabeled
                if sample_id in found:
                    continue
                sample = AnnotationSample(
                    id=sample_id,
                    source_path=str(f),
                    display_name=f.name,
                    label=label,
                )
                found[sample_id] = sample
                self._meta[sample_id] = {"original_folder": str(subdir)}

        # 也扫描根目录下的散落图像 (无子目录时)
        if not found:
            for f in sorted(root.iterdir()):
                if f.is_file() and f.suffix.lower() in _IMAGE_EXTS:
                    sample_id = f.stem
                    sample = AnnotationSample(
                        id=sample_id,
                        source_path=str(f),
                        display_name=f.name,
                    )
                    found[sample_id] = sample
                    self._meta[sample_id] = {"original_folder": str(root)}

        self._samples = sorted(found.values(), key=lambda s: s.display_name)
        self._rebuild_index()

        # 应用筛选
        if filter != "all":
            return self.get_filtered_samples(filter)

        return list(self._samples)

    def save_annotation(
        self,
        sample_id: str,
        label: str,
        bbox: Optional[BBox] = None,
        confidence: float = 1.0,
        detail_type: Optional[str] = None,
    ) -> None:
        """标注一个样本 — 修改标签并将文件移动到对应目录"""
        sample = self.get_sample(sample_id)
        if sample is None:
            logger.warning(f"标注失败: 样本 {sample_id} 不存在")
            return

        # 记录撤销信息
        old_value = {
            "label": sample.label,
            "detail_type": sample.detail_type,
            "source_path": sample.source_path,
        }

        # 更新样本标签
        sample.label = label
        sample.detail_type = detail_type

        # 移动文件到对应目录
        self._move_file(sample, label)

        new_value = {
            "label": sample.label,
            "detail_type": sample.detail_type,
            "source_path": sample.source_path,
        }

        self._push_undo(AnnotationAction(
            action_type="label",
            sample_id=sample_id,
            old_value=old_value,
            new_value=new_value,
        ))

    def get_image_data(self, sample: AnnotationSample) -> ImageData:
        """加载三联图 PNG 为 PIL Image"""
        return Image.open(sample.source_path)

    def get_display_info(self, sample: AnnotationSample) -> dict:
        return {
            "file_name": sample.display_name,
            "label": sample.label,
            "detail_type": sample.detail_type,
            "label_display": sample.label_display,
            "has_new_image": False,  # v1 不区分新旧图
            "has_old_image": False,
        }

    def export_dataset(
        self,
        output_dir: str,
        format: str = "native",
        include_unlabeled: bool = False,
        val_split: float = 0.0,
    ) -> ExportResult:
        """导出标注数据集"""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 筛选待导出样本
        if include_unlabeled:
            samples = list(self._samples)
        else:
            samples = [s for s in self._samples if s.is_labeled]

        if not samples:
            return ExportResult(
                success=True,
                output_dir=str(out),
                total_exported=0,
                format=format,
            )

        # 训练/验证拆分
        train_samples, val_samples = self._split_train_val(samples, val_split)

        if format == "native":
            return self._export_native(out, train_samples, val_samples, format)
        elif format == "csv":
            return self._export_csv(out, train_samples, val_samples, format)
        else:
            return ExportResult(
                success=False,
                output_dir=str(out),
                error_message=f"不支持的导出格式: {format}",
            )

    def get_statistics(self) -> AnnotationStats:
        stats = AnnotationStats()
        stats.update_from_samples(self._samples)
        return stats

    def supports_bbox(self) -> bool:
        return False

    # ─── 撤销/重做覆盖 ───

    def _apply_undo(self, action: AnnotationAction) -> None:
        sample = self.get_sample(action.sample_id)
        if sample is None:
            return
        if action.old_value is not None:
            # 移动文件回原位
            old_path = action.old_value.get("source_path")
            if old_path and old_path != sample.source_path:
                self._move_file_to(sample, old_path)
            sample.label = action.old_value.get("label")
            sample.detail_type = action.old_value.get("detail_type")

    def _apply_redo(self, action: AnnotationAction) -> None:
        sample = self.get_sample(action.sample_id)
        if sample is None:
            return
        if action.new_value is not None:
            sample.label = action.new_value.get("label")
            sample.detail_type = action.new_value.get("detail_type")
            new_path = action.new_value.get("source_path")
            if new_path and new_path != sample.source_path:
                self._move_file_to(sample, new_path)

    # ─── 私有方法 ───

    def _move_file(self, sample: AnnotationSample, label: str) -> None:
        """将文件移动到 label 对应的目录"""
        if self._dataset_root is None:
            return

        target_dir_name = "positive" if label == "real" else "negative"
        target_dir = self._dataset_root / target_dir_name
        target_dir.mkdir(parents=True, exist_ok=True)

        src = Path(sample.source_path)
        dst = target_dir / src.name

        if src == dst:
            return
        if src.exists():
            shutil.move(str(src), str(dst))
            sample.source_path = str(dst)

    def _move_file_to(self, sample: AnnotationSample, target_path: str) -> None:
        """将文件移动到指定路径"""
        src = Path(sample.source_path)
        dst = Path(target_path)
        if src == dst:
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.move(str(src), str(dst))
            sample.source_path = str(dst)

    def _split_train_val(
        self, samples: list[AnnotationSample], val_split: float
    ) -> tuple[list[AnnotationSample], list[AnnotationSample]]:
        """拆分训练/验证集"""
        if val_split <= 0 or val_split >= 1.0:
            return samples, []
        val_count = max(1, math.floor(len(samples) * val_split))
        return samples[:-val_count], samples[-val_count:]

    def _export_native(
        self,
        out: Path,
        train: list[AnnotationSample],
        val: list[AnnotationSample],
        format: str,
    ) -> ExportResult:
        """原生格式导出: 文件夹分类"""
        all_samples = train + val
        pos_dir = out / "positive"
        neg_dir = out / "negative"
        pos_dir.mkdir(exist_ok=True)
        neg_dir.mkdir(exist_ok=True)

        exported = 0
        for s in all_samples:
            src = Path(s.source_path)
            if not src.exists():
                continue
            dst_dir = pos_dir if s.label == "real" else neg_dir
            shutil.copy2(str(src), str(dst_dir / src.name))
            exported += 1

        return ExportResult(
            success=True,
            output_dir=str(out),
            total_exported=exported,
            train_count=len(train),
            val_count=len(val),
            format=format,
        )

    def _export_csv(
        self,
        out: Path,
        train: list[AnnotationSample],
        val: list[AnnotationSample],
        format: str,
    ) -> ExportResult:
        """CSV 格式导出: 文件路径 + 标签"""
        all_samples = train + val
        csv_path = out / "annotations.csv"

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["file_path", "label", "detail_type", "split"]
            )
            writer.writeheader()
            for s in train:
                writer.writerow({
                    "file_path": s.source_path,
                    "label": s.label or "",
                    "detail_type": s.detail_type or "",
                    "split": "train",
                })
            for s in val:
                writer.writerow({
                    "file_path": s.source_path,
                    "label": s.label or "",
                    "detail_type": s.detail_type or "",
                    "split": "val",
                })

        return ExportResult(
            success=True,
            output_dir=str(out),
            total_exported=len(all_samples),
            train_count=len(train),
            val_count=len(val),
            format=format,
        )
