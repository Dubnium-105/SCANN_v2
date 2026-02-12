"""v2 FITS 全图检测标注后端

加载 FITS 图像目录 (新图/旧图配对)，支持边界框标注 + 类别标签，
标注结果持久化为 JSON 文件 (兼容 FitsDetectionDataset 格式)。
"""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np

from scann.core.annotation_backend import AnnotationBackend, ImageData
from scann.core.annotation_models import (
    AnnotationAction,
    AnnotationSample,
    AnnotationStats,
    BBox,
    ExportResult,
)

logger = logging.getLogger(__name__)

_FITS_EXTS = {".fits", ".fit", ".fts"}


class FitsAnnotationBackend(AnnotationBackend):
    """v2 FITS 全图检测标注后端

    - 输入: FITS 图像目录 (new/old 配对) + JSON 标注文件
    - 标注方式: 边界框 + 详细类别标签
    - 持久化: annotations.json
    - supports_bbox: True
    """

    def __init__(self) -> None:
        super().__init__()
        self._dataset_root: Optional[Path] = None
        self._annotations_path: Optional[Path] = None
        # 内部映射: sample_id → {new_path, old_path}
        self._image_paths: dict[str, dict[str, str]] = {}

    # ─── 抽象方法实现 ───

    def load_samples(self, path: str, filter: str = "all") -> list[AnnotationSample]:
        """扫描 FITS 目录并自动配对 new/old 文件"""
        root = Path(path)
        if not root.exists():
            raise FileNotFoundError(f"数据集路径不存在: {path}")

        self._dataset_root = root
        self._samples.clear()
        self._image_paths.clear()

        new_dir = root / "new"
        old_dir = root / "old"

        # 收集 FITS 文件
        new_files: dict[str, Path] = {}
        old_files: dict[str, Path] = {}

        if new_dir.is_dir():
            for f in sorted(new_dir.iterdir()):
                if f.suffix.lower() in _FITS_EXTS:
                    new_files[f.stem] = f

        if old_dir.is_dir():
            for f in sorted(old_dir.iterdir()):
                if f.suffix.lower() in _FITS_EXTS:
                    old_files[f.stem] = f

        # 也扫描根目录下的 FITS 文件 (无 new/old 子目录时)
        if not new_files and not old_files:
            for f in sorted(root.iterdir()):
                if f.is_file() and f.suffix.lower() in _FITS_EXTS:
                    new_files[f.stem] = f

        # ─── 智能配对: 处理 FW_ 等常见前缀差异 ───
        # 构建旧图名称→路径映射 (去掉 FW_ 前缀后匹配)
        _STRIP_PREFIXES = ("FW_", "fw_", "Fw_")

        def _normalize_stem(stem: str) -> str:
            """去除常见前缀用于匹配"""
            for prefix in _STRIP_PREFIXES:
                if stem.startswith(prefix):
                    return stem[len(prefix):]
            return stem

        # 为旧图建立 normalized_stem → original_stem 映射
        old_norm_map: dict[str, str] = {}
        for stem in old_files:
            norm = _normalize_stem(stem)
            old_norm_map[norm] = stem

        # 尝试将 new 文件与 old 文件匹配（先精确匹配，再去前缀匹配）
        matched_old_stems: set[str] = set()
        new_to_old: dict[str, str] = {}  # new_stem → old_stem
        for stem in new_files:
            if stem in old_files:
                new_to_old[stem] = stem
                matched_old_stems.add(stem)
            else:
                # 尝试 normalize 后匹配
                norm = _normalize_stem(stem)
                if norm in old_norm_map:
                    old_stem = old_norm_map[norm]
                    if old_stem not in matched_old_stems:
                        new_to_old[stem] = old_stem
                        matched_old_stems.add(old_stem)

        # 配对: 以 new 为主 + 未匹配的 old
        unmatched_old = set(old_files.keys()) - matched_old_stems
        all_stems = sorted(new_files.keys())
        # 追加仅存在于 old 中的未匹配项
        all_stems.extend(sorted(unmatched_old))

        # 加载已有 JSON 标注
        existing_annotations = self._load_annotations_json(root)

        for stem in all_stems:
            new_path = new_files.get(stem)
            # 使用智能配对映射查找对应旧图
            old_stem = new_to_old.get(stem, stem)
            old_path = old_files.get(old_stem) if new_path else old_files.get(stem)

            # 使用 new 或 old 中存在的扩展名
            ref_file = new_path or old_path
            if ref_file is None:
                continue

            sample_id = stem
            sample = AnnotationSample(
                id=sample_id,
                source_path=str(new_path or old_path),
                display_name=ref_file.name,
            )

            # 合并已有标注
            if sample_id in existing_annotations:
                ann = existing_annotations[sample_id]
                if ann.get("annotations"):
                    sample.bboxes = [BBox.from_dict(b) for b in ann["annotations"]]
                    # 如果有 bbox 则样本已标注
                    if sample.bboxes:
                        # 用第一个 bbox 的标签作为样本标签
                        sample.label = sample.bboxes[0].label
                        sample.detail_type = sample.bboxes[0].detail_type
                if ann.get("label"):
                    sample.label = ann["label"]
                if ann.get("detail_type"):
                    sample.detail_type = ann["detail_type"]

            self._image_paths[sample_id] = {
                "new": str(new_path) if new_path else "",
                "old": str(old_path) if old_path else "",
            }

            self._samples.append(sample)

        self._samples.sort(key=lambda s: s.display_name)
        self._rebuild_index()

        # 设置标注文件路径
        self._annotations_path = root / "annotations.json"

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
        """保存标注 — 更新样本并写入 JSON"""
        sample = self.get_sample(sample_id)
        if sample is None:
            logger.warning(f"标注失败: 样本 {sample_id} 不存在")
            return

        # 记录撤销信息
        old_value = {
            "label": sample.label,
            "detail_type": sample.detail_type,
            "bboxes": [b.to_dict() for b in sample.bboxes],
        }

        # 更新标签
        sample.label = label
        if detail_type:
            sample.detail_type = detail_type

        # 添加边界框
        if bbox is not None:
            bbox.label = label
            bbox.confidence = confidence
            if detail_type:
                bbox.detail_type = detail_type
            sample.bboxes.append(bbox)

        new_value = {
            "label": sample.label,
            "detail_type": sample.detail_type,
            "bboxes": [b.to_dict() for b in sample.bboxes],
        }

        self._push_undo(AnnotationAction(
            action_type="bbox_add" if bbox else "label",
            sample_id=sample_id,
            old_value=old_value,
            new_value=new_value,
        ))

        # 持久化到 JSON
        self._save_annotations_json()

    def get_image_data(
        self, sample: AnnotationSample, image_type: str = "new"
    ) -> ImageData:
        """加载 FITS 图像数据

        Args:
            sample: 标注样本
            image_type: "new" 或 "old"

        Returns:
            numpy 数组
        """
        try:
            from astropy.io import fits
        except ImportError:
            raise ImportError("需要 astropy 库来读取 FITS 文件")

        paths = self._image_paths.get(sample.id, {})
        path = paths.get(image_type, "") or paths.get("new", "")

        if not path or not Path(path).exists():
            # 回退到 source_path
            path = sample.source_path

        with fits.open(path) as hdul:
            data = hdul[0].data
            if data is None:
                raise ValueError(f"FITS 文件无图像数据: {path}")
            return data.copy()

    def get_display_info(self, sample: AnnotationSample) -> dict:
        paths = self._image_paths.get(sample.id, {})
        return {
            "file_name": sample.display_name,
            "label": sample.label,
            "detail_type": sample.detail_type,
            "label_display": sample.label_display,
            "has_new_image": bool(paths.get("new")),
            "has_old_image": bool(paths.get("old")),
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

        if include_unlabeled:
            samples = list(self._samples)
        else:
            samples = [s for s in self._samples if s.is_labeled or s.bboxes]

        if not samples:
            return ExportResult(
                success=True, output_dir=str(out),
                total_exported=0, format=format,
            )

        train, val = self._split_train_val(samples, val_split)

        if format == "native":
            return self._export_native_json(out, train, val, format)
        elif format == "csv":
            return self._export_csv(out, train, val, format)
        else:
            return ExportResult(
                success=False, output_dir=str(out),
                error_message=f"不支持的导出格式: {format}",
            )

    def get_statistics(self) -> AnnotationStats:
        stats = AnnotationStats()
        # 有 bbox 或有 label 的都算已标注
        effective_samples = []
        for s in self._samples:
            if s.bboxes:
                # 为有 bbox 的样本计算统计
                for b in s.bboxes:
                    effective_samples.append(AnnotationSample(
                        id=f"{s.id}_bbox",
                        source_path=s.source_path,
                        display_name=s.display_name,
                        label=b.label,
                        detail_type=b.detail_type,
                    ))
            effective_samples.append(s)

        # 使用简单统计
        stats.total = len(self._samples)
        stats.labeled = sum(
            1 for s in self._samples if s.is_labeled or s.bboxes
        )
        stats.unlabeled = stats.total - stats.labeled
        stats.progress_percent = (
            (stats.labeled / stats.total * 100.0) if stats.total > 0 else 0.0
        )

        # 按 detail_type 统计
        stats.label_counts.clear()
        for s in self._samples:
            if s.bboxes:
                for b in s.bboxes:
                    key = b.detail_type or b.label
                    stats.label_counts[key] = stats.label_counts.get(key, 0) + 1
            elif s.detail_type:
                stats.label_counts[s.detail_type] = (
                    stats.label_counts.get(s.detail_type, 0) + 1
                )
            elif s.label:
                stats.label_counts[s.label] = (
                    stats.label_counts.get(s.label, 0) + 1
                )

        return stats

    def supports_bbox(self) -> bool:
        return True

    # ─── JSON 持久化 ───

    def _load_annotations_json(self, root: Path) -> dict[str, dict]:
        """加载已有的 annotations.json"""
        ann_path = root / "annotations.json"
        if not ann_path.exists():
            return {}

        try:
            data = json.loads(ann_path.read_text(encoding="utf-8"))
            result = {}
            for img in data.get("images", []):
                result[img["id"]] = img
            return result
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"无法解析标注文件: {e}")
            return {}

    def _save_annotations_json(self) -> None:
        """将所有标注保存到 annotations.json"""
        if self._annotations_path is None:
            return

        images = []
        for s in self._samples:
            if not s.is_labeled and not s.bboxes:
                continue
            img_data: dict = {
                "id": s.id,
                "file_name": s.display_name,
            }
            if s.label:
                img_data["label"] = s.label
            if s.detail_type:
                img_data["detail_type"] = s.detail_type
            if s.bboxes:
                img_data["annotations"] = [b.to_dict() for b in s.bboxes]
            images.append(img_data)

        doc = {
            "version": "2.0",
            "images": images,
        }

        self._annotations_path.write_text(
            json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ─── 工具方法 ───

    def _split_train_val(
        self, samples: list[AnnotationSample], val_split: float
    ) -> tuple[list[AnnotationSample], list[AnnotationSample]]:
        if val_split <= 0 or val_split >= 1.0:
            return samples, []
        val_count = max(1, math.floor(len(samples) * val_split))
        return samples[:-val_count], samples[-val_count:]

    def _export_native_json(
        self, out: Path,
        train: list[AnnotationSample],
        val: list[AnnotationSample],
        format: str,
    ) -> ExportResult:
        """导出原生 JSON 格式"""
        all_samples = train + val
        images = []
        for s in all_samples:
            img: dict = {
                "id": s.id,
                "file_name": s.display_name,
            }
            if s.label:
                img["label"] = s.label
            if s.detail_type:
                img["detail_type"] = s.detail_type
            if s.bboxes:
                img["annotations"] = [b.to_dict() for b in s.bboxes]
            images.append(img)

        doc = {"version": "2.0", "images": images}
        ann_path = out / "annotations.json"
        ann_path.write_text(
            json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        return ExportResult(
            success=True,
            output_dir=str(out),
            total_exported=len(all_samples),
            train_count=len(train),
            val_count=len(val),
            format=format,
        )

    def _export_csv(
        self, out: Path,
        train: list[AnnotationSample],
        val: list[AnnotationSample],
        format: str,
    ) -> ExportResult:
        """CSV 格式导出"""
        all_samples = train + val
        csv_path = out / "annotations.csv"

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "file_path", "label", "detail_type",
                    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
                    "confidence", "split",
                ],
            )
            writer.writeheader()

            for split_name, split_samples in [("train", train), ("val", val)]:
                for s in split_samples:
                    if s.bboxes:
                        for b in s.bboxes:
                            writer.writerow({
                                "file_path": s.source_path,
                                "label": b.label,
                                "detail_type": b.detail_type or "",
                                "bbox_x": b.x,
                                "bbox_y": b.y,
                                "bbox_w": b.width,
                                "bbox_h": b.height,
                                "confidence": b.confidence,
                                "split": split_name,
                            })
                    else:
                        writer.writerow({
                            "file_path": s.source_path,
                            "label": s.label or "",
                            "detail_type": s.detail_type or "",
                            "bbox_x": "",
                            "bbox_y": "",
                            "bbox_w": "",
                            "bbox_h": "",
                            "confidence": "",
                            "split": split_name,
                        })

        return ExportResult(
            success=True,
            output_dir=str(out),
            total_exported=len(all_samples),
            train_count=len(train),
            val_count=len(val),
            format=format,
        )
