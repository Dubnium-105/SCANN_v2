"""v2 FITS 全图检测标注后端。

加载 FITS 图像目录 (新图/旧图配对)，支持边界框标注 + 类别标签。
标注结果默认持久化到 SQLite（按样本增量写入），并兼容 legacy annotations.json 迁移。
"""

from __future__ import annotations

import csv
import json
import logging
import math
import shutil
from datetime import datetime
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
from scann.core.fits_annotation_storage import FitsAnnotationStorage
from scann.core.fits_io import read_fits, write_fits
from scann.core.image_aligner import align
from scann.data.file_manager import match_new_old_pairs
from scann.services.pair_service import PairService

logger = logging.getLogger(__name__)

_FITS_EXTS = {".fits", ".fit", ".fts"}


class FitsAnnotationBackend(AnnotationBackend):
    """v2 FITS 全图检测标注后端

    - 输入: FITS 图像目录 (new/old 配对) + SQLite/JSON 标注
    - 标注方式: 边界框 + 详细类别标签
    - 持久化: annotations.db (manifest: annotations.json)
    - supports_bbox: True
    """

    def __init__(self) -> None:
        super().__init__()
        self._dataset_root: Optional[Path] = None
        self._annotations_path: Optional[Path] = None
        self._annotation_storage: Optional[FitsAnnotationStorage] = None
        # 内部映射: sample_id → {new_path, old_path, new_marked_path}
        self._image_paths: dict[str, dict[str, str]] = {}
        self._pair_service = PairService()

    # ─── 抽象方法实现 ───

    def load_samples(self, path: str, filter: str = "all") -> list[AnnotationSample]:
        """扫描 FITS 目录；v2 优先对齐配对并只读取 __aligned_crop.fts。"""
        root = Path(path)
        if not root.exists():
            raise FileNotFoundError(f"数据集路径不存在: {path}")

        self._dataset_root = root
        self._samples.clear()
        self._image_paths.clear()

        self._standardize_dataset_by_date_obs(root)
        aligned_pairs = self._collect_aligned_pairs(root)
        marked_files = self._collect_marked_files(root / "new_marked")

        # 加载已有标注（优先 SQLite，兼容 legacy JSON）
        self._annotation_storage = FitsAnnotationStorage(root)
        loaded = self._annotation_storage.load_annotations()
        existing_annotations = loaded.by_id

        for sample_id, new_path, old_path in aligned_pairs:
            ref_file = new_path or old_path
            sample = AnnotationSample(
                id=sample_id,
                source_path=str(new_path or old_path or ""),
                display_name=ref_file.name if ref_file is not None else sample_id,
            )

            # 合并已有标注
            ann = self._resolve_annotation_entry(existing_annotations, sample_id)
            if ann is not None:
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
                sample.ai_suggestion = ann.get("ai_suggestion")
                sample.ai_confidence = ann.get("ai_confidence")

            self._image_paths[sample_id] = {
                "new": str(new_path) if new_path else "",
                "old": str(old_path) if old_path else "",
                "new_marked": str(marked_files.get(sample_id, "")),
            }

            self._samples.append(sample)

        self._samples.sort(key=lambda s: s.display_name)
        self._rebuild_index()

        # 设置标注文件路径
        self._annotations_path = root / "annotations.json"

        # legacy JSON 首次加载后自动迁移到 SQLite
        if loaded.loaded_from_legacy_json and self._annotation_storage is not None:
            self._annotation_storage.bulk_replace(self._samples)

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

        # 增量持久化到 SQLite
        self._save_sample_annotation(sample)

    def apply_ai_preannotations(
        self,
        sample_id: str,
        bboxes: list[BBox],
        ai_suggestion: Optional[str] = None,
        ai_confidence: Optional[float] = None,
    ) -> None:
        """写入 AI 预标注结果并持久化。"""
        sample = self.get_sample(sample_id)
        if sample is None:
            logger.warning("AI预标注失败: 样本 %s 不存在", sample_id)
            return

        old_value = {
            "bboxes": [bbox.to_dict() for bbox in sample.bboxes],
            "ai_suggestion": sample.ai_suggestion,
            "ai_confidence": sample.ai_confidence,
        }

        sample.bboxes = [BBox.from_dict(bbox.to_dict()) for bbox in bboxes]
        sample.ai_suggestion = ai_suggestion
        sample.ai_confidence = ai_confidence

        new_value = {
            "bboxes": [bbox.to_dict() for bbox in sample.bboxes],
            "ai_suggestion": sample.ai_suggestion,
            "ai_confidence": sample.ai_confidence,
        }

        self._push_undo(AnnotationAction(
            action_type="ai_prelabel",
            sample_id=sample_id,
            old_value=old_value,
            new_value=new_value,
        ))
        self._save_sample_annotation(sample)

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
            "has_new_marked_image": bool(paths.get("new_marked")),
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

    # ─── 标注持久化 ───

    def _load_annotations_json(self, root: Path) -> dict[str, dict]:
        """兼容旧接口：加载已有标注（SQLite/JSON）。"""
        storage = FitsAnnotationStorage(root)
        loaded = storage.load_annotations()
        return loaded.by_id

    def _resolve_annotation_entry(
        self,
        existing_annotations: dict[str, dict],
        sample_id: str,
    ) -> Optional[dict]:
        """兼容标准化前后的样本 ID。"""
        if sample_id in existing_annotations:
            return existing_annotations[sample_id]

        legacy_id = self._strip_datetime_prefix(sample_id)
        if legacy_id in existing_annotations:
            return existing_annotations[legacy_id]
        return None

    def _standardize_dataset_by_date_obs(self, root: Path) -> None:
        """标准化数据集命名并完成新旧图对齐准备。"""
        raw_root = root / "dataset_raw"
        folder_names = ("new", "old", "new_marked")

        for folder_name in folder_names:
            work_dir = root / folder_name
            if not work_dir.is_dir():
                continue

            raw_dir = raw_root / folder_name
            raw_dir.mkdir(parents=True, exist_ok=True)

            for file_path in sorted(work_dir.iterdir()):
                if not self._should_standardize_file(file_path):
                    continue

                backup_path = self._move_to_raw_folder(file_path, raw_dir)
                date_token = self._extract_date_obs_token(backup_path)
                normalized_name = self._build_standardized_filename(
                    src_path=backup_path,
                    date_token=date_token,
                    dst_dir=work_dir,
                )
                normalized_path = work_dir / normalized_name
                shutil.copy2(backup_path, normalized_path)

            # 标准化完成后统一执行对齐，确保对齐产物与标准化命名一致
            self._ensure_aligned_crop_files(root)

    @staticmethod
    def _should_standardize_file(file_path: Path) -> bool:
        if not file_path.is_file():
            return False
        if file_path.suffix.lower() not in _FITS_EXTS:
            return False
        if file_path.stem.lower().endswith("__aligned_crop"):
            return False
        if FitsAnnotationBackend._extract_datetime_prefix(file_path.stem) is not None:
            return False
        return True

    @staticmethod
    def _move_to_raw_folder(file_path: Path, raw_dir: Path) -> Path:
        dst_path = raw_dir / file_path.name
        if dst_path.exists():
            index = 1
            while True:
                candidate = raw_dir / f"{file_path.stem}__dup{index:02d}{file_path.suffix}"
                if not candidate.exists():
                    dst_path = candidate
                    break
                index += 1
        file_path.replace(dst_path)
        return dst_path

    @staticmethod
    def _extract_date_obs_token(path: Path) -> Optional[str]:
        date_obs = None
        try:
            fits_image = read_fits(path)
            date_obs = fits_image.header.raw.get("DATE-OBS")
        except Exception as exc:
            logger.warning("读取 DATE-OBS 失败: %s (%s)", path.name, exc)

        if isinstance(date_obs, str):
            for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
                try:
                    dt = datetime.strptime(date_obs, fmt)
                    return dt.strftime("%Y%m%dT%H%M%S")
                except ValueError:
                    continue

        return None

    @staticmethod
    def _build_standardized_filename(src_path: Path, date_token: Optional[str], dst_dir: Path) -> str:
        if not date_token:
            return src_path.name

        base = f"{date_token}__{src_path.stem}"
        candidate = dst_dir / f"{base}{src_path.suffix.lower()}"
        if not candidate.exists():
            return candidate.name

        index = 1
        while True:
            dedup_name = f"{base}__{index:02d}{src_path.suffix.lower()}"
            dedup_path = dst_dir / dedup_name
            if not dedup_path.exists():
                return dedup_name
            index += 1

    @staticmethod
    def _extract_datetime_prefix(stem: str) -> Optional[str]:
        if len(stem) < 16:
            return None
        prefix = stem[:15]
        if (
            prefix[0:8].isdigit()
            and prefix[8] == "T"
            and prefix[9:15].isdigit()
            and len(stem) > 16
            and stem[15:17] == "__"
        ):
            return prefix
        return None

    @staticmethod
    def _strip_datetime_prefix(sample_id: str) -> str:
        prefix = FitsAnnotationBackend._extract_datetime_prefix(sample_id)
        if prefix is None:
            return sample_id
        return sample_id[17:]

    def _ensure_aligned_crop_files(self, root: Path) -> None:
        """为可匹配的新旧图生成对齐裁剪产物。"""
        new_dir = root / "new"
        old_dir = root / "old"
        if not new_dir.is_dir() or not old_dir.is_dir():
            return

        pairs, _only_new, _only_old = match_new_old_pairs(str(new_dir), str(old_dir))
        for pair in pairs:
            new_aligned_path, old_aligned_path, new_marker_path, old_marker_path = (
                self._pair_service.aligned_artifact_paths(pair)
            )
            if new_aligned_path.is_file() and old_aligned_path.is_file():
                if not new_marker_path.exists() or not old_marker_path.exists():
                    marker_text = "aligned=1\n"
                    new_marker_path.write_text(marker_text, encoding="utf-8")
                    old_marker_path.write_text(marker_text, encoding="utf-8")
                self._ensure_marked_aligned_crop_file(root, pair, new_aligned_path, new_marker_path)
                continue

            self._align_pair_to_crop(
                pair,
                new_aligned_path,
                old_aligned_path,
                new_marker_path,
                old_marker_path,
            )
            self._ensure_marked_aligned_crop_file(root, pair, new_aligned_path, new_marker_path)

    def _align_pair_to_crop(
        self,
        pair,
        new_aligned_path: Path,
        old_aligned_path: Path,
        new_marker_path: Path,
        old_marker_path: Path,
    ) -> None:
        """执行单对图像对齐并写出裁剪产物。"""
        try:
            new_fits = read_fits(pair.new_path)
            old_fits = read_fits(pair.old_path)
        except Exception as exc:
            logger.warning("标注集对齐读取失败: %s (%s)", pair.name, exc)
            return

        new_data = np.nan_to_num(
            new_fits.data.astype(np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        old_data = np.nan_to_num(
            old_fits.data.astype(np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        h, w = new_data.shape[:2]
        fallback_max_shift = max(100, int(min(h, w) * 0.45))
        result = align(
            new_fits.data,
            old_fits.data,
            method="auto",
            max_shift=fallback_max_shift,
        )

        if not result.success or result.aligned_old is None:
            logger.warning("标注集对齐失败: %s (%s)", pair.name, result.error_message)
            return

        crop_bounds = self._pair_service.calc_overlap_crop_bounds(
            w=w,
            h=h,
            dx=result.dx,
            dy=result.dy,
            aligned_old=result.aligned_old,
        )
        if crop_bounds is None:
            logger.warning("标注集对齐后无有效重叠区域: %s", pair.name)
            return

        x0, x1, y0, y1 = crop_bounds
        cropped_new = new_data[y0:y1, x0:x1]
        cropped_old = result.aligned_old[y0:y1, x0:x1]

        write_fits(new_aligned_path, cropped_new, new_fits.header)
        write_fits(old_aligned_path, cropped_old, old_fits.header)
        marker_text = (
            "aligned=1\n"
            f"dx={result.dx:.6f}\n"
            f"dy={result.dy:.6f}\n"
            f"crop={x0},{x1},{y0},{y1}\n"
        )
        new_marker_path.write_text(marker_text, encoding="utf-8")
        old_marker_path.write_text(marker_text, encoding="utf-8")

    def _collect_aligned_pairs(
        self,
        root: Path,
    ) -> list[tuple[str, Optional[Path], Optional[Path]]]:
        """仅收集 __aligned_crop.fts 产物，并按原始样本名配对。"""
        new_dir = root / "new"
        old_dir = root / "old"

        if new_dir.is_dir() or old_dir.is_dir():
            new_files = self._collect_aligned_files(new_dir)
            old_files = self._collect_aligned_files(old_dir)
            if not new_files and not old_files:
                return []
            return self._pair_aligned_files(new_files, old_files)

        new_files = self._collect_aligned_files(root)
        return [(sample_id, path, None) for sample_id, path in sorted(new_files.items())]

    def _collect_aligned_files(self, folder: Path) -> dict[str, Path]:
        """扫描目录中的 __aligned_crop.fts 文件。"""
        if not folder.is_dir():
            return {}

        files: dict[str, Path] = {}
        for file_path in sorted(folder.iterdir()):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() != ".fts":
                continue
            if not file_path.stem.lower().endswith("__aligned_crop"):
                continue
            files[self._strip_aligned_crop_suffix(file_path.stem)] = file_path
        return files

    def _collect_marked_files(self, folder: Path) -> dict[str, Path]:
        """扫描带十字线标记新图目录，优先使用 __aligned_crop.fts。"""
        if not folder.is_dir():
            return {}

        aligned: dict[str, Path] = {}
        normal: dict[str, Path] = {}
        for file_path in sorted(folder.iterdir()):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in _FITS_EXTS:
                continue
            stem = file_path.stem
            if stem.lower().endswith("__aligned_crop"):
                sample_id = self._strip_aligned_crop_suffix(stem)
                aligned[sample_id] = file_path
                continue
            normal[stem] = file_path

        merged = dict(normal)
        merged.update(aligned)
        return merged

    def _ensure_marked_aligned_crop_file(
        self,
        root: Path,
        pair,
        new_aligned_path: Path,
        new_marker_path: Path,
    ) -> None:
        """为带标记新图生成 __aligned_crop.fts，确保与标注视图坐标一致。"""
        marked_dir = root / "new_marked"
        if not marked_dir.is_dir():
            return

        marked_source = marked_dir / f"{Path(pair.new_path).stem}.fits"
        if not marked_source.exists():
            for ext in (".fts", ".fit"):
                candidate = marked_dir / f"{Path(pair.new_path).stem}{ext}"
                if candidate.exists():
                    marked_source = candidate
                    break
        if not marked_source.exists():
            return

        marked_aligned = marked_source.with_name(f"{marked_source.stem}__aligned_crop.fts")
        if marked_aligned.exists():
            return

        crop_bounds = self._parse_crop_bounds_from_marker(new_marker_path)
        try:
            marked_fits = read_fits(marked_source)
            marked_data = np.nan_to_num(
                marked_fits.data.astype(np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            if crop_bounds is not None:
                x0, x1, y0, y1 = crop_bounds
                if 0 <= x0 < x1 <= marked_data.shape[1] and 0 <= y0 < y1 <= marked_data.shape[0]:
                    cropped = marked_data[y0:y1, x0:x1]
                else:
                    cropped = marked_data
            else:
                cropped = marked_data

            if new_aligned_path.exists() and (cropped.shape != read_fits(new_aligned_path).data.shape):
                aligned_shape = read_fits(new_aligned_path).data.shape
                ah, aw = aligned_shape[:2]
                h, w = cropped.shape[:2]
                y0 = max(0, (h - ah) // 2)
                x0 = max(0, (w - aw) // 2)
                cropped = cropped[y0:y0 + ah, x0:x0 + aw]

            write_fits(marked_aligned, cropped, marked_fits.header)
        except Exception as exc:
            logger.warning("带标记新图裁剪生成失败: %s (%s)", pair.name, exc)

    @staticmethod
    def _parse_crop_bounds_from_marker(marker_path: Path) -> tuple[int, int, int, int] | None:
        """从 .marker 文件解析裁剪区域。"""
        if not marker_path.exists():
            return None
        try:
            for line in marker_path.read_text(encoding="utf-8").splitlines():
                if not line.startswith("crop="):
                    continue
                values = line.split("=", 1)[1]
                x0, x1, y0, y1 = [int(v.strip()) for v in values.split(",")]
                return x0, x1, y0, y1
        except Exception:
            return None
        return None

    def _pair_aligned_files(
        self,
        new_files: dict[str, Path],
        old_files: dict[str, Path],
    ) -> list[tuple[str, Optional[Path], Optional[Path]]]:
        """对齐产物配对，支持 FW_ 前缀兼容。"""
        if not old_files:
            return [(sample_id, path, None) for sample_id, path in sorted(new_files.items())]

        old_norm_map = {
            self._normalize_pair_stem(stem): stem
            for stem in old_files
        }

        pairs: list[tuple[str, Optional[Path], Optional[Path]]] = []
        matched_old: set[str] = set()
        for sample_id, new_path in sorted(new_files.items()):
            old_stem = None
            if sample_id in old_files:
                old_stem = sample_id
            else:
                norm = self._normalize_pair_stem(sample_id)
                candidate = old_norm_map.get(norm)
                if candidate is not None and candidate not in matched_old:
                    old_stem = candidate

            if old_stem is None:
                continue

            matched_old.add(old_stem)
            pairs.append((sample_id, new_path, old_files[old_stem]))

        return pairs

    @staticmethod
    def _strip_aligned_crop_suffix(stem: str) -> str:
        suffix = "__aligned_crop"
        if stem.lower().endswith(suffix):
            return stem[:-len(suffix)]
        return stem

    @staticmethod
    def _normalize_pair_stem(stem: str) -> str:
        stem = FitsAnnotationBackend._strip_datetime_prefix(stem)
        for prefix in ("FW_", "fw_", "Fw_"):
            if stem.startswith(prefix):
                return stem[len(prefix):]
        return stem

    def _save_sample_annotation(self, sample: AnnotationSample) -> None:
        """按样本增量持久化，避免全量重写大 JSON。"""
        if self._annotation_storage is None:
            return
        self._annotation_storage.upsert_sample(sample)

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
