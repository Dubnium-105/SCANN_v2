from __future__ import annotations

import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from scann.core.brightness_match import brightness_match_anchors
from scann.core.fits_io import read_fits, write_fits
from scann.core.image_aligner import align
from scann.data.file_manager import FitsImagePair, match_new_old_pairs
from scann.services.pair_service import PairService

logger = logging.getLogger(__name__)

_FITS_EXTS = {".fits", ".fit", ".fts"}
_BRIGHTNESS_MATCH_DONE_MARKER = ".scann_brightness_match.done"
_MATCH_MAX_SAMPLES = 200000
_MATCH_HIGH_PERCENTILE = 99.9
_MATCH_HIGHLIGHT_SIGMA = 5.0
_MATCH_ADAPTIVE_HIGH_PERCENTILE = False


@dataclass(frozen=True)
class PreparedTaskPaths:
    task_id: str
    new_path: Path
    old_path: Optional[Path] = None
    new_marked_path: Optional[Path] = None


@dataclass(frozen=True)
class DatasetPreprocessReport:
    standardized_files: int = 0
    brightness_matched_files: int = 0
    reused_aligned_pairs: int = 0
    generated_aligned_pairs: int = 0
    generated_marked_crops: int = 0
    task_count: int = 0


class DatasetPreprocessService:
    def __init__(
        self,
        *,
        pair_service: Optional[PairService] = None,
        align_fn: Callable[..., object] = align,
        read_fits_fn: Callable[[str | Path], object] = read_fits,
        write_fits_fn: Callable[[str | Path, np.ndarray, object], None] = write_fits,
        max_workers: Optional[int] = None,
    ) -> None:
        self._pair_service = pair_service or PairService()
        self._align = align_fn
        self._read_fits = read_fits_fn
        self._write_fits = write_fits_fn
        self._max_workers = max_workers

    def prepare_dataset(self, root: Path) -> DatasetPreprocessReport:
        dataset_root = Path(root)
        standardized_files = self.standardize_dataset_by_date_obs(dataset_root)
        brightness_matched_files = self.apply_initial_brightness_match(dataset_root)
        reused_aligned_pairs, generated_aligned_pairs, generated_marked_crops = (
            self.ensure_aligned_crop_files(dataset_root)
        )
        task_count = len(self.collect_preprocessed_tasks(dataset_root))
        return DatasetPreprocessReport(
            standardized_files=standardized_files,
            brightness_matched_files=brightness_matched_files,
            reused_aligned_pairs=reused_aligned_pairs,
            generated_aligned_pairs=generated_aligned_pairs,
            generated_marked_crops=generated_marked_crops,
            task_count=task_count,
        )

    def standardize_dataset_by_date_obs(self, root: Path) -> int:
        raw_root = root / "dataset_raw"
        folder_names = ("new", "old", "new_marked")
        jobs: list[tuple[Path, Path, Path]] = []

        for folder_name in folder_names:
            work_dir = root / folder_name
            if not work_dir.is_dir():
                continue

            raw_dir = raw_root / folder_name
            raw_dir.mkdir(parents=True, exist_ok=True)

            for file_path in sorted(work_dir.iterdir()):
                if not self._should_standardize_file(file_path):
                    continue
                if (raw_dir / file_path.name).exists():
                    continue
                jobs.append((file_path, raw_dir, work_dir))

        return sum(self._run_jobs(jobs, self._standardize_single_file))

    def ensure_aligned_crop_files(self, root: Path) -> tuple[int, int, int]:
        new_dir = root / "new"
        old_dir = root / "old"
        if not new_dir.is_dir() or not old_dir.is_dir():
            return 0, 0, 0

        pairs, _only_new, _only_old = match_new_old_pairs(str(new_dir), str(old_dir))
        results = self._run_jobs(
            pairs,
            lambda pair: self._ensure_aligned_pair(root, pair),
        )
        reused_aligned_pairs = sum(item[0] for item in results)
        generated_aligned_pairs = sum(item[1] for item in results)
        generated_marked_crops = sum(item[2] for item in results)
        return reused_aligned_pairs, generated_aligned_pairs, generated_marked_crops

    def apply_initial_brightness_match(self, root: Path) -> int:
        marker_path = root / _BRIGHTNESS_MATCH_DONE_MARKER
        if marker_path.exists():
            return 0

        new_dir = root / "new"
        old_dir = root / "old"
        if not new_dir.is_dir() or not old_dir.is_dir():
            return 0

        pairs, _only_new, _only_old = match_new_old_pairs(str(new_dir), str(old_dir))
        if not pairs:
            return 0

        matched_files = sum(self._run_jobs(pairs, self._apply_brightness_match_to_pair))
        marker_path.write_text(
            f"matched_files={matched_files}\n",
            encoding="utf-8",
        )
        return matched_files

    def collect_preprocessed_tasks(self, root: Path) -> list[PreparedTaskPaths]:
        marked_files = self.collect_marked_files(root / "new_marked")
        tasks: list[PreparedTaskPaths] = []
        for task_id, new_path, old_path in self.collect_aligned_pairs(root):
            tasks.append(
                PreparedTaskPaths(
                    task_id=task_id,
                    new_path=new_path,
                    old_path=old_path,
                    new_marked_path=marked_files.get(task_id),
                )
            )
        return tasks

    def collect_aligned_pairs(
        self,
        root: Path,
    ) -> list[tuple[str, Optional[Path], Optional[Path]]]:
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

    def collect_marked_files(self, folder: Path) -> dict[str, Path]:
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
                sample_id = self.strip_aligned_crop_suffix(stem)
                aligned[sample_id] = file_path
                continue
            normal[stem] = file_path

        merged = dict(normal)
        merged.update(aligned)
        return merged

    @staticmethod
    def extract_datetime_prefix(stem: str) -> Optional[str]:
        if len(stem) < 17:
            return None
        prefix = stem[:15]
        if (
            prefix[0:8].isdigit()
            and prefix[8].lower() == "t"
            and prefix[9:15].isdigit()
            and stem[15:17] == "__"
        ):
            return prefix
        return None

    @staticmethod
    def strip_datetime_prefix(sample_id: str) -> str:
        prefix = DatasetPreprocessService.extract_datetime_prefix(sample_id)
        if prefix is None:
            return sample_id
        return sample_id[17:]

    @staticmethod
    def strip_aligned_crop_suffix(stem: str) -> str:
        suffix = "__aligned_crop"
        if stem.lower().endswith(suffix):
            return stem[:-len(suffix)]
        return stem

    @staticmethod
    def normalize_pair_stem(stem: str) -> str:
        stem = DatasetPreprocessService.strip_datetime_prefix(stem)
        for prefix in ("FW_", "fw_", "Fw_"):
            if stem.startswith(prefix):
                return stem[len(prefix):]
        return stem

    @staticmethod
    def parse_crop_bounds_from_marker(marker_path: Path) -> tuple[int, int, int, int] | None:
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

    @staticmethod
    def _should_standardize_file(file_path: Path) -> bool:
        if not file_path.is_file():
            return False
        if file_path.suffix.lower() not in _FITS_EXTS:
            return False
        if file_path.stem.lower().endswith("__aligned_crop"):
            return False
        if DatasetPreprocessService.extract_datetime_prefix(file_path.stem) is not None:
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

    def _extract_date_obs_token(self, path: Path) -> Optional[str]:
        date_obs = None
        try:
            fits_image = self._read_fits(path)
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

    def _standardize_single_file(self, job: tuple[Path, Path, Path]) -> int:
        file_path, raw_dir, work_dir = job
        if not file_path.exists():
            return 0
        if (raw_dir / file_path.name).exists():
            return 0

        backup_path = self._move_to_raw_folder(file_path, raw_dir)
        date_token = self._extract_date_obs_token(backup_path)
        normalized_name = self._build_standardized_filename(
            src_path=backup_path,
            date_token=date_token,
            dst_dir=work_dir,
        )
        normalized_path = work_dir / normalized_name
        shutil.copy2(backup_path, normalized_path)
        return 1

    def _apply_brightness_match_to_pair(self, pair: FitsImagePair) -> int:
        try:
            new_fits = self._read_fits(pair.new_path)
            old_fits = self._read_fits(pair.old_path)
        except Exception as exc:
            logger.warning("亮度匹配读取失败: %s (%s)", pair.name, exc)
            return 0

        try:
            new_data = np.asarray(new_fits.data, dtype=np.float32)
            old_data = np.asarray(old_fits.data, dtype=np.float32)
            matched_old = self._match_data_to_reference(new_data, old_data)
        except Exception as exc:
            logger.warning("亮度匹配失败: %s (%s)", pair.name, exc)
            return 0

        if not np.allclose(old_data, matched_old, rtol=1e-5, atol=1e-3, equal_nan=True):
            self._write_fits(pair.old_path, matched_old, old_fits.header)
            return 1
        return 0

    def _match_data_to_reference(self, reference_data: np.ndarray, source_data: np.ndarray) -> np.ndarray:
        ref_background, ref_highlight, *_ = brightness_match_anchors(
            reference_data,
            max_samples=_MATCH_MAX_SAMPLES,
            high_percentile=_MATCH_HIGH_PERCENTILE,
            highlight_sigma=_MATCH_HIGHLIGHT_SIGMA,
            adaptive_high_percentile=_MATCH_ADAPTIVE_HIGH_PERCENTILE,
        )
        src_background, src_highlight, *_ = brightness_match_anchors(
            source_data,
            max_samples=_MATCH_MAX_SAMPLES,
            high_percentile=_MATCH_HIGH_PERCENTILE,
            highlight_sigma=_MATCH_HIGHLIGHT_SIGMA,
            adaptive_high_percentile=_MATCH_ADAPTIVE_HIGH_PERCENTILE,
        )

        source_span = float(src_highlight - src_background)
        reference_span = float(ref_highlight - ref_background)
        if not np.isfinite(source_span) or source_span <= 0:
            raise ValueError("source anchors are degenerate")
        if not np.isfinite(reference_span) or reference_span <= 0:
            raise ValueError("reference anchors are degenerate")

        scale = reference_span / source_span
        offset = ref_background - (scale * src_background)
        matched = np.asarray(source_data, dtype=np.float32) * np.float32(scale) + np.float32(offset)
        return matched.astype(np.float32, copy=False)

    def _ensure_aligned_pair(
        self,
        root: Path,
        pair: FitsImagePair,
    ) -> tuple[int, int, int]:
        new_aligned_path, old_aligned_path, new_marker_path, old_marker_path = (
            self._pair_service.aligned_artifact_paths(pair)
        )
        if new_aligned_path.is_file() and old_aligned_path.is_file():
            if not new_marker_path.exists() or not old_marker_path.exists():
                marker_text = "aligned=1\n"
                new_marker_path.write_text(marker_text, encoding="utf-8")
                old_marker_path.write_text(marker_text, encoding="utf-8")
            generated_marked_crop = int(
                self._ensure_marked_aligned_crop_file(root, pair, new_aligned_path, new_marker_path)
            )
            return 1, 0, generated_marked_crop

        generated_aligned_pair = int(
            self._align_pair_to_crop(
                pair,
                new_aligned_path,
                old_aligned_path,
                new_marker_path,
                old_marker_path,
            )
        )
        if not generated_aligned_pair:
            return 0, 0, 0

        generated_marked_crop = int(
            self._ensure_marked_aligned_crop_file(root, pair, new_aligned_path, new_marker_path)
        )
        return 0, 1, generated_marked_crop

    def _align_pair_to_crop(
        self,
        pair: FitsImagePair,
        new_aligned_path: Path,
        old_aligned_path: Path,
        new_marker_path: Path,
        old_marker_path: Path,
    ) -> bool:
        try:
            new_fits = self._read_fits(pair.new_path)
            old_fits = self._read_fits(pair.old_path)
        except Exception as exc:
            logger.warning("标注集对齐读取失败: %s (%s)", pair.name, exc)
            return False

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
        result = self._align(
            new_fits.data,
            old_fits.data,
            method="auto",
            max_shift=fallback_max_shift,
        )

        if not getattr(result, "success", False) or getattr(result, "aligned_old", None) is None:
            logger.warning("标注集对齐失败: %s (%s)", pair.name, getattr(result, "error_message", ""))
            return False

        crop_bounds = self._pair_service.calc_overlap_crop_bounds(
            w=w,
            h=h,
            dx=result.dx,
            dy=result.dy,
            aligned_old=result.aligned_old,
            new_image=new_data,
        )
        if crop_bounds is None:
            logger.warning("标注集对齐后无有效重叠区域: %s", pair.name)
            return False

        x0, x1, y0, y1 = crop_bounds
        cropped_new = new_data[y0:y1, x0:x1]
        cropped_old = result.aligned_old[y0:y1, x0:x1]

        self._write_fits(new_aligned_path, cropped_new, new_fits.header)
        self._write_fits(old_aligned_path, cropped_old, old_fits.header)
        marker_text = (
            "aligned=1\n"
            f"dx={result.dx:.6f}\n"
            f"dy={result.dy:.6f}\n"
            f"crop={x0},{x1},{y0},{y1}\n"
        )
        new_marker_path.write_text(marker_text, encoding="utf-8")
        old_marker_path.write_text(marker_text, encoding="utf-8")
        return True

    def _collect_aligned_files(self, folder: Path) -> dict[str, Path]:
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
            files[self.strip_aligned_crop_suffix(file_path.stem)] = file_path
        return files

    def _ensure_marked_aligned_crop_file(
        self,
        root: Path,
        pair: FitsImagePair,
        new_aligned_path: Path,
        new_marker_path: Path,
    ) -> bool:
        marked_dir = root / "new_marked"
        if not marked_dir.is_dir():
            return False

        marked_source = marked_dir / f"{Path(pair.new_path).stem}.fits"
        if not marked_source.exists():
            for ext in (".fts", ".fit"):
                candidate = marked_dir / f"{Path(pair.new_path).stem}{ext}"
                if candidate.exists():
                    marked_source = candidate
                    break
        if not marked_source.exists():
            return False

        marked_aligned = marked_source.with_name(f"{marked_source.stem}__aligned_crop.fts")
        if marked_aligned.exists():
            return False

        crop_bounds = self.parse_crop_bounds_from_marker(new_marker_path)
        try:
            marked_fits = self._read_fits(marked_source)
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

            if new_aligned_path.exists() and (cropped.shape != self._read_fits(new_aligned_path).data.shape):
                aligned_shape = self._read_fits(new_aligned_path).data.shape
                ah, aw = aligned_shape[:2]
                h, w = cropped.shape[:2]
                y0 = max(0, (h - ah) // 2)
                x0 = max(0, (w - aw) // 2)
                cropped = cropped[y0:y0 + ah, x0:x0 + aw]

            self._write_fits(marked_aligned, cropped, marked_fits.header)
            return True
        except Exception as exc:
            logger.warning("带标记新图裁剪生成失败: %s (%s)", pair.name, exc)
            return False

    def _pair_aligned_files(
        self,
        new_files: dict[str, Path],
        old_files: dict[str, Path],
    ) -> list[tuple[str, Optional[Path], Optional[Path]]]:
        if not old_files:
            return [(sample_id, path, None) for sample_id, path in sorted(new_files.items())]

        old_norm_map = {
            self.normalize_pair_stem(stem): stem
            for stem in old_files
        }

        pairs: list[tuple[str, Optional[Path], Optional[Path]]] = []
        matched_old: set[str] = set()
        for sample_id, new_path in sorted(new_files.items()):
            old_stem = None
            if sample_id in old_files:
                old_stem = sample_id
            else:
                norm = self.normalize_pair_stem(sample_id)
                candidate = old_norm_map.get(norm)
                if candidate is not None and candidate not in matched_old:
                    old_stem = candidate

            if old_stem is None:
                continue

            matched_old.add(old_stem)
            pairs.append((sample_id, new_path, old_files[old_stem]))

        return pairs

    def _resolve_max_workers(self, job_count: int) -> int:
        if job_count <= 1:
            return 1
        if self._max_workers is not None:
            return max(1, min(job_count, self._max_workers))

        env_value = os.getenv("SCANN_DATASET_PREPROCESS_MAX_WORKERS", "").strip()
        if env_value:
            try:
                configured = int(env_value)
            except ValueError:
                configured = 0
            if configured > 0:
                return max(1, min(job_count, configured))

        default_workers = min(32, (os.cpu_count() or 1) + 4)
        return max(1, min(job_count, default_workers))

    def _run_jobs(self, jobs: list, worker: Callable) -> list:
        if not jobs:
            return []

        max_workers = self._resolve_max_workers(len(jobs))
        if max_workers <= 1:
            return [worker(job) for job in jobs]

        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="scann-preprocess",
        ) as executor:
            return list(executor.map(worker, jobs))
