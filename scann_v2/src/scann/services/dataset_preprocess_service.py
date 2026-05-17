from __future__ import annotations

import logging
import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Deque, Optional
import json

import numpy as np

from scann.core.brightness_match import brightness_match_anchors
from scann.core.dataset_storage import DatasetStorage, RawAssetRecord, TaskArtifactRecord, TaskRecord
from scann.core.fits_io import read_fits, write_fits
from scann.core.image_aligner import align, align_with_rot180_selection
from scann.data.file_manager import FitsImagePair, match_new_old_pairs
from scann.services.pair_service import PairService

logger = logging.getLogger(__name__)

_FITS_EXTS = {".fits", ".fit", ".fts"}
_BRIGHTNESS_MATCH_DONE_MARKER = ".scann_brightness_match.done"
_MATCH_MAX_SAMPLES = 200000
_MATCH_HIGH_PERCENTILE = 99.9
_MATCH_HIGHLIGHT_SIGMA = 5.0
_MATCH_ADAPTIVE_HIGH_PERCENTILE = False
_TASK_MANIFEST_FILE = "preprocessed_tasks.json"
_TASK_MANIFEST_VERSION = "1.0"
_PREPROCESS_ALIGN_METHOD = "siril"


@dataclass(frozen=True)
class _PlannedTask:
    task_id: str
    field_key: str
    field_name: str
    new_raw_path: Path
    old_raw_path: Optional[Path]
    new_marked_raw_path: Optional[Path]


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
    total_task_count: int = 0
    align_failed_count: int = 0


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
        self._progress_callback: Optional[Callable[[int, int, str], None]] = None

    def set_progress_callback(
        self,
        callback: Optional[Callable[[int, int, str], None]],
    ) -> None:
        """注册预处理进度回调。"""
        self._progress_callback = callback

    def _emit_progress(self, current: int, total: int, message: str) -> None:
        callback = self._progress_callback
        if callback is None:
            return
        safe_total = max(1, int(total))
        safe_current = max(0, min(int(current), safe_total))
        callback(safe_current, safe_total, message)

    @staticmethod
    def _dataset_storage(root: Path) -> DatasetStorage:
        storage = DatasetStorage(Path(root))
        storage.ensure_schema()
        return storage

    @staticmethod
    def _ensure_dataset_dirs(root: Path) -> None:
        dataset_root = Path(root)
        for folder in (
            dataset_root / "dataset_raw" / "new",
            dataset_root / "dataset_raw" / "old",
            dataset_root / "dataset_raw" / "new_marked",
            dataset_root / "new",
            dataset_root / "old",
            dataset_root / "new_marked",
        ):
            folder.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _task_manifest_path(root: Path) -> Path:
        return Path(root) / _TASK_MANIFEST_FILE

    @staticmethod
    def _parse_date_typed_raw_dir(folder_name: str) -> tuple[str, str] | None:
        suffix_to_role = (
            ("_marked", "new_marked"),
            ("_mark", "new_marked"),
            ("_new", "new"),
        )
        for suffix, role in suffix_to_role:
            if not folder_name.endswith(suffix):
                continue
            date_token = folder_name[:-len(suffix)]
            if len(date_token) == 8 and date_token.isdigit():
                return role, date_token
        return None

    @classmethod
    def _raw_asset_sources(cls, dataset_root: Path) -> list[tuple[str, Path, str | None]]:
        raw_root = dataset_root / "dataset_raw"
        sources: list[tuple[str, Path, str | None]] = []

        for role in ("new", "old", "new_marked"):
            raw_dir = raw_root / role
            if raw_dir.is_dir():
                sources.append((role, raw_dir, None))

        if not raw_root.is_dir():
            return sources

        for raw_dir in sorted(raw_root.iterdir()):
            if not raw_dir.is_dir():
                continue
            parsed = cls._parse_date_typed_raw_dir(raw_dir.name)
            if parsed is None:
                continue
            role, date_token = parsed
            sources.append((role, raw_dir, date_token))

        return sources

    @classmethod
    def load_task_manifest(cls, root: Path) -> list[PreparedTaskPaths]:
        storage = DatasetStorage(Path(root))
        try:
            rows = storage.list_prepared_task_paths()
        except Exception:
            rows = []
        if rows:
            tasks: list[PreparedTaskPaths] = []
            dataset_root = Path(root)
            for row in rows:
                new_rel = row.get("new_path")
                if not isinstance(new_rel, str) or not new_rel:
                    continue
                tasks.append(
                    PreparedTaskPaths(
                        task_id=str(row["task_id"]),
                        new_path=dataset_root / new_rel,
                        old_path=(
                            (dataset_root / row["old_path"])
                            if isinstance(row.get("old_path"), str) and row.get("old_path")
                            else None
                        ),
                        new_marked_path=(
                            (dataset_root / row["new_marked_path"])
                            if isinstance(row.get("new_marked_path"), str) and row.get("new_marked_path")
                            else None
                        ),
                    )
                )
            if tasks:
                return tasks

        manifest_path = cls._task_manifest_path(root)
        if not manifest_path.is_file():
            return []

        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("预处理任务清单解析失败: %s (%s)", manifest_path, exc)
            return []

        tasks_payload = payload.get("tasks")
        if not isinstance(tasks_payload, list):
            return []

        dataset_root = Path(root)
        tasks: list[PreparedTaskPaths] = []
        for item in tasks_payload:
            if not isinstance(item, dict):
                continue

            task_id = str(item.get("task_id") or "").strip()
            new_rel = item.get("new_path")
            if not task_id or not isinstance(new_rel, str) or not new_rel.strip():
                continue

            new_path = dataset_root / new_rel
            old_rel = item.get("old_path")
            marked_rel = item.get("new_marked_path")
            task = PreparedTaskPaths(
                task_id=task_id,
                new_path=new_path,
                old_path=(dataset_root / old_rel) if isinstance(old_rel, str) and old_rel.strip() else None,
                new_marked_path=(
                    (dataset_root / marked_rel)
                    if isinstance(marked_rel, str) and marked_rel.strip()
                    else None
                ),
            )
            if task.new_path.is_file():
                tasks.append(task)
        return tasks

    @classmethod
    def write_task_manifest(cls, root: Path, tasks: list[PreparedTaskPaths]) -> None:
        dataset_root = Path(root)
        manifest_path = cls._task_manifest_path(dataset_root)
        payload = {
            "version": _TASK_MANIFEST_VERSION,
            "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "tasks": [
                {
                    "task_id": task.task_id,
                    "new_path": task.new_path.relative_to(dataset_root).as_posix(),
                    "old_path": task.old_path.relative_to(dataset_root).as_posix() if task.old_path else None,
                    "new_marked_path": (
                        task.new_marked_path.relative_to(dataset_root).as_posix()
                        if task.new_marked_path
                        else None
                    ),
                }
                for task in tasks
            ],
        }
        manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _scan_raw_assets(self, root: Path) -> list[RawAssetRecord]:
        dataset_root = Path(root)
        assets: list[RawAssetRecord] = []
        for role, raw_dir, date_dir_token in self._raw_asset_sources(dataset_root):
            for file_path in sorted(raw_dir.iterdir()):
                if not file_path.is_file() or file_path.suffix.lower() not in _FITS_EXTS:
                    continue
                try:
                    stat = file_path.stat()
                except OSError:
                    continue
                field_name = DatasetStorage.normalize_field_name(file_path.stem)
                field_key = DatasetStorage.normalize_field_key(file_path.stem)
                capture_key = DatasetStorage.normalize_capture_key(file_path.stem)
                metadata = {"raw": True}
                if date_dir_token is not None and role in {"new", "new_marked"}:
                    capture_key = f"{date_dir_token}:{capture_key}"
                    metadata.update(
                        {
                            "date_type_input": True,
                            "date_folder": date_dir_token,
                            "date_type_folder": raw_dir.name,
                        }
                    )
                date_obs = self._extract_date_obs_token(file_path) or date_dir_token
                assets.append(
                    RawAssetRecord(
                        asset_id=uuid.uuid5(uuid.NAMESPACE_URL, file_path.relative_to(dataset_root).as_posix()).hex,
                        asset_role=role,
                        field_key=field_key,
                        field_name=field_name or file_path.stem,
                        capture_key=capture_key,
                        relpath=file_path.relative_to(dataset_root).as_posix(),
                        file_name=file_path.name,
                        file_stem=file_path.stem,
                        suffix=file_path.suffix.lower(),
                        date_obs=date_obs,
                        size_bytes=int(stat.st_size),
                        modified_time=float(stat.st_mtime),
                        metadata=metadata,
                    )
                )
        return assets

    def _scan_annotation_assets(self, root: Path) -> list[RawAssetRecord]:
        dataset_root = Path(root)
        assets: list[RawAssetRecord] = []
        for role in ("new", "old", "new_marked"):
            work_dir = dataset_root / role
            if not work_dir.is_dir():
                continue
            for file_path in sorted(work_dir.iterdir()):
                if not file_path.is_file() or file_path.suffix.lower() not in _FITS_EXTS:
                    continue
                try:
                    stat = file_path.stat()
                except OSError:
                    continue
                field_name = DatasetStorage.normalize_field_name(file_path.stem)
                field_key = DatasetStorage.normalize_field_key(file_path.stem)
                capture_key = DatasetStorage.normalize_capture_key(file_path.stem)
                date_obs = self.extract_datetime_prefix(file_path.stem)
                assets.append(
                    RawAssetRecord(
                        asset_id=uuid.uuid5(uuid.NAMESPACE_URL, file_path.relative_to(dataset_root).as_posix()).hex,
                        asset_role=role,
                        field_key=field_key,
                        field_name=field_name or file_path.stem,
                        capture_key=capture_key,
                        relpath=file_path.relative_to(dataset_root).as_posix(),
                        file_name=file_path.name,
                        file_stem=file_path.stem,
                        suffix=file_path.suffix.lower(),
                        date_obs=date_obs,
                        size_bytes=int(stat.st_size),
                        modified_time=float(stat.st_mtime),
                        metadata={"native_annotation": True},
                    )
                )
        return assets

    def _migrate_legacy_inputs_to_raw(
        self,
        root: Path,
        *,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> int:
        dataset_root = Path(root)
        raw_root = dataset_root / "dataset_raw"
        folder_names = ("new", "old", "new_marked")
        jobs: list[tuple[Path, Path, Path]] = []

        for folder_name in folder_names:
            work_dir = dataset_root / folder_name
            if not work_dir.is_dir():
                continue
            raw_dir = raw_root / folder_name
            raw_dir.mkdir(parents=True, exist_ok=True)
            for file_path in sorted(work_dir.iterdir()):
                if not self._should_standardize_file(file_path):
                    continue
                jobs.append((file_path, raw_dir, work_dir))
        def _on_job_progress(current: int, total: int, job: object) -> None:
            if progress_callback is None:
                return
            if not isinstance(job, tuple) or not job:
                progress_callback(current, total, "unknown")
                return
            src_path = job[0]
            if not isinstance(src_path, Path):
                progress_callback(current, total, str(src_path))
                return
            progress_callback(current, total, src_path.name)

        return sum(
            self._run_jobs(
                jobs,
                self._standardize_single_file,
                on_progress_detail=_on_job_progress,
            )
        )

    @staticmethod
    def _date_token_from_asset(asset: RawAssetRecord) -> str | None:
        if not asset.date_obs:
            return None
        compact = asset.date_obs.strip()
        if len(compact) >= 15 and compact[8].lower() == "t":
            return compact[:15]
        return compact or None

    @staticmethod
    def _is_date_typed_raw_asset(asset: RawAssetRecord) -> bool:
        metadata = asset.metadata or {}
        return bool(metadata.get("date_type_input"))

    @staticmethod
    def _pop_marked_asset(
        queue: Deque[RawAssetRecord] | None,
        used_asset_ids: set[str],
    ) -> RawAssetRecord | None:
        if queue is None:
            return None
        while queue:
            candidate = queue.popleft()
            if candidate.asset_id in used_asset_ids:
                continue
            used_asset_ids.add(candidate.asset_id)
            return candidate
        return None

    def _plan_tasks(self, root: Path) -> list[_PlannedTask]:
        storage = self._dataset_storage(root)
        new_assets = storage.list_raw_assets("new")
        old_assets = storage.list_raw_assets("old")
        marked_assets = storage.list_raw_assets("new_marked")

        old_by_field: dict[str, RawAssetRecord] = {}
        for asset in old_assets:
            old_by_field.setdefault(asset.field_key, asset)

        marked_by_capture: dict[str, Deque[RawAssetRecord]] = defaultdict(deque)
        marked_by_field: dict[str, Deque[RawAssetRecord]] = defaultdict(deque)
        for asset in marked_assets:
            marked_by_capture[asset.capture_key].append(asset)
            if not self._is_date_typed_raw_asset(asset):
                marked_by_field[asset.field_key].append(asset)

        planned: list[_PlannedTask] = []
        task_rows: list[TaskRecord] = []
        used_marked_asset_ids: set[str] = set()
        for asset in new_assets:
            existing_task = storage.get_task_by_new_asset_id(asset.asset_id)
            task_id = (
                existing_task.task_id
                if existing_task is not None
                else storage.allocate_task_id(
                    date_token=self._date_token_from_asset(asset),
                    field_name=asset.field_name,
                    capture_key=asset.capture_key,
                )
            )
            old_asset = old_by_field.get(asset.field_key)
            marked_asset = self._pop_marked_asset(
                marked_by_capture.get(asset.capture_key),
                used_marked_asset_ids,
            )
            if marked_asset is None and not self._is_date_typed_raw_asset(asset):
                marked_asset = self._pop_marked_asset(
                    marked_by_field.get(asset.field_key),
                    used_marked_asset_ids,
                )
            task_rows.append(
                TaskRecord(
                    task_id=task_id,
                    field_key=asset.field_key,
                    field_name=asset.field_name,
                    capture_key=asset.capture_key,
                    new_asset_id=asset.asset_id,
                    old_asset_id=old_asset.asset_id if old_asset else None,
                    new_marked_asset_id=marked_asset.asset_id if marked_asset else None,
                    preprocess_status="pending",
                )
            )
            planned.append(
                _PlannedTask(
                    task_id=task_id,
                    field_key=asset.field_key,
                    field_name=asset.field_name,
                    new_raw_path=Path(root) / asset.relpath,
                    old_raw_path=(Path(root) / old_asset.relpath) if old_asset else None,
                    new_marked_raw_path=(Path(root) / marked_asset.relpath) if marked_asset else None,
                )
            )
        storage.sync_tasks(task_rows)
        return planned

    def prepare_dataset(self, root: Path) -> DatasetPreprocessReport:
        dataset_root = Path(root)
        logger.info("开始预处理数据集: %s", dataset_root)
        self._emit_progress(0, 100, f"预处理初始化: {dataset_root}")
        self._ensure_dataset_dirs(dataset_root)
        self._emit_progress(5, 100, "目录检查完成")

        self._emit_progress(6, 100, "开始标准化原始数据")

        def _standardize_progress(current: int, total: int, item_name: str) -> None:
            step_total = max(1, total)
            percent = 6 + int((current / step_total) * 19)
            self._emit_progress(
                percent,
                100,
                f"标准化迁移处理中: {current}/{step_total} · {item_name}",
            )
            logger.info(
                "标准化迁移进度: %d/%d, 文件=%s",
                current,
                step_total,
                item_name,
            )

        standardized_files = self.standardize_dataset_by_date_obs(
            dataset_root,
            progress_callback=_standardize_progress,
        )
        self._emit_progress(30, 100, f"标准化完成: {standardized_files} 文件")
        logger.info("标准化完成: %s, 文件数=%d", dataset_root, standardized_files)

        self._emit_progress(31, 100, "开始亮度匹配")
        brightness_matched_files = self.apply_initial_brightness_match(dataset_root)
        self._emit_progress(40, 100, f"亮度匹配完成: {brightness_matched_files} 文件")
        logger.info("亮度匹配完成: %s, 文件数=%d", dataset_root, brightness_matched_files)

        def _aligned_progress(current: int, total: int, task_id: str) -> None:
            step_total = max(1, total)
            percent = 40 + int((current / step_total) * 45)
            self._emit_progress(
                percent,
                100,
                f"对齐裁剪处理中: {current}/{step_total} · {task_id}",
            )
            logger.info(
                "对齐裁剪进度: %d/%d, 任务=%s",
                current,
                step_total,
                task_id,
            )

        reused_aligned_pairs, generated_aligned_pairs, generated_marked_crops = (
            self.ensure_aligned_crop_files(dataset_root, progress_callback=_aligned_progress)
        )
        self._emit_progress(90, 100, "对齐裁剪阶段完成")
        logger.info(
            "对齐裁剪完成: %s, 复用=%d, 新生成=%d, 标注裁剪=%d",
            dataset_root,
            reused_aligned_pairs,
            generated_aligned_pairs,
            generated_marked_crops,
        )
        self._emit_progress(92, 100, "汇总任务状态")
        task_rows = self._dataset_storage(dataset_root).list_tasks(active_only=True)
        tasks = self.collect_preprocessed_tasks(dataset_root)
        self._emit_progress(95, 100, f"任务清单生成完成: {len(tasks)}")
        self.write_task_manifest(dataset_root, tasks)
        task_count = len(tasks)
        total_task_count = len(task_rows)
        align_failed_count = sum(1 for task in task_rows if task.preprocess_status == "align_failed")
        self._emit_progress(
            100,
            100,
            f"预处理完成: 总任务 {total_task_count}, 就绪 {task_count}, 对齐失败 {align_failed_count}",
        )
        logger.info(
            "预处理完成: %s, 总任务=%d, 就绪=%d, 对齐失败=%d",
            dataset_root,
            total_task_count,
            task_count,
            align_failed_count,
        )
        return DatasetPreprocessReport(
            standardized_files=standardized_files,
            brightness_matched_files=brightness_matched_files,
            reused_aligned_pairs=reused_aligned_pairs,
            generated_aligned_pairs=generated_aligned_pairs,
            generated_marked_crops=generated_marked_crops,
            task_count=task_count,
            total_task_count=total_task_count,
            align_failed_count=align_failed_count,
        )

    def prepare_annotation_dataset(self, root: Path) -> DatasetPreprocessReport:
        dataset_root = Path(root)
        logger.info("开始为原生标注准备数据集: %s", dataset_root)
        self._ensure_dataset_dirs(dataset_root)

        storage = self._dataset_storage(dataset_root)
        annotation_assets = self._scan_annotation_assets(dataset_root)
        storage.upsert_raw_assets(annotation_assets)

        planned_tasks = self._plan_tasks(dataset_root)
        for task in planned_tasks:
            storage.update_task_preprocess_state(task.task_id, preprocess_status="ready")

        task_rows = storage.list_tasks(active_only=True)
        tasks = self.collect_preprocessed_tasks(dataset_root)
        self.write_task_manifest(dataset_root, tasks)
        return DatasetPreprocessReport(
            standardized_files=0,
            brightness_matched_files=0,
            reused_aligned_pairs=0,
            generated_aligned_pairs=0,
            generated_marked_crops=0,
            task_count=len(tasks),
            total_task_count=len(task_rows),
            align_failed_count=0,
        )

    def standardize_dataset_by_date_obs(
        self,
        root: Path,
        *,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> int:
        self._ensure_dataset_dirs(root)
        migrated = self._migrate_legacy_inputs_to_raw(root, progress_callback=progress_callback)
        assets = self._scan_raw_assets(root)
        self._dataset_storage(root).upsert_raw_assets(assets)
        self._plan_tasks(root)
        return migrated if migrated > 0 else len(assets)

    def ensure_aligned_crop_files(
        self,
        root: Path,
        *,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> tuple[int, int, int]:
        planned_tasks = self._plan_tasks(root)

        def _on_task_progress(current: int, total: int, task: object) -> None:
            if progress_callback is None:
                return
            task_id = task.task_id if isinstance(task, _PlannedTask) else str(task)
            progress_callback(current, total, task_id)

        results = self._run_jobs(
            planned_tasks,
            lambda task: self._ensure_planned_task(root, task),
            on_progress_detail=_on_task_progress,
        )
        reused_aligned_pairs = sum(item[0] for item in results)
        generated_aligned_pairs = sum(item[1] for item in results)
        generated_marked_crops = sum(item[2] for item in results)
        return reused_aligned_pairs, generated_aligned_pairs, generated_marked_crops

    def apply_initial_brightness_match(self, root: Path) -> int:
        return 0

    def collect_preprocessed_tasks(self, root: Path) -> list[PreparedTaskPaths]:
        rows = self._dataset_storage(root).list_prepared_task_paths()
        tasks: list[PreparedTaskPaths] = []
        for row in rows:
            new_rel = row.get("new_path")
            if not isinstance(new_rel, str) or not new_rel:
                continue
            tasks.append(
                PreparedTaskPaths(
                    task_id=str(row["task_id"]),
                    new_path=Path(root) / new_rel,
                    old_path=(Path(root) / row["old_path"]) if isinstance(row.get("old_path"), str) and row.get("old_path") else None,
                    new_marked_path=(
                        (Path(root) / row["new_marked_path"])
                        if isinstance(row.get("new_marked_path"), str) and row.get("new_marked_path")
                        else None
                    ),
                )
            )
        if tasks:
            self.write_task_manifest(root, tasks)
        return tasks

    def collect_aligned_pairs(
        self,
        root: Path,
    ) -> list[tuple[str, Optional[Path], Optional[Path]]]:
        return [
            (task.task_id, task.new_path, task.old_path)
            for task in self.collect_preprocessed_tasks(root)
        ]

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

    @staticmethod
    def _task_artifact_paths(root: Path, task_id: str) -> tuple[Path, Path, Path, Path, Path | None]:
        dataset_root = Path(root)
        new_aligned_path = dataset_root / "new" / f"{task_id}__aligned_crop.fts"
        old_aligned_path = dataset_root / "old" / f"{task_id}__aligned_crop.fts"
        new_marker_path = dataset_root / "new" / f"{task_id}__aligned.marker"
        old_marker_path = dataset_root / "old" / f"{task_id}__aligned.marker"
        marked_aligned_path = dataset_root / "new_marked" / f"{task_id}__aligned_crop.fts"
        return new_aligned_path, old_aligned_path, new_marker_path, old_marker_path, marked_aligned_path

    def _ensure_planned_task(
        self,
        root: Path,
        task: _PlannedTask,
    ) -> tuple[int, int, int]:
        storage = self._dataset_storage(root)
        if task.old_raw_path is None or not task.old_raw_path.is_file():
            storage.update_task_preprocess_state(task.task_id, preprocess_status="missing_old")
            return 0, 0, 0

        new_aligned_path, old_aligned_path, new_marker_path, old_marker_path, marked_aligned_path = (
            self._task_artifact_paths(root, task.task_id)
        )
        if new_aligned_path.is_file() and old_aligned_path.is_file():
            generated_marked_crop = 0
            if task.new_marked_raw_path is not None and marked_aligned_path is not None:
                generated_marked_crop = int(
                    self._ensure_marked_task_crop(
                        task,
                        new_aligned_path=new_aligned_path,
                        new_marker_path=new_marker_path,
                        marked_aligned_path=marked_aligned_path,
                    )
                )
            storage.upsert_task_artifact(
                TaskArtifactRecord(task_id=task.task_id, artifact_role="aligned_new", relpath=new_aligned_path.relative_to(root).as_posix())
            )
            storage.upsert_task_artifact(
                TaskArtifactRecord(task_id=task.task_id, artifact_role="aligned_old", relpath=old_aligned_path.relative_to(root).as_posix())
            )
            if generated_marked_crop:
                storage.upsert_task_artifact(
                    TaskArtifactRecord(
                        task_id=task.task_id,
                        artifact_role="aligned_new_marked",
                        relpath=marked_aligned_path.relative_to(root).as_posix(),
                    )
                )
            storage.update_task_preprocess_state(task.task_id, preprocess_status="ready")
            return 1, 0, generated_marked_crop

        generated_aligned_pair = int(
            self._align_planned_task_to_crop(
                root=root,
                task=task,
                new_aligned_path=new_aligned_path,
                old_aligned_path=old_aligned_path,
                new_marker_path=new_marker_path,
                old_marker_path=old_marker_path,
            )
        )
        if not generated_aligned_pair:
            storage.update_task_preprocess_state(task.task_id, preprocess_status="align_failed")
            return 0, 0, 0

        storage.upsert_task_artifact(
            TaskArtifactRecord(task_id=task.task_id, artifact_role="aligned_new", relpath=new_aligned_path.relative_to(root).as_posix())
        )
        storage.upsert_task_artifact(
            TaskArtifactRecord(task_id=task.task_id, artifact_role="aligned_old", relpath=old_aligned_path.relative_to(root).as_posix())
        )

        generated_marked_crop = 0
        if task.new_marked_raw_path is not None and marked_aligned_path is not None:
            generated_marked_crop = int(
                self._ensure_marked_task_crop(
                    task,
                    new_aligned_path=new_aligned_path,
                    new_marker_path=new_marker_path,
                    marked_aligned_path=marked_aligned_path,
                )
            )
            if generated_marked_crop:
                storage.upsert_task_artifact(
                    TaskArtifactRecord(
                        task_id=task.task_id,
                        artifact_role="aligned_new_marked",
                        relpath=marked_aligned_path.relative_to(root).as_posix(),
                    )
                )
        storage.update_task_preprocess_state(task.task_id, preprocess_status="ready")
        return 0, 1, generated_marked_crop

    def _align_planned_task_to_crop(
        self,
        *,
        root: Path,
        task: _PlannedTask,
        new_aligned_path: Path,
        old_aligned_path: Path,
        new_marker_path: Path,
        old_marker_path: Path,
    ) -> bool:
        try:
            new_fits = self._read_fits(task.new_raw_path)
            old_fits = self._read_fits(task.old_raw_path)
        except Exception as exc:
            logger.warning("标注集对齐读取失败: %s (%s)", task.task_id, exc)
            return False

        new_data = np.nan_to_num(new_fits.data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        old_data = np.nan_to_num(old_fits.data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        try:
            old_data_for_align = self._match_data_to_reference(new_data, old_data)
        except Exception:
            old_data_for_align = old_data

        h, w = new_data.shape[:2]
        fallback_max_shift = max(100, int(min(h, w) * 0.45))
        logger.info(
            "任务对齐使用 Siril: task_id=%s, max_shift=%s",
            task.task_id,
            fallback_max_shift,
        )
        result = self._align(
            new_data,
            old_data_for_align,
            method=_PREPROCESS_ALIGN_METHOD,
            max_shift=fallback_max_shift,
        )
        if not getattr(result, "success", False) or getattr(result, "aligned_old", None) is None:
            logger.warning("标注集对齐失败: %s (%s)", task.task_id, getattr(result, "error_message", ""))
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
            logger.warning("标注集对齐后无有效重叠区域: %s", task.task_id)
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
        self._dataset_storage(root).update_task_preprocess_state(
            task.task_id,
            preprocess_status="ready",
            crop_bounds=crop_bounds,
            align_dx=float(result.dx),
            align_dy=float(result.dy),
        )
        return True

    def _ensure_marked_task_crop(
        self,
        task: _PlannedTask,
        *,
        new_aligned_path: Path,
        new_marker_path: Path,
        marked_aligned_path: Path,
    ) -> bool:
        if task.new_marked_raw_path is None or not task.new_marked_raw_path.is_file():
            return False
        if marked_aligned_path.exists():
            return False

        crop_bounds = self.parse_crop_bounds_from_marker(new_marker_path)
        try:
            target_shape = None
            if new_aligned_path.exists():
                target_shape = self._read_fits(new_aligned_path).data.shape[:2]
            cropped, marked_header = self._build_marked_aligned_crop(
                reference_new_path=task.new_raw_path,
                marked_source_path=task.new_marked_raw_path,
                crop_bounds=crop_bounds,
                target_shape=target_shape,
                log_context=task.task_id,
            )
            self._write_fits(marked_aligned_path, cropped, marked_header)
            return True
        except Exception as exc:
            logger.warning("带标记新图裁剪生成失败: %s (%s)", task.task_id, exc)
            return False

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
        logger.info(
            "配对对齐使用 Siril: pair=%s, max_shift=%s",
            pair.name,
            fallback_max_shift,
        )
        result = self._align(
            new_fits.data,
            old_fits.data,
            method=_PREPROCESS_ALIGN_METHOD,
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
            target_shape = None
            if new_aligned_path.exists():
                target_shape = self._read_fits(new_aligned_path).data.shape[:2]
            cropped, marked_header = self._build_marked_aligned_crop(
                reference_new_path=Path(pair.new_path),
                marked_source_path=marked_source,
                crop_bounds=crop_bounds,
                target_shape=target_shape,
                log_context=pair.name,
            )
            self._write_fits(marked_aligned, cropped, marked_header)
            return True
        except Exception as exc:
            logger.warning("带标记新图裁剪生成失败: %s (%s)", pair.name, exc)
            return False

    @staticmethod
    def _sanitize_image_data(data: np.ndarray) -> np.ndarray:
        return np.nan_to_num(
            np.asarray(data, dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def _align_marked_to_new(
        self,
        reference_new_data: np.ndarray,
        marked_data: np.ndarray,
        *,
        log_context: str,
    ) -> np.ndarray:
        aligned_marked = marked_data
        if reference_new_data.shape[:2] != marked_data.shape[:2]:
            logger.warning(
                "带标记新图与新图尺寸不一致，无法执行额外对齐: %s; new=%s marked=%s",
                log_context,
                reference_new_data.shape,
                marked_data.shape,
            )
            return aligned_marked

        max_shift = max(100, int(min(reference_new_data.shape[:2]) * 0.45))
        result, attempt_name, original_score, rotated_score = align_with_rot180_selection(
            reference_new_data,
            marked_data,
            method="auto",
            max_shift=max_shift,
            align_fn=self._align,
        )
        if attempt_name == "rot180" and rotated_score > original_score + 1e-3:
            logger.info(
                "检测到带标记新图更接近旋转180度版本，优先旋转后对齐: %s (original=%.4f, rot180=%.4f)",
                log_context,
                original_score,
                rotated_score,
            )
        if getattr(result, "success", False) and getattr(result, "aligned_old", None) is not None:
            if attempt_name == "rot180":
                logger.info("带标记新图旋转180度后对齐成功: %s", log_context)
            return self._sanitize_image_data(result.aligned_old)

        logger.warning(
            "带标记新图对齐新图失败，旋转180度兜底后仍未成功: %s (%s)",
            log_context,
            getattr(result, "error_message", ""),
        )
        return aligned_marked

    def _build_marked_aligned_crop(
        self,
        *,
        reference_new_path: Path,
        marked_source_path: Path,
        crop_bounds: tuple[int, int, int, int] | None,
        target_shape: tuple[int, int] | None,
        log_context: str,
    ) -> tuple[np.ndarray, object]:
        reference_new_fits = self._read_fits(reference_new_path)
        marked_fits = self._read_fits(marked_source_path)
        reference_new_data = self._sanitize_image_data(reference_new_fits.data)
        marked_data = self._sanitize_image_data(marked_fits.data)
        aligned_marked = self._align_marked_to_new(
            reference_new_data,
            marked_data,
            log_context=log_context,
        )

        cropped = aligned_marked
        if crop_bounds is not None:
            x0, x1, y0, y1 = crop_bounds
            if 0 <= x0 < x1 <= aligned_marked.shape[1] and 0 <= y0 < y1 <= aligned_marked.shape[0]:
                cropped = aligned_marked[y0:y1, x0:x1]

        if target_shape is not None and cropped.shape[:2] != target_shape:
            ah, aw = target_shape
            h, w = cropped.shape[:2]
            y0 = max(0, (h - ah) // 2)
            x0 = max(0, (w - aw) // 2)
            cropped = cropped[y0:y0 + ah, x0:x0 + aw]

        return cropped, marked_fits.header

    def _pair_aligned_files(
        self,
        new_files: dict[str, Path],
        old_files: dict[str, Path],
    ) -> list[tuple[str, Optional[Path], Optional[Path]]]:
        if not old_files:
            return [(sample_id, path, None) for sample_id, path in sorted(new_files.items())]

        old_norm_map: dict[str, Deque[str]] = defaultdict(deque)
        for stem in sorted(old_files):
            old_norm_map[self.normalize_pair_stem(stem)].append(stem)

        pairs: list[tuple[str, Optional[Path], Optional[Path]]] = []
        matched_old: set[str] = set()
        for sample_id, new_path in sorted(new_files.items()):
            old_stem = None
            if sample_id in old_files:
                old_stem = sample_id
            else:
                norm = self.normalize_pair_stem(sample_id)
                candidates = old_norm_map.get(norm)
                if candidates is not None:
                    while candidates and candidates[0] in matched_old:
                        candidates.popleft()
                    if candidates:
                        old_stem = candidates.popleft()

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

    def _run_jobs(
        self,
        jobs: list,
        worker: Callable,
        *,
        on_progress: Optional[Callable[[int, int], None]] = None,
        on_progress_detail: Optional[Callable[[int, int, object], None]] = None,
    ) -> list:
        if not jobs:
            return []

        total = len(jobs)

        max_workers = self._resolve_max_workers(len(jobs))
        if max_workers <= 1:
            results: list = []
            for index, job in enumerate(jobs, start=1):
                results.append(worker(job))
                if on_progress is not None:
                    on_progress(index, total)
                if on_progress_detail is not None:
                    on_progress_detail(index, total, job)
            return results

        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="scann-preprocess",
        ) as executor:
            future_to_index = {
                executor.submit(worker, job): idx
                for idx, job in enumerate(jobs)
            }
            results: list = [None] * total
            completed = 0
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                results[idx] = future.result()
                completed += 1
                if on_progress is not None:
                    on_progress(completed, total)
                if on_progress_detail is not None:
                    on_progress_detail(completed, total, jobs[idx])
            return results
