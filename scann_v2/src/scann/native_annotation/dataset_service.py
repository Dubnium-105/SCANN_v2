from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from scann.core.dataset_storage import DatasetStorage
from scann.services.dataset_preprocess_service import DatasetPreprocessService


class TaskSession(BaseModel):
    task_id: str
    new_path: str
    old_path: Optional[str] = None
    new_marked_path: Optional[str] = None
    field_key: Optional[str] = None
    field_name: Optional[str] = None
    capture_key: Optional[str] = None
    preprocess_status: Optional[str] = None


class DatasetService:
    def __init__(
        self,
        dataset_root: Path,
        preprocess_service: Optional[DatasetPreprocessService] = None,
    ) -> None:
        self.dataset_root = dataset_root
        self._storage = DatasetStorage(dataset_root)
        self._storage.ensure_schema()
        self.old_dir = dataset_root / "old"
        self.new_dir = dataset_root / "new"
        self.new_marked_dir = dataset_root / "new_marked"
        self._preprocess_service = preprocess_service or DatasetPreprocessService()

    def _task_metadata_by_id(self) -> dict[str, dict[str, str | None]]:
        metadata_by_id: dict[str, dict[str, str | None]] = {}
        for task in self._storage.list_tasks(active_only=False):
            metadata_by_id[task.task_id] = {
                "field_key": task.field_key,
                "field_name": task.field_name,
                "capture_key": task.capture_key,
                "preprocess_status": task.preprocess_status,
            }
        return metadata_by_id

    @staticmethod
    def _is_fits_file(path: Path) -> bool:
        return path.suffix.lower() in {".fts", ".fit", ".fits"}

    def _scan_dir(self, directory: Path) -> dict[str, str]:
        if not directory.exists() or not directory.is_dir():
            return {}

        result: dict[str, str] = {}
        for file_path in directory.iterdir():
            if not file_path.is_file() or not self._is_fits_file(file_path):
                continue
            task_id = self._normalize_task_id(file_path.stem)
            if not task_id:
                continue
            if task_id in result and not file_path.stem.lower().endswith("__aligned_crop"):
                continue
            result[task_id] = file_path.relative_to(self.dataset_root).as_posix()
        return result

    @staticmethod
    def _normalize_task_id(stem: str) -> str:
        normalized = stem.strip()
        if not normalized:
            return normalized
        date_token = DatasetPreprocessService.extract_datetime_prefix(normalized)
        field_name = DatasetStorage.normalize_field_name(normalized)
        if date_token and field_name:
            return f"{date_token}__{field_name}"
        stripped = DatasetPreprocessService.strip_aligned_crop_suffix(normalized)
        return field_name or stripped

    def _prepared_task_sessions(self) -> list[TaskSession]:
        tasks: list[TaskSession] = []
        metadata_by_id = self._task_metadata_by_id()
        for task in self._preprocess_service.collect_preprocessed_tasks(self.dataset_root):
            metadata = metadata_by_id.get(task.task_id, {})
            tasks.append(
                TaskSession(
                    task_id=task.task_id,
                    new_path=task.new_path.relative_to(self.dataset_root).as_posix(),
                    old_path=task.old_path.relative_to(self.dataset_root).as_posix() if task.old_path else None,
                    new_marked_path=(
                        task.new_marked_path.relative_to(self.dataset_root).as_posix()
                        if task.new_marked_path
                        else None
                    ),
                    field_key=str(metadata.get("field_key") or "") or None,
                    field_name=str(metadata.get("field_name") or "") or None,
                    capture_key=str(metadata.get("capture_key") or "") or None,
                    preprocess_status=str(metadata.get("preprocess_status") or "") or None,
                )
            )
        return tasks

    def list_tasks(self) -> list[TaskSession]:
        tasks = self._prepared_task_sessions()
        if tasks:
            return tasks

        new_files = self._scan_dir(self.new_dir)
        old_files = self._scan_dir(self.old_dir)
        new_marked_files = self._scan_dir(self.new_marked_dir)
        metadata_by_id = self._task_metadata_by_id()

        legacy_tasks: list[TaskSession] = []
        for task_id in sorted(new_files.keys()):
            metadata = metadata_by_id.get(task_id, {})
            legacy_tasks.append(
                TaskSession(
                    task_id=task_id,
                    new_path=new_files[task_id],
                    old_path=old_files.get(task_id),
                    new_marked_path=new_marked_files.get(task_id),
                    field_key=str(metadata.get("field_key") or "") or None,
                    field_name=str(metadata.get("field_name") or "") or None,
                    capture_key=str(metadata.get("capture_key") or "") or None,
                    preprocess_status=str(metadata.get("preprocess_status") or "") or None,
                )
            )

        self._preprocess_service.prepare_annotation_dataset(self.dataset_root)
        tasks = self._prepared_task_sessions()
        return tasks or legacy_tasks
