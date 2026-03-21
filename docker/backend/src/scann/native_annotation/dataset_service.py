from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class TaskSession(BaseModel):
    task_id: str
    new_path: str
    old_path: Optional[str] = None
    new_marked_path: Optional[str] = None


class DatasetService:
    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = dataset_root
        self.old_dir = dataset_root / "old"
        self.new_dir = dataset_root / "new"
        self.new_marked_dir = dataset_root / "new_marked"

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
            result[file_path.stem] = file_path.relative_to(self.dataset_root).as_posix()
        return result

    def list_tasks(self) -> list[TaskSession]:
        new_files = self._scan_dir(self.new_dir)
        old_files = self._scan_dir(self.old_dir)
        new_marked_files = self._scan_dir(self.new_marked_dir)

        tasks: list[TaskSession] = []
        for task_id in sorted(new_files.keys()):
            tasks.append(
                TaskSession(
                    task_id=task_id,
                    new_path=new_files[task_id],
                    old_path=old_files.get(task_id),
                    new_marked_path=new_marked_files.get(task_id),
                )
            )
        return tasks
