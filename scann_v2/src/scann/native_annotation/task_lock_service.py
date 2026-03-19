from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from .dataset_service import TaskSession


@dataclass
class TaskLock:
    task_id: str
    client_id: str
    locked_at: datetime
    expires_at: datetime

    def is_expired(self, now: datetime) -> bool:
        return now >= self.expires_at


class TaskLockService:
    def __init__(self, lock_timeout_seconds: int = 20 * 60) -> None:
        self.lock_timeout_seconds = max(1, int(lock_timeout_seconds))
        self._locks_by_task: Dict[str, TaskLock] = {}
        self._task_by_client: Dict[str, str] = {}

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    def _cleanup_expired_locks(self) -> None:
        now = self._now()
        expired_task_ids = [
            task_id for task_id, lock in self._locks_by_task.items() if lock.is_expired(now)
        ]
        for task_id in expired_task_ids:
            lock = self._locks_by_task.pop(task_id, None)
            if lock is not None and self._task_by_client.get(lock.client_id) == task_id:
                self._task_by_client.pop(lock.client_id, None)

    def _build_lock(self, task_id: str, client_id: str) -> TaskLock:
        now = self._now()
        return TaskLock(
            task_id=task_id,
            client_id=client_id,
            locked_at=now,
            expires_at=now + timedelta(seconds=self.lock_timeout_seconds),
        )

    def claim_next_task(self, client_id: str, tasks: list[TaskSession]) -> Optional[TaskSession]:
        normalized_client_id = client_id.strip()
        if not normalized_client_id:
            raise ValueError("client_id cannot be empty")

        self._cleanup_expired_locks()

        existing_task_id = self._task_by_client.get(normalized_client_id)
        if existing_task_id:
            for task in tasks:
                if task.task_id == existing_task_id:
                    refreshed_lock = self._build_lock(task_id=task.task_id, client_id=normalized_client_id)
                    self._locks_by_task[task.task_id] = refreshed_lock
                    self._task_by_client[normalized_client_id] = task.task_id
                    return task

            # 任务列表发生变化，移除悬空映射
            self._task_by_client.pop(normalized_client_id, None)
            self._locks_by_task.pop(existing_task_id, None)

        for task in tasks:
            lock = self._locks_by_task.get(task.task_id)
            if lock is not None and lock.client_id != normalized_client_id:
                continue

            new_lock = self._build_lock(task_id=task.task_id, client_id=normalized_client_id)
            self._locks_by_task[task.task_id] = new_lock
            self._task_by_client[normalized_client_id] = task.task_id
            return task

        return None

    def release_task(self, task_id: str, client_id: Optional[str] = None) -> bool:
        self._cleanup_expired_locks()
        lock = self._locks_by_task.get(task_id)
        if lock is None:
            return False

        if client_id is not None and lock.client_id != client_id:
            return False

        self._locks_by_task.pop(task_id, None)
        if self._task_by_client.get(lock.client_id) == task_id:
            self._task_by_client.pop(lock.client_id, None)
        return True

    def get_task_lock(self, task_id: str) -> Optional[TaskLock]:
        self._cleanup_expired_locks()
        return self._locks_by_task.get(task_id)
