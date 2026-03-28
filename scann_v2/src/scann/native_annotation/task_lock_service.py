from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

from scann.core.dataset_storage import DatasetStorage

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
    def __init__(
        self,
        lock_timeout_seconds: int = 20 * 60,
        dataset_root: Path | None = None,
    ) -> None:
        self.lock_timeout_seconds = max(1, int(lock_timeout_seconds))
        self._locks_by_task: Dict[str, TaskLock] = {}
        self._task_by_client: Dict[str, str] = {}
        self._storage = DatasetStorage(Path(dataset_root)) if dataset_root is not None else None
        if self._storage is not None:
            self._storage.ensure_schema()

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _normalize_client_id(client_id: str) -> str:
        normalized_client_id = client_id.strip()
        if not normalized_client_id:
            raise ValueError("client_id cannot be empty")
        return normalized_client_id

    def _build_lock(self, task_id: str, client_id: str, *, now: datetime | None = None) -> TaskLock:
        locked_at = now or self._now()
        return TaskLock(
            task_id=task_id,
            client_id=client_id,
            locked_at=locked_at,
            expires_at=locked_at + timedelta(seconds=self.lock_timeout_seconds),
        )

    def _cleanup_expired_locks(self) -> None:
        now = self._now()
        if self._storage is not None:
            self._storage.clear_expired_claims(now.isoformat(timespec="seconds"))
            return

        expired_task_ids = [
            task_id for task_id, lock in self._locks_by_task.items() if lock.is_expired(now)
        ]
        for task_id in expired_task_ids:
            lock = self._locks_by_task.pop(task_id, None)
            if lock is not None and self._task_by_client.get(lock.client_id) == task_id:
                self._task_by_client.pop(lock.client_id, None)

    @staticmethod
    def _task_by_id(tasks: list[TaskSession]) -> dict[str, TaskSession]:
        return {task.task_id: task for task in tasks}

    @staticmethod
    def _parse_lock(task_id: str, client_id: str, locked_at: str | None, expires_at: str | None) -> TaskLock | None:
        if not client_id or not expires_at:
            return None
        try:
            expires_dt = datetime.fromisoformat(expires_at)
        except ValueError:
            return None
        try:
            locked_dt = datetime.fromisoformat(locked_at) if locked_at else expires_dt
        except ValueError:
            locked_dt = expires_dt
        return TaskLock(task_id=task_id, client_id=client_id, locked_at=locked_dt, expires_at=expires_dt)

    def claim_next_task(self, client_id: str, tasks: list[TaskSession]) -> Optional[TaskSession]:
        normalized_client_id = self._normalize_client_id(client_id)
        self._cleanup_expired_locks()

        if self._storage is not None:
            now = self._now()
            tasks_by_id = self._task_by_id(tasks)
            existing = self._storage.get_claimed_task_by_client(normalized_client_id)
            if existing is not None and existing.task_id in tasks_by_id:
                if existing.claim_expires_at:
                    try:
                        expires_at = datetime.fromisoformat(existing.claim_expires_at)
                    except ValueError:
                        expires_at = None
                else:
                    expires_at = None
                if expires_at is None or expires_at > now:
                    refreshed = self._build_lock(existing.task_id, normalized_client_id, now=now)
                    self._storage.refresh_claim(
                        task_id=existing.task_id,
                        client_id=normalized_client_id,
                        expires_at=refreshed.expires_at.isoformat(timespec="seconds"),
                    )
                    return tasks_by_id[existing.task_id]

            for task in tasks:
                new_lock = self._build_lock(task_id=task.task_id, client_id=normalized_client_id, now=now)
                claimed = self._storage.try_claim_task(
                    task_id=task.task_id,
                    client_id=normalized_client_id,
                    expires_at=new_lock.expires_at.isoformat(timespec="seconds"),
                    now_iso=now.isoformat(timespec="seconds"),
                )
                if claimed:
                    return task
            return None

        existing_task_id = self._task_by_client.get(normalized_client_id)
        if existing_task_id:
            for task in tasks:
                if task.task_id == existing_task_id:
                    refreshed_lock = self._build_lock(task_id=task.task_id, client_id=normalized_client_id)
                    self._locks_by_task[task.task_id] = refreshed_lock
                    self._task_by_client[normalized_client_id] = task.task_id
                    return task
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

    def refresh_task(self, task_id: str, client_id: str) -> Optional[TaskLock]:
        normalized_client_id = self._normalize_client_id(client_id)
        self._cleanup_expired_locks()

        if self._storage is not None:
            now = self._now()
            refreshed_lock = self._build_lock(task_id=task_id, client_id=normalized_client_id, now=now)
            ok = self._storage.refresh_claim(
                task_id=task_id,
                client_id=normalized_client_id,
                expires_at=refreshed_lock.expires_at.isoformat(timespec="seconds"),
            )
            return refreshed_lock if ok else None

        lock = self._locks_by_task.get(task_id)
        if lock is None or lock.client_id != normalized_client_id:
            return None

        refreshed_lock = self._build_lock(task_id=task_id, client_id=normalized_client_id)
        self._locks_by_task[task_id] = refreshed_lock
        self._task_by_client[normalized_client_id] = task_id
        return refreshed_lock

    def release_task(self, task_id: str, client_id: Optional[str] = None) -> bool:
        self._cleanup_expired_locks()
        if self._storage is not None:
            return self._storage.release_claim(task_id=task_id, client_id=client_id)

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
        if self._storage is not None:
            task = self._storage.get_task_by_id(task_id)
            if task is None:
                return None
            return self._parse_lock(
                task_id=task.task_id,
                client_id=str(task.claim_client_id or ""),
                locked_at=task.claim_locked_at,
                expires_at=task.claim_expires_at,
            )
        return self._locks_by_task.get(task_id)
