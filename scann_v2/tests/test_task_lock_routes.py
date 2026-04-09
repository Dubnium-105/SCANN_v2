from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi import HTTPException

from scann.native_annotation.auth_service import AuthUser
from scann.native_annotation import routes as native_routes


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"SIMPLE FITS PLACEHOLDER")


def _annotator() -> AuthUser:
    return AuthUser(username="annotator", role="annotator")


def test_claim_next_task_rejects_blank_client_id(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    _touch(dataset_root / "new" / "PGC 17069.fts")

    monkeypatch.setenv("SCANN_NATIVE_DATASET_ROOT", str(dataset_root))

    with pytest.raises(HTTPException) as exc_info:
        native_routes.claim_next_task(client_id="   ", current_user=_annotator())

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "client_id cannot be empty"


def test_claim_task_route_rejects_other_client_for_locked_task(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"

    _touch(dataset_root / "new" / "PGC 17069.fts")
    _touch(dataset_root / "old" / "PGC 17069.fts")
    _touch(dataset_root / "new_marked" / "PGC 17069.fts")

    monkeypatch.setenv("SCANN_NATIVE_DATASET_ROOT", str(dataset_root))

    claimed = native_routes.claim_task(
        task_id="PGC 17069",
        client_id="client-a",
        current_user=_annotator(),
    )
    assert claimed.task_id == "PGC 17069"

    with pytest.raises(HTTPException) as exc_info:
        native_routes.claim_task(
            task_id="PGC 17069",
            client_id="client-b",
            current_user=_annotator(),
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Task locked by another client"


def test_release_task_lock_route_releases_task_for_next_client(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"

    _touch(dataset_root / "new" / "PGC 17069.fts")
    _touch(dataset_root / "old" / "PGC 17069.fts")
    _touch(dataset_root / "new_marked" / "PGC 17069.fts")

    monkeypatch.setenv("SCANN_NATIVE_DATASET_ROOT", str(dataset_root))

    claimed = native_routes.claim_next_task(client_id="client-a", current_user=_annotator())
    assert claimed.task_id == "PGC 17069"

    released = native_routes.release_task_lock(
        task_id="PGC 17069",
        client_id="client-a",
        current_user=_annotator(),
    )
    assert released.released is True
    assert released.task_id == "PGC 17069"

    reclaimed = native_routes.claim_next_task(client_id="client-b", current_user=_annotator())
    assert reclaimed.task_id == "PGC 17069"


def test_heartbeat_route_extends_lock_expiry(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"

    _touch(dataset_root / "new" / "PGC 17069.fts")
    _touch(dataset_root / "old" / "PGC 17069.fts")
    _touch(dataset_root / "new_marked" / "PGC 17069.fts")

    monkeypatch.setenv("SCANN_NATIVE_DATASET_ROOT", str(dataset_root))

    claimed = native_routes.claim_next_task(client_id="client-a", current_user=_annotator())
    initial_expires_at = datetime.fromisoformat(claimed.lock_expires_at)

    lock_service = native_routes.get_task_lock_service()
    existing_lock = lock_service.get_task_lock("PGC 17069")
    assert existing_lock is not None

    future_now = existing_lock.locked_at.replace(microsecond=0) + timedelta(seconds=1)
    monkeypatch.setattr(lock_service, "_now", lambda: future_now)

    refreshed = native_routes.heartbeat_task_lock(
        task_id="PGC 17069",
        client_id="client-a",
        current_user=_annotator(),
    )
    refreshed_expires_at = datetime.fromisoformat(refreshed.lock_expires_at)
    assert refreshed_expires_at > initial_expires_at

    with pytest.raises(HTTPException) as exc_info:
        native_routes.claim_next_task(client_id="client-b", current_user=_annotator())

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "No available task"


def test_release_and_heartbeat_require_lock_owner(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"

    _touch(dataset_root / "new" / "PGC 17069.fts")
    _touch(dataset_root / "old" / "PGC 17069.fts")
    _touch(dataset_root / "new_marked" / "PGC 17069.fts")

    monkeypatch.setenv("SCANN_NATIVE_DATASET_ROOT", str(dataset_root))

    claimed = native_routes.claim_next_task(client_id="client-a", current_user=_annotator())
    assert claimed.task_id == "PGC 17069"

    with pytest.raises(HTTPException) as release_exc:
        native_routes.release_task_lock(
            task_id="PGC 17069",
            client_id="client-b",
            current_user=_annotator(),
        )
    assert release_exc.value.status_code == 409
    assert release_exc.value.detail == "Task locked by another client"

    with pytest.raises(HTTPException) as heartbeat_exc:
        native_routes.heartbeat_task_lock(
            task_id="PGC 17069",
            client_id="client-b",
            current_user=_annotator(),
        )
    assert heartbeat_exc.value.status_code == 409
    assert heartbeat_exc.value.detail == "Task locked by another client"
