from __future__ import annotations

from datetime import datetime, timedelta, timezone

from scann.native_annotation.worker_liveness import (
    effective_worker_status,
    parse_worker_timestamp,
)


def test_effective_worker_status_marks_stale_online_worker_offline() -> None:
    now = datetime(2026, 7, 31, 0, 0, tzinfo=timezone.utc)
    last_seen = (now - timedelta(minutes=4)).isoformat()

    assert effective_worker_status(
        "online",
        last_seen,
        offline_after_seconds=180,
        now=now,
    ) == "offline"


def test_effective_worker_status_keeps_recent_worker_online() -> None:
    now = datetime(2026, 7, 31, 0, 0, tzinfo=timezone.utc)
    last_seen = (now - timedelta(seconds=30)).isoformat()

    assert effective_worker_status(
        "online",
        last_seen,
        offline_after_seconds=180,
        now=now,
    ) == "online"


def test_effective_worker_status_preserves_explicit_non_online_state() -> None:
    assert effective_worker_status("maintenance", None) == "maintenance"


def test_parse_worker_timestamp_supports_sqlite_and_utc_forms() -> None:
    assert parse_worker_timestamp("2026-07-31 00:00:00") == datetime(
        2026,
        7,
        31,
        0,
        0,
        tzinfo=timezone.utc,
    )
    assert parse_worker_timestamp("2026-07-31T00:00:00Z") == datetime(
        2026,
        7,
        31,
        0,
        0,
        tzinfo=timezone.utc,
    )
