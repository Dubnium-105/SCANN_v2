from __future__ import annotations

from datetime import datetime, timezone


DEFAULT_WORKER_OFFLINE_SECONDS = 3 * 60


def parse_worker_timestamp(value: str | None) -> datetime | None:
    normalized = str(value or "").strip()
    if not normalized:
        return None
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def effective_worker_status(
    status: str | None,
    last_seen_at: str | None,
    *,
    offline_after_seconds: int = DEFAULT_WORKER_OFFLINE_SECONDS,
    now: datetime | None = None,
) -> str:
    normalized_status = str(status or "offline").strip().lower() or "offline"
    if normalized_status != "online":
        return normalized_status

    last_seen = parse_worker_timestamp(last_seen_at)
    if last_seen is None:
        return "offline"

    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)
    timeout = max(1, int(offline_after_seconds))
    return "offline" if (current - last_seen).total_seconds() > timeout else "online"
