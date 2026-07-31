"""Replayable sky/time association with an explicit WCS quality gate."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping


OBJECT_ASSOCIATION_VERSION = "sky-time-association-v1"


def angular_separation_arcsec(
    ra1_deg: float,
    dec1_deg: float,
    ra2_deg: float,
    dec2_deg: float,
) -> float:
    ra1 = math.radians(float(ra1_deg))
    dec1 = math.radians(float(dec1_deg))
    ra2 = math.radians(float(ra2_deg))
    dec2 = math.radians(float(dec2_deg))
    cosine = (
        math.sin(dec1) * math.sin(dec2)
        + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2)
    )
    angle = math.acos(max(-1.0, min(1.0, cosine)))
    return math.degrees(angle) * 3600.0


def _parse_time(value: Any) -> datetime:
    normalized = str(value or "").strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class Association:
    left_id: str
    right_id: str
    separation_arcsec: float
    time_delta_seconds: float
    version: str = OBJECT_ASSOCIATION_VERSION


def associate_observations(
    observations: Iterable[Mapping[str, Any]],
    *,
    maximum_separation_arcsec: float,
    maximum_time_delta_seconds: float,
    wcs_valid_fraction: float,
    minimum_wcs_valid_fraction: float = 0.95,
) -> dict[str, Any]:
    if float(wcs_valid_fraction) < float(minimum_wcs_valid_fraction):
        return {
            "version": OBJECT_ASSOCIATION_VERSION,
            "enabled": False,
            "reason": "wcs_quality_gate_failed",
            "wcs_valid_fraction": float(wcs_valid_fraction),
            "minimum_wcs_valid_fraction": float(minimum_wcs_valid_fraction),
            "associations": [],
        }
    items = [dict(item) for item in observations]
    associations: list[Association] = []
    for left_index, left in enumerate(items):
        for right in items[left_index + 1 :]:
            time_delta = abs(
                (
                    _parse_time(left.get("observed_at"))
                    - _parse_time(right.get("observed_at"))
                ).total_seconds()
            )
            if time_delta > float(maximum_time_delta_seconds):
                continue
            separation = angular_separation_arcsec(
                float(left["ra_deg"]),
                float(left["dec_deg"]),
                float(right["ra_deg"]),
                float(right["dec_deg"]),
            )
            if separation > float(maximum_separation_arcsec):
                continue
            associations.append(
                Association(
                    left_id=str(left.get("observation_id") or ""),
                    right_id=str(right.get("observation_id") or ""),
                    separation_arcsec=separation,
                    time_delta_seconds=time_delta,
                )
            )
    return {
        "version": OBJECT_ASSOCIATION_VERSION,
        "enabled": True,
        "wcs_valid_fraction": float(wcs_valid_fraction),
        "maximum_separation_arcsec": float(maximum_separation_arcsec),
        "maximum_time_delta_seconds": float(maximum_time_delta_seconds),
        "associations": [asdict(item) for item in associations],
    }
