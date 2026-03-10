from __future__ import annotations

import math


def hms_to_degrees(hms: str) -> float:
    """将 hms 格式（hh:mm:ss.ss）转换为度。"""
    try:
        parts = [float(x) for x in hms.split(":")]
        if len(parts) >= 3:
            hours, minutes, seconds = parts[:3]
            return (hours + minutes / 60.0 + seconds / 3600.0) * 15.0
        if len(parts) == 2:
            hours, minutes = parts
            return (hours + minutes / 60.0) * 15.0
        return float(parts[0]) * 15.0
    except (ValueError, AttributeError):
        try:
            return float(hms)
        except (TypeError, ValueError):
            return 0.0


def dms_to_degrees(dms: str) -> float:
    """将 dms 格式（dd:mm:ss.ss）转换为度。"""
    try:
        sign = 1
        if dms.startswith("-"):
            sign = -1
            dms = dms[1:]

        parts = [float(x) for x in dms.split(":")]
        if len(parts) >= 3:
            degrees, minutes, seconds = parts[:3]
            return sign * (degrees + minutes / 60.0 + seconds / 3600.0)
        if len(parts) == 2:
            degrees, minutes = parts
            return sign * (degrees + minutes / 60.0)
        return sign * float(parts[0])
    except (ValueError, AttributeError):
        try:
            return float(dms)
        except (TypeError, ValueError):
            return 0.0


def calculate_distance(
    ra1_deg: float,
    dec1_deg: float,
    ra2_deg: float,
    dec2_deg: float,
) -> float:
    """计算天球上两点之间的角距离，返回角秒。"""
    ra1 = math.radians(ra1_deg)
    dec1 = math.radians(dec1_deg)
    ra2 = math.radians(ra2_deg)
    dec2 = math.radians(dec2_deg)

    cos_distance = (
        math.sin(dec1) * math.sin(dec2)
        + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2)
    )
    cos_distance = max(-1.0, min(1.0, cos_distance))
    distance_rad = math.acos(cos_distance)
    return math.degrees(distance_rad) * 3600.0