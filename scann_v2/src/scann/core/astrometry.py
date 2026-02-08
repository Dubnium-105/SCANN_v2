"""天文坐标转换模块

职责:
- 像素坐标 ↔ WCS 天球坐标
- 依赖 FITS Header 中的 WCS 信息
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from scann.core.models import FitsHeader, SkyPosition


def pixel_to_wcs(
    x: float,
    y: float,
    header: FitsHeader,
) -> Optional[SkyPosition]:
    """像素坐标转天球坐标

    Args:
        x: 像素 X 坐标
        y: 像素 Y 坐标
        header: FITS 文件头 (需包含 WCS 关键字)

    Returns:
        天球坐标 (RA, Dec), 无 WCS 信息则返回 None
    """
    from astropy.wcs import WCS

    try:
        wcs = WCS(header.raw)
        result = wcs.pixel_to_world(x, y)
        return SkyPosition(
            ra=float(result.ra.deg),
            dec=float(result.dec.deg),
        )
    except Exception:
        return None


def wcs_to_pixel(
    ra: float,
    dec: float,
    header: FitsHeader,
) -> Optional[Tuple[float, float]]:
    """天球坐标转像素坐标

    Args:
        ra: 赤经 (度)
        dec: 赤纬 (度)
        header: FITS 文件头

    Returns:
        (x, y) 像素坐标, 无 WCS 信息则返回 None
    """
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    try:
        wcs = WCS(header.raw)
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        px, py = wcs.world_to_pixel(coord)
        return (float(px), float(py))
    except Exception:
        return None


def format_ra_hms(ra_deg: float) -> str:
    """赤经 (度) 格式化为 HH MM SS.ss"""
    ra_h = ra_deg / 15.0
    h = int(ra_h)
    m = int((ra_h - h) * 60)
    s = (ra_h - h - m / 60.0) * 3600
    return f"{h:02d} {m:02d} {s:05.2f}"


def format_dec_dms(dec_deg: float) -> str:
    """赤纬 (度) 格式化为 ±DD MM SS.s"""
    sign = "+" if dec_deg >= 0 else "-"
    dec_abs = abs(dec_deg)
    d = int(dec_abs)
    m = int((dec_abs - d) * 60)
    s = (dec_abs - d - m / 60.0) * 3600
    return f"{sign}{d:02d} {m:02d} {s:04.1f}"
