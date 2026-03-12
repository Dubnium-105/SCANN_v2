"""已知天体排除服务

职责:
- 综合 MPCORB + 外部查询排除已知天体
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from scann.core.models import Candidate, FitsHeader, ObservatoryConfig, SkyPosition
from scann.services.siril_astrometry import (
    CandidateSkyCoordinateCache,
    ResolvedSkyCoordinate,
    SirilAstrometryResolver,
)


class ExclusionService:
    """已知天体排除"""

    # 默认匹配半径（角秒）
    DEFAULT_MATCH_RADIUS_ARCSEC = 5.0

    def __init__(
        self,
        mpcorb_path: Optional[str] = None,
        observatory: Optional[ObservatoryConfig] = None,
        limit_magnitude: float = 20.0,
        match_radius_arcsec: float = DEFAULT_MATCH_RADIUS_ARCSEC,
        coordinate_resolver=None,
        coordinate_cache: Optional[CandidateSkyCoordinateCache] = None,
    ):
        self.mpcorb_path = mpcorb_path
        self.observatory = observatory or ObservatoryConfig()
        self.limit_magnitude = limit_magnitude
        self.match_radius_arcsec = match_radius_arcsec
        self._asteroids = None
        self._coordinate_resolver = coordinate_resolver or SirilAstrometryResolver()
        self._coordinate_cache = coordinate_cache or CandidateSkyCoordinateCache()

    def load_mpcorb(self) -> int:
        """加载 MPCORB 数据

        Returns:
            加载的小行星数量
        """
        if not self.mpcorb_path:
            return 0

        from scann.core.mpcorb import filter_by_magnitude, load_mpcorb

        all_asteroids = load_mpcorb(self.mpcorb_path)
        self._asteroids = filter_by_magnitude(all_asteroids, self.limit_magnitude)
        return len(self._asteroids)

    def get_candidate_sky_coordinate(
        self,
        header: FitsHeader,
        x: float,
        y: float,
        image_path: Optional[str | Path] = None,
    ) -> ResolvedSkyCoordinate:
        """获取候选体天球坐标，并在服务层缓存结果。"""
        if image_path:
            try:
                return self._coordinate_cache.get_or_resolve(
                    image_path,
                    x,
                    y,
                    self._coordinate_resolver,
                )
            except Exception:
                pass

        return self._fallback_coordinate(header, x, y)

    def get_cached_candidate_sky_coordinate(
        self,
        image_path: str | Path,
        x: float,
        y: float,
    ) -> Optional[ResolvedSkyCoordinate]:
        """返回当前已缓存的候选体天球坐标。"""
        return self._coordinate_cache.get(image_path, x, y)

    def _pixel_to_sky(
        self,
        header: FitsHeader,
        x: float,
        y: float,
        image_path: Optional[str | Path] = None,
    ) -> SkyPosition:
        """将像素坐标转换为天球坐标。"""
        return self.get_candidate_sky_coordinate(
            header,
            x,
            y,
            image_path=image_path,
        ).position

    def _fallback_coordinate(
        self,
        header: FitsHeader,
        x: float,
        y: float,
    ) -> ResolvedSkyCoordinate:
        """当 Siril 无法解析时，退回到 FITS header 的线性 WCS。"""
        crval1 = header.raw.get("CRVAL1")
        crval2 = header.raw.get("CRVAL2")
        crpix1 = header.raw.get("CRPIX1", 1.0)
        crpix2 = header.raw.get("CRPIX2", 1.0)
        cdelt1 = header.raw.get("CDELT1", 1.0/3600.0)
        cdelt2 = header.raw.get("CDELT2", 1.0/3600.0)

        if crval1 is None or crval2 is None:
            ra = header.ra or 0.0
            dec = header.dec or 0.0
            if header.ra is None or header.dec is None:
                raise ValueError("FITS 头缺少可用的天球坐标信息")
            return self._resolved_from_position(SkyPosition(ra=ra, dec=dec))

        ra = float(crval1) + (x - float(crpix1)) * float(cdelt1)
        dec = float(crval2) + (y - float(crpix2)) * float(cdelt2)

        return self._resolved_from_position(SkyPosition(ra=ra, dec=dec))

    def _resolved_from_position(self, position: SkyPosition) -> ResolvedSkyCoordinate:
        ra_hms, dec_dms = self._to_hms_dms(position)
        return ResolvedSkyCoordinate(
            position=position,
            raw_coordinate=f"{ra_hms}{dec_dms}",
            normalized_coordinate=self._normalize_hms_dms(ra_hms, dec_dms),
        )

    def _to_hms_dms(self, position: SkyPosition) -> tuple[str, str]:
        ra_hours = position.ra / 15.0
        ra_h = int(ra_hours)
        ra_m_float = (ra_hours - ra_h) * 60.0
        ra_m = int(ra_m_float)
        ra_s = (ra_m_float - ra_m) * 60.0

        dec_sign = "+" if position.dec >= 0 else "-"
        dec_abs = abs(position.dec)
        dec_d = int(dec_abs)
        dec_m_float = (dec_abs - dec_d) * 60.0
        dec_m = int(dec_m_float)
        dec_s = (dec_m_float - dec_m) * 60.0

        return (
            f"{ra_h:02d}h{ra_m:02d}m{ra_s:05.2f}s",
            f"{dec_sign}{dec_d:02d}°{dec_m:02d}'{dec_s:05.2f}\"",
        )

    def _normalize_hms_dms(self, ra_hms: str, dec_dms: str) -> str:
        resolved = ResolvedSkyCoordinate.from_hms_dms(ra_hms=ra_hms, dec_dms=dec_dms)
        return resolved.normalized_coordinate

    def _calculate_angular_distance(
        self,
        pos1: SkyPosition,
        pos2: SkyPosition,
    ) -> float:
        """计算两个天球坐标之间的角距离（角秒）

        Args:
            pos1: 第一个坐标
            pos2: 第二个坐标

        Returns:
            角距离（角秒）
        """
        import math

        # 转换为弧度
        ra1 = math.radians(pos1.ra)
        dec1 = math.radians(pos1.dec)
        ra2 = math.radians(pos2.ra)
        dec2 = math.radians(pos2.dec)

        # 球面余弦定理
        cos_distance = (
            math.sin(dec1) * math.sin(dec2)
            + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2)
        )

        # 处理数值误差
        cos_distance = max(-1.0, min(1.0, cos_distance))

        # 角距离（弧度）
        distance_rad = math.acos(cos_distance)

        # 转换为角秒
        return math.degrees(distance_rad) * 3600.0

    def check_candidates(
        self,
        candidates: List[Candidate],
        header: Optional[FitsHeader] = None,
        image_path: Optional[str | Path] = None,
    ) -> List[Candidate]:
        """检查候选体是否为已知天体

        Args:
            candidates: 候选体列表
            header: FITS 头 (用于坐标转换和时间)

        Returns:
            更新后的候选体列表 (已知天体被标记)
        """
        if not self._asteroids or not header:
            return candidates

        # 准备已知天体列表
        # 注意：真实场景中需要计算小行星在观测时刻的位置
        # 这里简化为使用小行星的 epoch 位置
        known_objects = []
        for asteroid in self._asteroids:
            # 检查小行星是否有 ra/dec 属性
            # 如果没有，说明需要实现轨道计算（TODO）
            if hasattr(asteroid, 'ra') and hasattr(asteroid, 'dec'):
                known_objects.append({
                    'id': asteroid.designation,
                    'ra': asteroid.ra,
                    'dec': asteroid.dec,
                    'mag': asteroid.mag if hasattr(asteroid, 'mag') else 0.0
                })

        # 检查每个候选体
        for candidate in candidates:
            sky_pos = self._pixel_to_sky(
                header,
                candidate.x,
                candidate.y,
                image_path=image_path,
            )

            # 查找匹配的已知天体
            for known in known_objects:
                known_pos = SkyPosition(ra=known['ra'], dec=known['dec'])
                distance = self._calculate_angular_distance(sky_pos, known_pos)

                if distance <= self.match_radius_arcsec:
                    candidate.is_known = True
                    candidate.known_id = known['id']
                    break

        return candidates
