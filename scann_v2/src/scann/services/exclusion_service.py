"""Known-object exclusion service."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from scann.core.mpcorb import compute_apparent_positions
from scann.core.models import Candidate, FitsHeader, ObservatoryConfig, SkyPosition
from scann.services.siril_astrometry import (
    CandidateSkyCoordinateCache,
    ResolvedSkyCoordinate,
    SirilAstrometryResolver,
)

logger = logging.getLogger(__name__)


class ExclusionService:
    """Mark candidates that match known objects."""

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
        """Load MPCORB data and apply the configured magnitude cut."""
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
        """Resolve a candidate pixel position to sky coordinates."""
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
        """Return a cached coordinate if available."""
        return self._coordinate_cache.get(image_path, x, y)

    def _pixel_to_sky(
        self,
        header: FitsHeader,
        x: float,
        y: float,
        image_path: Optional[str | Path] = None,
    ) -> SkyPosition:
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
        """Fallback to header WCS-like information when Siril is unavailable."""
        crval1 = header.raw.get("CRVAL1")
        crval2 = header.raw.get("CRVAL2")
        crpix1 = header.raw.get("CRPIX1", 1.0)
        crpix2 = header.raw.get("CRPIX2", 1.0)
        cdelt1 = header.raw.get("CDELT1", 1.0 / 3600.0)
        cdelt2 = header.raw.get("CDELT2", 1.0 / 3600.0)

        if crval1 is None or crval2 is None:
            ra = header.ra or 0.0
            dec = header.dec or 0.0
            if header.ra is None or header.dec is None:
                raise ValueError("FITS header lacks usable sky coordinate information")
            return self._resolved_from_position(SkyPosition(ra=ra, dec=dec))

        ra = float(crval1) + (x - float(crpix1)) * float(cdelt1)
        dec = float(crval2) + (y - float(crpix2)) * float(cdelt2)

        return self._resolved_from_position(SkyPosition(ra=ra, dec=dec))

    def _resolved_from_position(self, position: SkyPosition) -> ResolvedSkyCoordinate:
        return ResolvedSkyCoordinate.from_decimal_degrees(position.ra, position.dec)

    def _calculate_angular_distance(
        self,
        pos1: SkyPosition,
        pos2: SkyPosition,
    ) -> float:
        """Return angular separation in arcseconds."""
        import math

        ra1 = math.radians(pos1.ra)
        dec1 = math.radians(pos1.dec)
        ra2 = math.radians(pos2.ra)
        dec2 = math.radians(pos2.dec)

        cos_distance = (
            math.sin(dec1) * math.sin(dec2)
            + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2)
        )
        cos_distance = max(-1.0, min(1.0, cos_distance))
        distance_rad = math.acos(cos_distance)
        return math.degrees(distance_rad) * 3600.0

    def _resolve_observation_datetime(self, header: FitsHeader) -> Optional[datetime]:
        """Use the observation midpoint when exposure time is known."""
        obs_datetime = header.observation_datetime
        if obs_datetime is None:
            return None

        exposure_time = header.exposure_time
        if exposure_time is None or exposure_time <= 0:
            return obs_datetime
        return obs_datetime + timedelta(seconds=float(exposure_time) / 2.0)

    def _build_known_objects(self, header: FitsHeader) -> list[dict[str, float | str]]:
        """Build propagated known-object positions, with a static fallback."""
        known_objects: list[dict[str, float | str]] = []
        propagated_ids: set[int] = set()

        obs_datetime = self._resolve_observation_datetime(header)
        if obs_datetime is not None:
            try:
                positions = compute_apparent_positions(
                    self._asteroids,
                    obs_datetime,
                    self.observatory,
                )
            except Exception:
                logger.exception("Failed to propagate MPCORB positions at observation time")
            else:
                for asteroid, position in zip(self._asteroids, positions):
                    known_objects.append(
                        {
                            "id": position.name or getattr(asteroid, "designation", ""),
                            "ra": float(position.ra),
                            "dec": float(position.dec),
                            "mag": float(position.mag or getattr(asteroid, "abs_magnitude", 0.0)),
                        }
                    )
                    propagated_ids.add(id(asteroid))

        for asteroid in self._asteroids:
            if id(asteroid) in propagated_ids:
                continue

            ra = getattr(asteroid, "ra", None)
            dec = getattr(asteroid, "dec", None)
            if ra is None or dec is None:
                continue

            known_objects.append(
                {
                    "id": getattr(asteroid, "designation", ""),
                    "ra": float(ra),
                    "dec": float(dec),
                    "mag": float(getattr(asteroid, "mag", getattr(asteroid, "abs_magnitude", 0.0))),
                }
            )

        return known_objects

    def check_candidates(
        self,
        candidates: list[Candidate],
        header: Optional[FitsHeader] = None,
        image_path: Optional[str | Path] = None,
    ) -> list[Candidate]:
        """Mark candidates that match a known object position."""
        if not self._asteroids or not header:
            return candidates

        known_objects = self._build_known_objects(header)
        if not known_objects:
            return candidates

        for candidate in candidates:
            sky_pos = self._pixel_to_sky(
                header,
                candidate.x,
                candidate.y,
                image_path=image_path,
            )

            for known in known_objects:
                known_pos = SkyPosition(ra=float(known["ra"]), dec=float(known["dec"]))
                distance = self._calculate_angular_distance(sky_pos, known_pos)
                if distance <= self.match_radius_arcsec:
                    candidate.is_known = True
                    candidate.known_id = str(known["id"])
                    break

        return candidates
