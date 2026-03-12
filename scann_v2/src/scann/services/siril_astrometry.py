"""Siril astrometry and PSF-based sky coordinate resolution."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol
from uuid import uuid4

from scann.core.models import SkyPosition


_RA_RE = re.compile(
    r"(?P<h>\d{1,2})h(?P<m>\d{1,2})m(?P<s>\d{1,2}(?:\.\d+)?)s",
    re.IGNORECASE,
)
_DEC_RE = re.compile(
    r"(?P<sign>[+-])(?P<d>\d{1,2})[°º](?P<m>\d{1,2})'(?P<s>\d{1,2}(?:\.\d+)?)(?:\"|”)",
    re.IGNORECASE,
)


class CoordinateResolver(Protocol):
    def resolve(self, image_path: str | Path, x: float, y: float) -> "ResolvedSkyCoordinate":
        ...


@dataclass(frozen=True)
class ResolvedSkyCoordinate:
    """Resolved sky coordinate plus display strings."""

    position: SkyPosition
    raw_coordinate: str
    normalized_coordinate: str

    @classmethod
    def from_hms_dms(cls, ra_hms: str, dec_dms: str) -> "ResolvedSkyCoordinate":
        ra_parts = _parse_ra_hms(ra_hms)
        dec_parts = _parse_dec_dms(dec_dms)
        position = SkyPosition(
            ra=_ra_hms_to_degrees(*ra_parts),
            dec=_dec_dms_to_degrees(*dec_parts),
        )
        raw = f"{_format_ra_hms_raw(*ra_parts)}{_format_dec_dms_raw(*dec_parts)}"
        normalized = f"{_format_ra_normalized(*ra_parts)} {_format_dec_normalized(*dec_parts)}"
        return cls(
            position=position,
            raw_coordinate=raw,
            normalized_coordinate=normalized,
        )

    @classmethod
    def from_decimal_degrees(cls, ra: float, dec: float) -> "ResolvedSkyCoordinate":
        ra_parts = _degrees_to_ra_hms(float(ra))
        dec_parts = _degrees_to_dec_dms(float(dec))
        return cls(
            position=SkyPosition(ra=float(ra), dec=float(dec)),
            raw_coordinate=f"{_format_ra_hms_raw(*ra_parts)}{_format_dec_dms_raw(*dec_parts)}",
            normalized_coordinate=(
                f"{_format_ra_normalized(*ra_parts)} {_format_dec_normalized(*dec_parts)}"
            ),
        )


def parse_psf_sky_coordinate(output: str) -> ResolvedSkyCoordinate:
    """Parse PSF console output into a resolved sky coordinate."""
    ra_match = _RA_RE.search(output)
    dec_match = _DEC_RE.search(output)
    if ra_match is None or dec_match is None:
        raise ValueError("PSF 输出中未找到天球坐标")

    ra_hms = ra_match.group(0)
    dec_dms = dec_match.group(0)
    return ResolvedSkyCoordinate.from_hms_dms(ra_hms=ra_hms, dec_dms=dec_dms)


def parse_findstar_catalog(
    catalog_text: str,
    target_x: float,
    target_y: float,
    selection_size: int | None = None,
) -> ResolvedSkyCoordinate:
    """Parse findstar tabular export and select the detection nearest the target pixel."""
    rows = _parse_findstar_rows(catalog_text)
    if not rows:
        raise ValueError("findstar 导出中未找到恒星数据")

    candidate_rows = rows
    if selection_size is not None:
        half_size = selection_size / 2.0
        in_selection = [
            row
            for row in rows
            if abs(row.x - float(target_x)) <= half_size and abs(row.y - float(target_y)) <= half_size
        ]
        if in_selection:
            candidate_rows = in_selection

    nearest = min(
        candidate_rows,
        key=lambda row: (row.x - float(target_x)) ** 2 + (row.y - float(target_y)) ** 2,
    )
    return ResolvedSkyCoordinate.from_decimal_degrees(nearest.ra, nearest.dec)


class CandidateSkyCoordinateCache:
    """Cache resolved candidate sky coordinates outside of ExclusionService internals."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, float, float], ResolvedSkyCoordinate] = {}

    def get(self, image_path: str | Path, x: float, y: float) -> ResolvedSkyCoordinate | None:
        return self._cache.get(self._make_key(image_path, x, y))

    def store(
        self,
        image_path: str | Path,
        x: float,
        y: float,
        resolved: ResolvedSkyCoordinate,
    ) -> ResolvedSkyCoordinate:
        self._cache[self._make_key(image_path, x, y)] = resolved
        return resolved

    def get_or_resolve(
        self,
        image_path: str | Path,
        x: float,
        y: float,
        resolver: CoordinateResolver,
    ) -> ResolvedSkyCoordinate:
        cached = self.get(image_path, x, y)
        if cached is not None:
            return cached
        resolved = resolver.resolve(image_path, x, y)
        return self.store(image_path, x, y, resolved)

    def _make_key(self, image_path: str | Path, x: float, y: float) -> tuple[str, float, float]:
        path = Path(image_path)
        return (str(path), round(float(x), 4), round(float(y), 4))


def default_run_siril_script(working_dir: Path, script: str) -> str:
    """Run a Siril script and return merged console output."""
    executable = _find_siril_executable()
    if executable is None:
        raise FileNotFoundError("未找到 Siril CLI (siril-cli/siril)")

    with tempfile.TemporaryDirectory(prefix="scann_siril_astrometry_") as td:
        script_path = Path(td) / "astrometry.ssf"
        script_path.write_text(script, encoding="ascii")
        process = subprocess.run(
            [executable, "-d", str(working_dir), "-s", str(script_path)],
            capture_output=True,
            text=False,
            timeout=180,
            check=False,
        )

    stdout = _safe_decode(process.stdout or b"")
    stderr = _safe_decode(process.stderr or b"")
    merged = "\n".join(part for part in (stdout, stderr) if part)
    if process.returncode != 0:
        raise RuntimeError(f"Siril 执行失败: rc={process.returncode}; output={merged[-800:]}")
    return merged


class SirilAstrometryResolver:
    """Resolve pixel coordinates to sky coordinates by platesolving then running PSF."""

    FOCAL_LENGTH_MM = 2042.6
    PIXEL_SIZE_UM = 18.0
    SELECTION_SIZE = 20

    def __init__(
        self,
        run_script_fn: Callable[[Path, str], str] = default_run_siril_script,
    ) -> None:
        self._run_script = run_script_fn

    def resolve(self, image_path: str | Path, x: float, y: float) -> ResolvedSkyCoordinate:
        path = Path(image_path)
        catalog_name = f"scann_findstar_{uuid4().hex}.txt"
        catalog_path = path.parent / catalog_name
        script = self._build_script(path, x, y, catalog_name)
        try:
            output = self._run_script(path.parent, script)
            if catalog_path.exists():
                catalog_text = catalog_path.read_text(encoding="utf-8", errors="replace")
                return parse_findstar_catalog(
                    catalog_text,
                    target_x=x,
                    target_y=y,
                    selection_size=self.SELECTION_SIZE,
                )
            return parse_psf_sky_coordinate(output)
        finally:
            if catalog_path.exists():
                catalog_path.unlink()

    def _build_script(self, image_path: Path, x: float, y: float, catalog_name: str) -> str:
        left = max(0, int(round(float(x))) - self.SELECTION_SIZE // 2)
        top = max(0, int(round(float(y))) - self.SELECTION_SIZE // 2)
        return "\n".join(
            [
                "requires 1.2.0",
                f'load "{image_path.name}"',
                f"platesolve -force -focal={self.FOCAL_LENGTH_MM} -pixelsize={self.PIXEL_SIZE_UM:.1f}",
                (
                    f"setfindstar reset -focal={self.FOCAL_LENGTH_MM} "
                    f"-pixelsize={self.PIXEL_SIZE_UM:.1f}"
                ),
                f"findstar -out={catalog_name}",
                f"boxselect {left} {top} {self.SELECTION_SIZE} {self.SELECTION_SIZE}",
                "psf",
                "close",
                "exit",
                "",
            ]
        )


def _find_siril_executable() -> str | None:
    for name in ("siril-cli", "siril-cli.exe", "siril", "siril.exe"):
        executable = shutil.which(name)
        if executable:
            return executable
    return None


def _safe_decode(data: bytes) -> str:
    for encoding in ("utf-8", "gbk", "mbcs"):
        try:
            return data.decode(encoding)
        except Exception:
            continue
    return data.decode("utf-8", errors="replace")


def _parse_ra_hms(value: str) -> tuple[int, int, float]:
    match = _RA_RE.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"无法解析赤经: {value}")
    return int(match.group("h")), int(match.group("m")), float(match.group("s"))


def _parse_dec_dms(value: str) -> tuple[str, int, int, float]:
    match = _DEC_RE.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"无法解析赤纬: {value}")
    return (
        match.group("sign"),
        int(match.group("d")),
        int(match.group("m")),
        float(match.group("s")),
    )


@dataclass(frozen=True)
class _FindStarRow:
    x: float
    y: float
    ra: float
    dec: float


def _parse_findstar_rows(catalog_text: str) -> list[_FindStarRow]:
    header: list[str] | None = None
    rows: list[_FindStarRow] = []
    for raw_line in catalog_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("# star#"):
            header = [value.strip() for value in line.lstrip("#").split("\t")]
            continue
        if line.startswith("#") or header is None:
            continue

        values = [value.strip() for value in raw_line.split("\t")]
        if len(values) != len(header):
            continue
        row = dict(zip(header, values))
        try:
            rows.append(
                _FindStarRow(
                    x=float(row["X"]),
                    y=float(row["Y"]),
                    ra=float(row["RA"]),
                    dec=float(row["Dec"]),
                )
            )
        except (KeyError, ValueError):
            continue
    return rows


def _ra_hms_to_degrees(hours: int, minutes: int, seconds: float) -> float:
    return (hours + minutes / 60.0 + seconds / 3600.0) * 15.0


def _dec_dms_to_degrees(sign: str, degrees: int, minutes: int, seconds: float) -> float:
    value = degrees + minutes / 60.0 + seconds / 3600.0
    return value if sign == "+" else -value


def _degrees_to_ra_hms(ra: float) -> tuple[int, int, float]:
    total_seconds = (ra / 15.0) * 3600.0
    hours, minutes, seconds = _split_sexagesimal(total_seconds, wrap_at=24)
    return hours, minutes, seconds


def _degrees_to_dec_dms(dec: float) -> tuple[str, int, int, float]:
    sign = "+" if dec >= 0 else "-"
    degrees, minutes, seconds = _split_sexagesimal(abs(dec) * 3600.0)
    return sign, degrees, minutes, seconds


def _split_sexagesimal(total_seconds: float, wrap_at: int | None = None) -> tuple[int, int, float]:
    major = int(total_seconds // 3600.0)
    remainder = total_seconds - major * 3600.0
    minutes = int(remainder // 60.0)
    seconds = round(remainder - minutes * 60.0, 2)

    if seconds >= 60.0:
        seconds = 0.0
        minutes += 1
    if minutes >= 60:
        minutes = 0
        major += 1
    if wrap_at is not None and major >= wrap_at:
        major %= wrap_at

    return major, minutes, seconds


def _format_ra_hms_raw(hours: int, minutes: int, seconds: float) -> str:
    return f"{hours:02d}h{minutes:02d}m{seconds:05.2f}s"


def _format_dec_dms_raw(sign: str, degrees: int, minutes: int, seconds: float) -> str:
    return f"{sign}{degrees:02d}°{minutes:02d}'{seconds:05.2f}\""


def _format_ra_normalized(hours: int, minutes: int, seconds: float) -> str:
    return f"{hours:02d} {minutes:02d} {seconds:05.2f}"


def _format_dec_normalized(sign: str, degrees: int, minutes: int, seconds: float) -> str:
    return f"{sign}{degrees:02d} {minutes:02d} {seconds:05.2f}"
