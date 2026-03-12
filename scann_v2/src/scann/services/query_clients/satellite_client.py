from __future__ import annotations

from datetime import datetime, timezone
import html
import re

from bs4 import BeautifulSoup

from scann.core.models import ObservatoryConfig
from scann.services.query_http import post_with_proxy_fallback
from scann.services.query_models import QueryResponse, QueryResult

DEFAULT_SATELLITE_RADIUS_ARCSEC = 5.0 * 3600.0
PROJECT_PLUTO_SAT_ID_URL = "https://www.projectpluto.com/cgi-bin/sat_id/sat_id2"
PROJECT_PLUTO_SAT_ID_PAGE_URL = "https://www.projectpluto.com/sat_id2.htm"
DEFAULT_OBSERVATORY_CODE = "500"
DEFAULT_QUERY_DESIGNATION = "TLE01"
DEFAULT_QUERY_MAGNITUDE = 18.52


class SatelliteClient:
    def __init__(self, timeout: int = 10, max_results: int = 100):
        self.timeout = timeout
        self.max_results = max_results

    def check(
        self,
        ra_deg: float,
        dec_deg: float,
        obs_datetime=None,
        radius_arcsec: float = DEFAULT_SATELLITE_RADIUS_ARCSEC,
        observatory: ObservatoryConfig | None = None,
    ) -> QueryResponse:
        import requests

        try:
            obs_datetime = self._normalize_obs_datetime(obs_datetime)
            observation_line = self._build_sat_id_observation(
                ra_deg,
                dec_deg,
                obs_datetime,
                observatory=observatory,
            )
            response = post_with_proxy_fallback(
                PROJECT_PLUTO_SAT_ID_URL,
                timeout=self.timeout,
                data={
                    "TextArea": observation_line,
                    "radius": self._format_radius_degrees(radius_arcsec),
                },
            )
            if response.status_code != 200:
                return QueryResponse(error=f"卫星数据请求失败: HTTP {response.status_code}")

            results = self._parse_sat_id_response(
                response.text,
                obs_datetime,
                observation_line,
            )
            results = [
                result for result in results if result.distance_arcsec <= radius_arcsec
            ]
            results.sort(key=lambda item: item.distance_arcsec)
            return QueryResponse(results=results[: self.max_results])
        except requests.RequestException as exc:
            return QueryResponse(error=f"卫星数据请求失败: {exc}")
        except Exception as exc:
            return QueryResponse(error=f"卫星查询异常: {exc}")

    @staticmethod
    def _normalize_obs_datetime(obs_datetime: datetime | None) -> datetime:
        if obs_datetime is None:
            obs_datetime = datetime.utcnow()
        if obs_datetime.tzinfo is None:
            return obs_datetime.replace(tzinfo=timezone.utc)
        return obs_datetime.astimezone(timezone.utc)

    @staticmethod
    def _format_radius_degrees(radius_arcsec: float) -> str:
        radius_deg = max(radius_arcsec / 3600.0, 0.1)
        return f"{radius_deg:.3f}".rstrip("0").rstrip(".")

    def _build_sat_id_observation(
        self,
        ra_deg: float,
        dec_deg: float,
        obs_datetime: datetime,
        observatory: ObservatoryConfig | None,
    ) -> str:
        designation = f"     {DEFAULT_QUERY_DESIGNATION[:7].ljust(7)}"
        day_fraction = (
            obs_datetime.hour * 3600
            + obs_datetime.minute * 60
            + obs_datetime.second
            + (obs_datetime.microsecond / 1_000_000.0)
        ) / 86400.0
        date_str = f"{obs_datetime.year:4d} {obs_datetime.month:02d} {obs_datetime.day + day_fraction:08.5f}"
        ra_str = self._format_ra_for_sat_id(ra_deg)
        dec_str = self._format_dec_for_sat_id(dec_deg)
        observatory_code = self._resolve_observatory_code(observatory)
        return (
            f"{designation}"
            f" "
            f" "
            f"C"
            f"{date_str}"
            f" {ra_str}"
            f"{dec_str}"
            f"         "
            f"{DEFAULT_QUERY_MAGNITUDE:5.2f}"
            f"V"
            f"      "
            f"{observatory_code}"
        )[:80]

    @staticmethod
    def _resolve_observatory_code(observatory: ObservatoryConfig | None) -> str:
        if observatory is None:
            return DEFAULT_OBSERVATORY_CODE
        code = getattr(observatory, "code", "") or DEFAULT_OBSERVATORY_CODE
        return code.ljust(3)[:3]

    @staticmethod
    def _format_ra_for_sat_id(ra_deg: float) -> str:
        ra_hours = (ra_deg / 15.0) % 24.0
        hours = int(ra_hours)
        minutes_float = (ra_hours - hours) * 60.0
        minutes = int(minutes_float)
        seconds = (minutes_float - minutes) * 60.0
        return f"{hours:02d} {minutes:02d} {seconds:06.3f}"

    @staticmethod
    def _format_dec_for_sat_id(dec_deg: float) -> str:
        sign = "+" if dec_deg >= 0 else "-"
        dec_abs = abs(dec_deg)
        degrees = int(dec_abs)
        minutes_float = (dec_abs - degrees) * 60.0
        minutes = int(minutes_float)
        seconds = (minutes_float - minutes) * 60.0
        return f"{sign}{degrees:02d} {minutes:02d} {seconds:05.2f}"

    def _parse_sat_id_response(
        self,
        response_text: str,
        obs_datetime: datetime,
        observation_line: str,
    ) -> list[QueryResult]:
        pre_text = self._extract_preformatted_text(response_text)
        if "0 observations found" in pre_text or "0 objects" in pre_text:
            return []

        lines = [line.rstrip() for line in pre_text.splitlines() if line.strip()]
        results: list[QueryResult] = []
        index = 0
        while index < len(lines):
            line = lines[index]
            if line == observation_line and index + 1 < len(lines):
                parsed = self._parse_candidate_block(
                    lines[index + 1],
                    lines[index + 2 :],
                    obs_datetime,
                )
                if parsed is not None:
                    result, consumed_detail_lines = parsed
                    results.append(result)
                    index += consumed_detail_lines + 2
                    continue
            parsed = self._parse_candidate_block(
                line,
                lines[index + 1 :],
                obs_datetime,
            )
            if parsed is not None:
                result, consumed_detail_lines = parsed
                results.append(result)
                index += consumed_detail_lines + 1
                continue
            index += 1
        return results

    @staticmethod
    def _extract_preformatted_text(response_text: str) -> str:
        soup = BeautifulSoup(response_text, "html.parser")
        pre = soup.find("pre")
        if pre is None:
            return html.unescape(response_text)
        return pre.get_text("\n")

    def _parse_candidate_block(
        self,
        candidate_line: str,
        detail_lines: list[str],
        obs_datetime: datetime,
    ) -> tuple[QueryResult, int] | None:
        if ":" not in candidate_line or "=" not in candidate_line:
            return None

        prefix, name = candidate_line.rsplit(":", 1)
        norad_id, remainder = prefix.split("=", 1)
        norad_id = norad_id.strip()
        remainder = remainder.strip()
        name = name.strip()
        if not norad_id or not name:
            return None

        international_designator, _, elements_summary = remainder.partition("   ")
        consumed_detail_lines = 0
        details: list[str] = []
        for line in detail_lines:
            if line.startswith("             "):
                details.append(line.strip())
                consumed_detail_lines += 1
                continue
            break

        joined_details = " ".join(details)
        offset_match = re.search(r"offset=\s*([0-9.]+)\s*deg", joined_details)
        distance_match = re.search(r"dist=\s*([0-9.]+)\s*km", joined_details)
        motion_match = re.search(r'motion\s+([0-9.]+)"/sec at PA\s+([0-9.]+)', joined_details)

        offset_arcsec = float(offset_match.group(1)) * 3600.0 if offset_match else 0.0
        raw_data = {
            "norad_id": norad_id,
            "international_designator": international_designator.strip(),
            "elements_summary": elements_summary.strip(),
            "details": details,
            "obs_datetime": obs_datetime,
        }
        if distance_match:
            raw_data["distance_km"] = float(distance_match.group(1))
        if motion_match:
            raw_data["motion_arcsec_per_sec"] = float(motion_match.group(1))
            raw_data["motion_pa_deg"] = float(motion_match.group(2))

        return (
            QueryResult(
                source="Satellite",
                name=name,
                object_type="satellite",
                distance_arcsec=offset_arcsec,
                magnitude=0.0,
                url=PROJECT_PLUTO_SAT_ID_PAGE_URL,
                raw_data=raw_data,
            ),
            consumed_detail_lines,
        )