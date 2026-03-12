from __future__ import annotations

import re
from datetime import datetime

from scann.services.query_http import post_with_proxy_fallback
from scann.services.query_models import QueryResponse, QueryResult
from scann.services.query_utils import calculate_distance, dms_to_degrees, hms_to_degrees


class MpcClient:
    MPCHECKER_URL = "https://minorplanetcenter.net/cgi-bin/mpcheck.cgi"

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def query(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_arcsec: float = 10.0,
        obs_datetime: datetime | None = None,
    ) -> QueryResponse:
        import requests
        from astropy import units as u
        from astropy.coordinates import SkyCoord

        try:
            response = post_with_proxy_fallback(
                self.MPCHECKER_URL,
                data=self._build_form_data(ra_deg, dec_deg, radius_arcsec, obs_datetime),
                timeout=self.timeout,
            )
            if response.status_code != 200:
                return QueryResponse(error=f"MPC 请求失败: HTTP {response.status_code}")

            if "Error from WebCS Script" in response.text:
                return QueryResponse(error="MPC MPChecker 请求格式无效或服务拒绝处理")

            if self._is_no_result_response(response.text):
                return QueryResponse(results=[])

            pre_match = re.search(r"<pre>(?P<content>.*?)</pre>", response.text, flags=re.DOTALL | re.IGNORECASE)
            if pre_match is None:
                return QueryResponse(error="MPC MPChecker 响应缺少结果区块")

            results = []
            for parsed in self._parse_results_block(pre_match.group("content"), ra_deg, dec_deg, radius_arcsec):
                name = parsed["name"]
                coord = SkyCoord(
                    ra=parsed["ra_text"],
                    dec=parsed["dec_text"],
                    unit=(u.hourangle, u.deg),
                    frame="icrs",
                )
                item_ra = coord.ra.degree
                item_dec = coord.dec.degree
                distance = calculate_distance(ra_deg, dec_deg, item_ra, item_dec)
                if distance > radius_arcsec:
                    continue
                raw_line = parsed["raw_line"]
                object_type = "comet" if self._looks_like_comet(name) else "asteroid"
                results.append(
                    QueryResult(
                        source="MPC",
                        name=name,
                        object_type=object_type,
                        distance_arcsec=distance,
                        magnitude=parsed["magnitude"],
                        url=f"https://minorplanetcenter.net/db_search/show_object?object_id={name}",
                        raw_data={
                            "ra": parsed["ra_text"],
                            "dec": parsed["dec_text"],
                            "comment": parsed["comment"],
                            "line": raw_line,
                        },
                    )
                )
            return QueryResponse(results=results)
        except requests.RequestException as exc:
            return QueryResponse(error=f"MPC 请求失败: {exc}")
        except ValueError as exc:
            return QueryResponse(error=f"MPC 响应解析失败: {exc}")
        except Exception as exc:
            return QueryResponse(error=f"MPC 查询异常: {exc}")

    def _build_form_data(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_arcsec: float,
        obs_datetime: datetime | None,
    ) -> dict[str, str]:
        from astropy import units as u
        from astropy.coordinates import SkyCoord

        when = obs_datetime or datetime.utcnow()
        fractional_day = when.day + (
            when.hour / 24.0
            + when.minute / 1440.0
            + when.second / 86400.0
            + when.microsecond / 86400000000.0
        )
        coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
        query_radius_arcmin = max(5.0, min(300.0, radius_arcsec / 60.0))
        return {
            "year": f"{when.year:04d}",
            "month": f"{when.month:02d}",
            "day": f"{fractional_day:05.2f}",
            "which": "pos",
            "ra": coord.ra.to_string(unit=u.hour, sep=" ", precision=2, pad=True),
            "decl": coord.dec.to_string(unit=u.deg, sep=" ", precision=1, alwayssign=True, pad=True),
            "TextArea": "",
            "radius": f"{query_radius_arcmin:.1f}",
            "limit": "24.0",
            "oc": "500",
            "sort": "d",
            "mot": "h",
            "tmot": "s",
            "pdes": "u",
            "needed": "f",
            "ps": "n",
            "type": "p",
        }

    def _parse_results_block(
        self,
        content: str,
        ra_deg: float,
        dec_deg: float,
        radius_arcsec: float,
    ) -> list[dict[str, object]]:
        del ra_deg, dec_deg, radius_arcsec
        pattern = re.compile(
            r"^(?P<name>.+?)\s{2,}"
            r"(?P<ra>\d{2}\s+\d{2}\s+\d{1,2}(?:\.\d+)?)\s+"
            r"(?P<dec>[+-]\d{2}\s+\d{2}\s+\d{1,2})\s+"
            r"(?P<mag>\S+)\s+"
            r"(?P<rest>.+)$"
        )
        parsed_results: list[dict[str, object]] = []
        for raw_line in content.splitlines():
            line = raw_line.rstrip()
            if not line or line.lstrip().startswith("Object designation"):
                continue
            if line.lstrip().startswith("h  m  s"):
                continue
            match = pattern.match(line)
            if match is None:
                continue
            magnitude = 0.0
            mag_token = match.group("mag")
            try:
                magnitude = float(mag_token)
            except ValueError:
                magnitude = 0.0
            rest = match.group("rest")
            comment = rest.split("  ", 4)[-1].strip() if "  " in rest else rest.strip()
            parsed_results.append(
                {
                    "name": match.group("name").strip(),
                    "ra_text": match.group("ra").replace("  ", " ").strip(),
                    "dec_text": match.group("dec").replace("  ", " ").strip(),
                    "magnitude": magnitude,
                    "comment": comment,
                    "raw_line": line,
                }
            )
        return parsed_results

    @staticmethod
    def _looks_like_comet(name: str) -> bool:
        comet_prefixes = ("C/", "P/", "D/", "X/", "A/")
        return name.startswith(comet_prefixes)

    @staticmethod
    def _is_no_result_response(html: str) -> bool:
        normalized = " ".join(html.split())
        return "No known minor planets" in normalized