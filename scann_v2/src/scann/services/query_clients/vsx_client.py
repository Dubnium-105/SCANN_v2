from __future__ import annotations

from scann.services.query_http import get_with_proxy_fallback
from scann.services.query_models import QueryResponse, QueryResult
from scann.services.query_utils import calculate_distance, dms_to_degrees, hms_to_degrees


class VsxClient:
    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    @staticmethod
    def _clean_text(value: object) -> str:
        if value is None:
            return ""
        return " ".join(str(value).split())

    @classmethod
    def _first_non_empty(cls, item: dict[str, object], *keys: str) -> str:
        for key in keys:
            value = cls._clean_text(item.get(key))
            if value:
                return value
        return ""

    @classmethod
    def _parse_ra_deg(cls, item: dict[str, object]) -> float:
        ra_2000 = cls._first_non_empty(item, "RA2000")
        if ra_2000:
            try:
                return float(ra_2000)
            except ValueError:
                pass
        return hms_to_degrees(cls._first_non_empty(item, "RA"))

    @classmethod
    def _parse_dec_deg(cls, item: dict[str, object]) -> float:
        dec_2000 = cls._first_non_empty(item, "Declination2000")
        if dec_2000:
            try:
                return float(dec_2000)
            except ValueError:
                pass
        return dms_to_degrees(cls._first_non_empty(item, "Dec"))

    def query(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_arcsec: float = 10.0,
    ) -> QueryResponse:
        import requests

        try:
            url = (
                f"https://www.aavso.org/vsx/index.php?view=api.list"
                f"&ra={ra_deg}&dec={dec_deg}&radius={radius_arcsec / 60.0}"
                f"&format=json"
            )
            response = get_with_proxy_fallback(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("VSXObjects", {}).get("VSXObject", []):
                item_ra = self._parse_ra_deg(item)
                item_dec = self._parse_dec_deg(item)
                distance = calculate_distance(item_ra, item_dec, ra_deg, dec_deg)
                results.append(
                    QueryResult(
                        source="VSX",
                        name=self._first_non_empty(item, "Name"),
                        object_type=self._first_non_empty(
                            item,
                            "VariabilityType",
                            "Type",
                            "Category",
                        ),
                        distance_arcsec=distance,
                        raw_data=dict(item),
                    )
                )
            return QueryResponse(results=results)
        except requests.RequestException as exc:
            return QueryResponse(error=f"VSX 请求失败: {exc}")
        except ValueError as exc:
            return QueryResponse(error=f"VSX 响应解析失败: {exc}")
        except Exception as exc:
            return QueryResponse(error=f"VSX 查询异常: {exc}")