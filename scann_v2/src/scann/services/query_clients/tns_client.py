from __future__ import annotations

from scann.services.query_http import post_with_proxy_fallback
from scann.services.query_models import QueryResponse, QueryResult
from scann.services.query_utils import calculate_distance, dms_to_degrees, hms_to_degrees


class TnsClient:
    LEGACY_SEARCH_URL = "https://www.wis-tns.weizmann.ac.il/api/get/search"
    AUTH_SEARCH_URL = "https://www.wis-tns.org/api/get/search"

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def query(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_arcsec: float = 10.0,
    ) -> QueryResponse:
        import requests

        try:
            response = post_with_proxy_fallback(
                self.LEGACY_SEARCH_URL,
                json={
                    "ra": ra_deg,
                    "dec": dec_deg,
                    "radius": radius_arcsec / 3600.0,
                    "units": "degrees",
                },
                headers={"User-Agent": "SCANN/1.0"},
                timeout=self.timeout,
            )
            if response.status_code != 200:
                return QueryResponse(error=f"TNS 请求失败: HTTP {response.status_code}")

            data = response.json()
            if not data or "object" not in data:
                return QueryResponse(results=[])

            item = data["object"]
            item_ra = hms_to_degrees(item.get("ra", "0:00:00"))
            item_dec = dms_to_degrees(item.get("dec", "+00:00:00"))
            distance = calculate_distance(ra_deg, dec_deg, item_ra, item_dec)
            object_type = {
                "1": "SuperNova",
                "2": "Nova",
                "3": "LBV",
                "4": "Cataclysmic Variable",
                "5": "AGN",
                "6": "Gamma Ray Burst",
                "12": "Supernova",
            }.get(item.get("objtype", "99"), "Transient")
            magnitude = 0.0
            discovery_data = item.get("discovery_data", {})
            if isinstance(discovery_data, dict):
                mag_str = discovery_data.get("mag", "0.0")
                if mag_str and mag_str != "0.0":
                    try:
                        magnitude = float(mag_str)
                    except ValueError:
                        magnitude = 0.0
            return QueryResponse(
                results=[
                    QueryResult(
                        source="TNS",
                        name=item.get("name", ""),
                        object_type=object_type,
                        distance_arcsec=distance,
                        magnitude=magnitude,
                        url=f"https://www.wis-tns.weizmann.ac.il/object/{item.get('name', '')}",
                        raw_data=item,
                    )
                ]
            )
        except requests.RequestException as exc:
            auth_response = self._probe_authenticated_endpoint()
            if auth_response is not None:
                return auth_response
            return QueryResponse(error=f"TNS 请求失败: 旧公开端点不可用，且当前网络无法完成直连访问: {exc}")
        except ValueError as exc:
            return QueryResponse(error=f"TNS 响应解析失败: {exc}")
        except Exception as exc:
            return QueryResponse(error=f"TNS 查询异常: {exc}")

    def _probe_authenticated_endpoint(self) -> QueryResponse | None:
        import requests

        try:
            response = post_with_proxy_fallback(
                self.AUTH_SEARCH_URL,
                timeout=min(self.timeout, 10),
                json={},
                headers={"User-Agent": "SCANN/1.0"},
            )
        except requests.RequestException:
            return None

        if response.status_code == 401:
            return QueryResponse(
                error=(
                    "TNS 查询失败: 新版 TNS API 需要认证，旧公开端点当前不可用。"
                    "如需继续接入，需要提供 TNS API 凭据或改为网页抓取方案。"
                )
            )
        return QueryResponse(error=f"TNS 请求失败: HTTP {response.status_code}")