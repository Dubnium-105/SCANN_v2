from __future__ import annotations

from typing import List

from scann.services.query_models import QueryResult
from scann.services.query_utils import calculate_distance, dms_to_degrees, hms_to_degrees


class TnsClient:
    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def query(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_arcsec: float = 10.0,
    ) -> List[QueryResult]:
        try:
            import requests

            response = requests.post(
                "https://www.wis-tns.weizmann.ac.il/api/get/search",
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
                return []

            data = response.json()
            if not data or "object" not in data:
                return []

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
            return [
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
        except Exception:
            return []