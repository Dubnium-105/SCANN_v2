from __future__ import annotations

from typing import List

from scann.services.query_models import QueryResult
from scann.services.query_utils import calculate_distance, dms_to_degrees, hms_to_degrees


class VsxClient:
    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def query(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_arcsec: float = 10.0,
    ) -> List[QueryResult]:
        import requests

        try:
            url = (
                f"https://www.aavso.org/vsx/index.php?view=api.list"
                f"&ra={ra_deg}&dec={dec_deg}&radius={radius_arcsec / 60.0}"
                f"&format=json"
            )
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("VSXObjects", {}).get("VSXObject", []):
                item_ra = hms_to_degrees(item.get("RA", ""))
                item_dec = dms_to_degrees(item.get("Dec", ""))
                distance = calculate_distance(item_ra, item_dec, ra_deg, dec_deg)
                results.append(
                    QueryResult(
                        source="VSX",
                        name=item.get("Name", ""),
                        object_type=item.get("Type", ""),
                        distance_arcsec=distance,
                    )
                )
            return results
        except Exception:
            return []