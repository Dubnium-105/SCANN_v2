from __future__ import annotations

from typing import List

from scann.services.query_models import QueryResult
from scann.services.query_utils import calculate_distance, dms_to_degrees, hms_to_degrees


class MpcClient:
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

            response = requests.get(
                "https://minorplanetcenter.net/api/mpc_ws",
                params={
                    "ra": ra_deg,
                    "dec": dec_deg,
                    "radius": radius_arcsec / 3600.0,
                    "format": "json",
                },
                timeout=self.timeout,
            )
            if response.status_code != 200:
                return []

            data = response.json()
            if not data or "results" not in data:
                return []

            results = []
            for item in data["results"]:
                name = item.get("name", "")
                number = item.get("number", "")
                full_name = f"{number} {name}" if number else name
                item_ra = hms_to_degrees(item.get("ra", "0:00:00"))
                item_dec = dms_to_degrees(item.get("dec", "+00:00:00"))
                distance = calculate_distance(ra_deg, dec_deg, item_ra, item_dec)
                object_type = "comet" if item.get("type", "asteroid") == "comet" else "asteroid"
                results.append(
                    QueryResult(
                        source="MPC",
                        name=full_name,
                        object_type=object_type,
                        distance_arcsec=distance,
                        magnitude=float(item.get("v", "0.0") or "0.0"),
                        url=f"https://minorplanetcenter.net/db_search/show_object?object_id={full_name}",
                        raw_data=item,
                    )
                )
            return results
        except Exception:
            return []