from __future__ import annotations

from datetime import datetime
from typing import List

from scann.services.query_models import QueryResult


class SatelliteClient:
    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def check(self, ra_deg: float, dec_deg: float, obs_datetime=None) -> List[QueryResult]:
        del ra_deg, dec_deg
        try:
            import requests

            if obs_datetime is None:
                obs_datetime = datetime.utcnow()

            response = requests.get(
                "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle",
                timeout=self.timeout,
            )
            if response.status_code != 200:
                return []

            tle_text = response.text.strip()
            if not tle_text:
                return []

            lines = tle_text.split("\n")
            results = []
            satellite_count = 0
            for index in range(0, len(lines) - 2, 3):
                if satellite_count >= 100:
                    break
                name_line = lines[index].strip()
                line1 = lines[index + 1].strip()
                line2 = lines[index + 2].strip()
                if not name_line or not line1 or not line2:
                    continue
                try:
                    satellite_count += 1
                    results.append(
                        QueryResult(
                            source="Satellite",
                            name=name_line,
                            object_type="satellite",
                            distance_arcsec=0.0,
                            magnitude=0.0,
                            url=f"https://celestrak.org/satcat/?search={name_line}",
                            raw_data={
                                "name": name_line,
                                "line1": line1,
                                "line2": line2,
                                "obs_datetime": obs_datetime,
                            },
                        )
                    )
                except Exception:
                    continue
            return results
        except Exception:
            return []