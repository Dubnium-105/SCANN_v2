from __future__ import annotations

from typing import List

from scann.services.query_models import QueryResult
from scann.services.query_utils import calculate_distance


class SimbadClient:
    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def query(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_arcsec: float = 10.0,
    ) -> List[QueryResult]:
        try:
            import astropy.units as u
            from astropy.coordinates import SkyCoord
            from astroquery.simbad import Simbad

            Simbad.reset_votable_fields()
            Simbad.add_votable_fields(
                "ra(d;ICRS;J2000)",
                "dec(d;ICRS;J2000)",
                "otype",
                "flux(V)",
                "coo_bibcode",
            )

            coord = SkyCoord(ra=ra_deg * u.degree, dec=dec_deg * u.degree, frame="icrs")
            result_table = Simbad.query_region(coord, radius=radius_arcsec * u.arcsec)
            if result_table is None:
                return []

            type_map = {
                "*": "Star",
                "Blue*": "Blue Straggler Star",
                "EB*": "Eclipsing Binary",
                "V*": "Variable Star",
                "Pulsar": "Pulsar",
                "G": "Galaxy",
                "GCl": "Globular Cluster",
                "HII": "HII Region",
                "PN": "Planetary Nebula",
                "SN": "Supernova",
                "SyG": "Seyfert Galaxy",
                "Neb": "Nebula",
            }
            results = []
            for row in result_table:
                name = row["MAIN_ID"].decode() if isinstance(row["MAIN_ID"], bytes) else row["MAIN_ID"]
                item_ra = float(row["RA_d_ICRS_J2000"])
                item_dec = float(row["DEC_d_ICRS_J2000"])
                distance = calculate_distance(ra_deg, dec_deg, item_ra, item_dec)
                obj_type = row["OTYPE"]
                if isinstance(obj_type, bytes):
                    obj_type = obj_type.decode()
                magnitude = 0.0
                mag_value = row["FLUX_V"]
                if mag_value is not None:
                    try:
                        magnitude = float(mag_value)
                    except (TypeError, ValueError):
                        magnitude = 0.0
                results.append(
                    QueryResult(
                        source="SIMBAD",
                        name=name,
                        object_type=type_map.get(obj_type, obj_type),
                        distance_arcsec=distance,
                        magnitude=magnitude,
                        url=f"http://simbad.u-strasbg.fr/simbad/sim-id?Ident={name}",
                        raw_data={"row": row},
                    )
                )
            return results
        except ImportError:
            return []
        except Exception:
            return []