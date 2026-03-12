from __future__ import annotations

import warnings

from scann.services.query_models import QueryResponse, QueryResult
from scann.services.query_utils import calculate_distance


class SimbadClient:
    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def query(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_arcsec: float = 10.0,
    ) -> QueryResponse:
        try:
            import astropy.units as u
            from astropy.coordinates import SkyCoord
            from astroquery.exceptions import NoResultsWarning
            from astroquery.simbad import Simbad

            simbad = Simbad()
            simbad.add_votable_fields("otype", "V")

            coord = SkyCoord(ra=ra_deg * u.degree, dec=dec_deg * u.degree, frame="icrs")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", NoResultsWarning)
                result_table = simbad.query_region(coord, radius=radius_arcsec * u.arcsec)
            if result_table is None:
                return QueryResponse(results=[])

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
                name_value = row["main_id"]
                name = name_value.decode() if isinstance(name_value, bytes) else str(name_value)
                row_coord = SkyCoord(
                    ra=str(row["ra"]),
                    dec=str(row["dec"]),
                    unit=(u.hourangle, u.deg),
                    frame="icrs",
                )
                item_ra = row_coord.ra.degree
                item_dec = row_coord.dec.degree
                distance = calculate_distance(ra_deg, dec_deg, item_ra, item_dec)
                obj_type = row["otype"]
                if isinstance(obj_type, bytes):
                    obj_type = obj_type.decode()
                magnitude = 0.0
                mag_value = row["V"]
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
                        url=f"https://simbad.cds.unistra.fr/simbad/sim-id?Ident={name}",
                        raw_data={"row": row},
                    )
                )
            return QueryResponse(results=results)
        except ImportError:
            return QueryResponse(error="SIMBAD 查询依赖缺失: astroquery 或 astropy 未安装")
        except OSError as exc:
            return QueryResponse(error=f"SIMBAD 网络访问失败: {exc}")
        except Exception as exc:
            return QueryResponse(error=f"SIMBAD 查询异常: {exc}")