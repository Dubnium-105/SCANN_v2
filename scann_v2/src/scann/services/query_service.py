"""外部查询服务聚合层。"""

from __future__ import annotations

from typing import List

from scann.services.query_clients import (
    MpcClient,
    SatelliteClient,
    SimbadClient,
    TnsClient,
    VsxClient,
)
from scann.services.query_models import QueryResult
from scann.services.query_utils import calculate_distance, dms_to_degrees, hms_to_degrees


class QueryService:
    """外部天体查询服务。"""

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.vsx_client = VsxClient(timeout=timeout)
        self.mpc_client = MpcClient(timeout=timeout)
        self.simbad_client = SimbadClient(timeout=timeout)
        self.tns_client = TnsClient(timeout=timeout)
        self.satellite_client = SatelliteClient(timeout=timeout)

    @staticmethod
    def _hms_to_degrees(hms: str) -> float:
        """兼容旧测试入口，委托 query_utils。"""
        return hms_to_degrees(hms)

    @staticmethod
    def _dms_to_degrees(dms: str) -> float:
        """兼容旧测试入口，委托 query_utils。"""
        return dms_to_degrees(dms)

    @staticmethod
    def _calculate_distance(
        ra1_deg: float,
        dec1_deg: float,
        ra2_deg: float,
        dec2_deg: float,
    ) -> float:
        """兼容旧测试入口，委托 query_utils。"""
        return calculate_distance(ra1_deg, dec1_deg, ra2_deg, dec2_deg)

    def query_vsx(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_arcsec: float = 10.0,
    ) -> List[QueryResult]:
        """查询 AAVSO VSX 变星数据库。"""
        return self.vsx_client.query(ra_deg, dec_deg, radius_arcsec)

    def query_mpc(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_arcsec: float = 10.0,
    ) -> List[QueryResult]:
        """查询 MPC 小行星/彗星数据库。"""
        return self.mpc_client.query(ra_deg, dec_deg, radius_arcsec)

    def query_simbad(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_arcsec: float = 10.0,
    ) -> List[QueryResult]:
        """查询 SIMBAD 天文数据库。"""
        return self.simbad_client.query(ra_deg, dec_deg, radius_arcsec)

    def query_tns(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_arcsec: float = 10.0,
    ) -> List[QueryResult]:
        """查询 TNS 暂现源数据库。"""
        return self.tns_client.query(ra_deg, dec_deg, radius_arcsec)

    def check_satellite(
        self,
        ra_deg: float,
        dec_deg: float,
        obs_datetime=None,
    ) -> List[QueryResult]:
        """检查人造卫星。"""
        return self.satellite_client.check(ra_deg, dec_deg, obs_datetime)


__all__ = ["QueryResult", "QueryService"]
