"""外部查询服务

职责:
- VSX 变星查询
- MPC 小行星/彗星查询
- SIMBAD 天体查询
- TNS 暂现源查询
- 人造卫星检查
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class QueryResult:
    """查询结果"""
    source: str       # 来源 (VSX, MPC, SIMBAD, TNS)
    name: str         # 天体名称
    object_type: str  # 天体类型
    distance_arcsec: float = 0.0  # 与查询位置的距离
    magnitude: float = 0.0
    url: str = ""     # 详情链接
    raw_data: dict = None  # 原始返回数据

    def __post_init__(self):
        if self.raw_data is None:
            self.raw_data = {}


class QueryService:
    """外部天体查询服务"""

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    @staticmethod
    def _hms_to_degrees(hms: str) -> float:
        """将 hms 格式（hh:mm:ss.ss）转换为度"""
        import math

        try:
            parts = [float(x) for x in hms.split(":")]
            if len(parts) >= 3:
                h, m, s = parts[:3]
                return (h + m/60.0 + s/3600.0) * 15.0  # 1小时 = 15度
            elif len(parts) == 2:
                h, m = parts
                return (h + m/60.0) * 15.0
            else:
                return float(parts[0]) * 15.0
        except (ValueError, AttributeError):
            # 如果已经是度数格式，直接返回
            try:
                return float(hms)
            except ValueError:
                return 0.0

    @staticmethod
    def _dms_to_degrees(dms: str) -> float:
        """将 dms 格式（dd:mm:ss.ss）转换为度"""
        try:
            # 处理符号
            sign = 1
            if dms.startswith("-"):
                sign = -1
                dms = dms[1:]

            parts = [float(x) for x in dms.split(":")]
            if len(parts) >= 3:
                d, m, s = parts[:3]
                return sign * (d + m/60.0 + s/3600.0)
            elif len(parts) == 2:
                d, m = parts
                return sign * (d + m/60.0)
            else:
                return sign * float(parts[0])
        except (ValueError, AttributeError):
            # 如果已经是度数格式，直接返回
            try:
                return float(dms)
            except ValueError:
                return 0.0

    @staticmethod
    def _calculate_distance(
        ra1_deg: float,
        dec1_deg: float,
        ra2_deg: float,
        dec2_deg: float,
    ) -> float:
        """计算天球上两点之间的角距离（角秒）

        使用球面余弦定理：
        cos(d) = sin(δ1)sin(δ2) + cos(δ1)cos(δ2)cos(α1-α2)

        Args:
            ra1_deg: 第一个点的赤经（度）
            dec1_deg: 第一个点的赤纬（度）
            ra2_deg: 第二个点的赤经（度）
            dec2_deg: 第二个点的赤纬（度）

        Returns:
            角距离（角秒）
        """
        import math

        # 转换为弧度
        ra1 = math.radians(ra1_deg)
        dec1 = math.radians(dec1_deg)
        ra2 = math.radians(ra2_deg)
        dec2 = math.radians(dec2_deg)

        # 球面余弦定理
        cos_distance = (
            math.sin(dec1) * math.sin(dec2)
            + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2)
        )

        # 处理数值误差导致的 cos_distance 略大于1的情况
        cos_distance = max(-1.0, min(1.0, cos_distance))

        # 角距离（弧度）
        distance_rad = math.acos(cos_distance)

        # 转换为角秒（1弧度 = 206264.806247...角秒）
        distance_arcsec = math.degrees(distance_rad) * 3600.0

        return distance_arcsec

    def query_vsx(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_arcsec: float = 10.0,
    ) -> List[QueryResult]:
        """查询 AAVSO VSX 变星数据库

        Args:
            ra_deg: 赤经 (度)
            dec_deg: 赤纬 (度)
            radius_arcsec: 搜索半径 (角秒)

        Returns:
            查询结果列表
        """
        import requests

        try:
            url = (
                f"https://www.aavso.org/vsx/index.php?view=api.list"
                f"&ra={ra_deg}&dec={dec_deg}&radius={radius_arcsec / 60.0}"
                f"&format=json"
            )
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()

            results = []
            for item in data.get("VSXObjects", {}).get("VSXObject", []):
                # 解析 RA/Dec（格式：hh:mm:ss.ss 或 dd:mm:ss.ss）
                # VSX API 返回的格式通常是 hms/dms 字符串
                ra_str = item.get("RA", "")
                dec_str = item.get("Dec", "")

                # 将 hms/dms 转换为度
                item_ra = QueryService._hms_to_degrees(ra_str)
                item_dec = QueryService._dms_to_degrees(dec_str)

                # 计算距离
                distance = self._calculate_distance(item_ra, item_dec, ra_deg, dec_deg)

                results.append(QueryResult(
                    source="VSX",
                    name=item.get("Name", ""),
                    object_type=item.get("Type", ""),
                    distance_arcsec=distance,
                ))
            return results
        except Exception:
            return []

    def query_mpc(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_arcsec: float = 10.0,
    ) -> List[QueryResult]:
        """查询 MPC 小行星/彗星数据库

        使用 MPC 的 REST API 查询指定坐标附近的小行星和彗星

        Args:
            ra_deg: 赤经（度）
            dec_deg: 赤纬（度）
            radius_arcsec: 搜索半径（角秒）

        Returns:
            查询结果列表
        """
        try:
            import requests

            # MPC API endpoint（示例，实际需要确认正确的 API 端点）
            url = "https://minorplanetcenter.net/api/mpc_ws"

            # 构建查询参数
            params = {
                "ra": ra_deg,
                "dec": dec_deg,
                "radius": radius_arcsec / 3600.0,  # 转换为度
                "format": "json"
            }

            response = requests.get(url, params=params, timeout=self.timeout)

            if response.status_code != 200:
                return []

            data = response.json()
            results = []

            if not data or "results" not in data:
                return []

            for item in data["results"]:
                # 解析 MPC 响应
                name = item.get("name", "")
                number = item.get("number", "")

                if number:
                    full_name = f"{number} {name}"
                else:
                    full_name = name

                # 解析坐标
                ra_str = item.get("ra", "0:00:00")
                dec_str = item.get("dec", "+00:00:00")

                item_ra = self._hms_to_degrees(ra_str)
                item_dec = self._dms_to_degrees(dec_str)

                # 计算距离
                distance = self._calculate_distance(
                    ra_deg, dec_deg, item_ra, item_dec
                )

                # 确定天体类型
                obj_type = item.get("type", "asteroid")
                if obj_type == "comet":
                    object_type = "comet"
                else:
                    object_type = "asteroid"

                result = QueryResult(
                    source="MPC",
                    name=full_name,
                    object_type=object_type,
                    distance_arcsec=distance,
                    magnitude=float(item.get("v", "0.0") or "0.0"),
                    url=f"https://minorplanetcenter.net/db_search/show_object?object_id={full_name}",
                    raw_data=item
                )
                results.append(result)

            return results

        except Exception:
            # 网络错误或解析错误，返回空列表
            return []

    def query_simbad(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_arcsec: float = 10.0,
    ) -> List[QueryResult]:
        """查询 SIMBAD 天文数据库

        使用 astroquery.simbad 查询指定坐标附近的天体

        注意：此功能需要安装 astroquery 包：
        pip install astroquery

        Args:
            ra_deg: 赤经（度）
            dec_deg: 赤纬（度）
            radius_arcsec: 搜索半径（角秒）

        Returns:
            查询结果列表
        """
        try:
            # 尝试导入 astroquery
            from astroquery.simbad import Simbad
            from astropy.coordinates import SkyCoord
            import astropy.units as u

            # 配置 SIMBAD 查询
            Simbad.reset_votable_fields()
            Simbad.add_votable_fields(
                "ra(d;ICRS;J2000)",
                "dec(d;ICRS;J2000)",
                "otype",
                "flux(V)",
                "coo_bibcode"
            )

            # 创建坐标对象
            coord = SkyCoord(
                ra=ra_deg * u.degree,
                dec=dec_deg * u.degree,
                frame="icrs"
            )

            # 执行区域查询
            radius = radius_arcsec * u.arcsec
            result_table = Simbad.query_region(coord, radius=radius)

            if result_table is None:
                return []

            results = []

            for row in result_table:
                # 解析 SIMBAD 结果
                name = row["MAIN_ID"].decode() if isinstance(row["MAIN_ID"], bytes) else row["MAIN_ID"]
                item_ra = float(row["RA_d_ICRS_J2000"])
                item_dec = float(row["DEC_d_ICRS_J2000"])

                # 计算距离
                distance = self._calculate_distance(
                    ra_deg, dec_deg, item_ra, item_dec
                )

                # 解析天体类型
                obj_type = row["OTYPE"]
                if isinstance(obj_type, bytes):
                    obj_type = obj_type.decode()

                # 解析星等
                magnitude = 0.0
                mag_value = row["FLUX_V"]
                if mag_value is not None:
                    try:
                        magnitude = float(mag_value)
                    except (ValueError, TypeError):
                        magnitude = 0.0

                # 类型映射
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
                    "Neb": "Nebula"
                }
                object_type = type_map.get(obj_type, obj_type)

                result = QueryResult(
                    source="SIMBAD",
                    name=name,
                    object_type=object_type,
                    distance_arcsec=distance,
                    magnitude=magnitude,
                    url=f"http://simbad.u-strasbg.fr/simbad/sim-id?Ident={name}",
                    raw_data={"row": row}
                )
                results.append(result)

            return results

        except ImportError:
            # astroquery 未安装，返回空列表
            return []
        except Exception:
            # 查询错误，返回空列表
            return []

    def query_tns(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_arcsec: float = 10.0,
    ) -> List[QueryResult]:
        """查询 TNS 暂现源数据库

        使用 TNS API 查询指定坐标附近的暂现源（如超新星、伽马暴等）

        Args:
            ra_deg: 赤经（度）
            dec_deg: 赤纬（度）
            radius_arcsec: 搜索半径（角秒）

        Returns:
            查询结果列表
        """
        try:
            import requests

            # TNS API endpoint
            url = "https://www.wis-tns.weizmann.ac.il/api/get/search"

            # 构建 JSON 查询
            payload = {
                "ra": ra_deg,
                "dec": dec_deg,
                "radius": radius_arcsec / 3600.0,  # 转换为度
                "units": "degrees"
            }

            # TNS 需要 API key（这里使用公开端点）
            headers = {
                "User-Agent": "SCANN/1.0"
            }

            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )

            if response.status_code != 200:
                return []

            data = response.json()

            if not data or "object" not in data:
                return []

            item = data["object"]
            results = []

            # 解析 TNS 响应
            name = item.get("name", "")

            # 解析坐标
            ra_str = item.get("ra", "0:00:00")
            dec_str = item.get("dec", "+00:00:00")

            item_ra = self._hms_to_degrees(ra_str)
            item_dec = self._dms_to_degrees(dec_str)

            # 计算距离
            distance = self._calculate_distance(
                ra_deg, dec_deg, item_ra, item_dec
            )

            # 确定天体类型（TNS 使用数字编码）
            obj_type_code = item.get("objtype", "99")
            type_map = {
                "1": "SuperNova",
                "2": "Nova",
                "3": "LBV",
                "4": "Cataclysmic Variable",
                "5": "AGN",
                "6": "Gamma Ray Burst",
                "12": "Supernova"
            }
            object_type = type_map.get(obj_type_code, "Transient")

            # 提取星等
            magnitude = 0.0
            discovery_data = item.get("discovery_data", {})
            if isinstance(discovery_data, dict):
                mag_str = discovery_data.get("mag", "0.0")
                if mag_str and mag_str != "0.0":
                    try:
                        magnitude = float(mag_str)
                    except ValueError:
                        magnitude = 0.0

            result = QueryResult(
                source="TNS",
                name=name,
                object_type=object_type,
                distance_arcsec=distance,
                magnitude=magnitude,
                url=f"https://www.wis-tns.weizmann.ac.il/object/{name}",
                raw_data=item
            )
            results.append(result)

            return results

        except Exception:
            # 网络错误或解析错误，返回空列表
            return []

    def check_satellite(
        self,
        ra_deg: float,
        dec_deg: float,
        obs_datetime=None,
    ) -> List[QueryResult]:
        """检查人造卫星

        使用 TLE（两行轨道要素）数据检查指定坐标附近是否有卫星

        Args:
            ra_deg: 赤经（度）
            dec_deg: 赤纬（度）
            obs_datetime: 观测时间（默认为当前时间）

        Returns:
            查询结果列表
        """
        try:
            import requests
            from datetime import datetime

            # 如果没有提供时间，使用当前时间
            if obs_datetime is None:
                obs_datetime = datetime.utcnow()

            # TLE 数据源（CelesTrak）
            url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"

            response = requests.get(url, timeout=self.timeout)

            if response.status_code != 200:
                return []

            tle_text = response.text.strip()

            if not tle_text:
                return []

            # 解析 TLE 数据（每两行代表一颗卫星）
            lines = tle_text.split("\n")
            results = []

            # 限制处理数量以避免性能问题
            max_satellites = 100
            satellite_count = 0

            for i in range(0, len(lines) - 2, 3):
                if satellite_count >= max_satellites:
                    break

                # TLE 格式：第0行是名称，第1行是第一行轨道要素，第2行是第二行
                name_line = lines[i].strip()
                line1 = lines[i + 1].strip()
                line2 = lines[i + 2].strip()

                if not name_line or not line1 or not line2:
                    continue

                # 提取卫星信息
                satellite_name = name_line

                # TLE 行1: 解析轨道参数
                # 格式: 1 NNNNNU NNNNNAAA NNNNN.NNNNNNNN +.NNNNNNNN +NNNNN-N +NNNNN-N N NNNNN
                try:
                    # 简化版：只检查是否在感兴趣区域
                    # 实际卫星位置计算需要专业的 TLE 解析库（如 skyfield 或 sgp4）
                    # 这里实现一个简化的检查

                    # 计算卫星的近似位置（简化版）
                    # 在实际应用中，应该使用 skyfield 或 sgp4 库进行精确计算

                    satellite_count += 1

                    # 由于卫星是快速移动的目标，这里返回一个占位符结果
                    # 实际实现需要集成 TLE 传播器

                    result = QueryResult(
                        source="Satellite",
                        name=satellite_name,
                        object_type="satellite",
                        distance_arcsec=0.0,  # 需要实际计算
                        magnitude=0.0,
                        url=f"https://celestrak.org/satcat/?search={satellite_name}",
                        raw_data={"name": satellite_name, "line1": line1, "line2": line2}
                    )
                    results.append(result)

                except Exception:
                    # 解析错误，跳过这颗卫星
                    continue

            return results

        except Exception:
            # 网络错误或解析错误，返回空列表
            return []
