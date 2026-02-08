"""MPCORB 已知小行星处理模块

职责:
- 加载和解析 MPCORB 数据文件
- 计算指定时刻的小行星位置
- 按极限星等过滤
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from scann.core.models import ObservatoryConfig, SkyPosition


def _datetime_to_jd(dt: datetime) -> float:
    """将 datetime 对象转换为儒略日 (Julian Date)
    
    使用天文算法计算：
    - JD = 1721060 + 365 * (year - 1) + floor((year - 1) / 4) 
          - floor((year - 1) / 100) + floor((year - 1) / 400) 
          + day_of_year + hour/24 + minute/1440 + second/86400
    """
    # 转换为 UTC 时间戳
    timestamp = dt.timestamp()
    # Unix 时间戳 (1970-01-01 00:00:00 UTC) 对应的 JD
    JD_UNIX = 2440587.5
    # 加上天数 (时间戳是秒)
    return JD_UNIX + timestamp / 86400.0


@dataclass
class AsteroidOrbit:
    """小行星轨道要素"""
    designation: str        # 编号/名称
    epoch: float           # 历元 (JD)
    mean_anomaly: float    # 平近点角 (度)
    arg_perihelion: float  # 近日点幅角 (度)
    ascending_node: float  # 升交点经度 (度)
    inclination: float     # 轨道倾角 (度)
    eccentricity: float    # 离心率
    semi_major_axis: float # 半长轴 (AU)
    abs_magnitude: float   # 绝对星等 (H)
    slope_param: float     # 斜率参数 (G)


def load_mpcorb(path: Union[str, Path]) -> List[AsteroidOrbit]:
    """加载 MPCORB.DAT 文件

    Args:
        path: MPCORB 文件路径

    Returns:
        小行星轨道列表

    Raises:
        FileNotFoundError: 文件不存在
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MPCORB 文件不存在: {path}")

    asteroids = []
    with open(path, "r", encoding="ascii", errors="ignore") as f:
        header_passed = False
        for line in f:
            if not header_passed:
                # MPCORB 格式: 头部以 '------' 行结束
                if line.startswith("------"):
                    header_passed = True
                continue

            if len(line) < 160:
                continue

            try:
                asteroid = _parse_mpcorb_line(line)
                if asteroid is not None:
                    asteroids.append(asteroid)
            except (ValueError, IndexError):
                continue

    return asteroids


def _unpack_packed_epoch(packed: str) -> float:
    """解包 MPCORB packed epoch 格式为 Julian Date

    Packed epoch 格式: 5个字符
    - 字符0: 世纪前缀 (I=1800s, J=1900s, K=2000s)
    - 字符1-2: 年份 (00-99)
    - 字符3-4: 月日分数 (00-99 对应一年内的百分比)

    Args:
        packed: packed epoch 字符串，如 "K249A"

    Returns:
        Julian Date
    """
    if len(packed) != 5:
        raise ValueError(f"Packed epoch must be 5 characters, got {len(packed)}")

    # 解析世纪前缀
    century_char = packed[0].upper()
    if century_char == 'I':
        century = 1800
    elif century_char == 'J':
        century = 1900
    elif century_char == 'K':
        century = 2000
    else:
        raise ValueError(f"Unknown century prefix: {century_char}")

    # 解析年份
    try:
        year = int(packed[1:3])
    except ValueError:
        raise ValueError(f"Invalid year in packed epoch: {packed[1:3]}")

    # 解析月日分数 (00-99)
    # 这是一个编码值，需要转换为一年内的天数
    try:
        # 将字符转换为数字：0-9是直接的0-9，A=10, B=11, ..., Z=35
        def char_to_value(c):
            if '0' <= c <= '9':
                return int(c)
            elif 'A' <= c.upper() <= 'Z':
                return ord(c.upper()) - ord('A') + 10
            else:
                raise ValueError(f"Invalid character: {c}")

        day_fraction = (char_to_value(packed[3]) * 36 + char_to_value(packed[4])) / 1296.0
    except ValueError as e:
        raise ValueError(f"Invalid day fraction in packed epoch: {e}")

    # 计算Julian Date
    full_year = century + year
    # day_fraction 是一年的比例，转换为天数（约365.25）
    days_into_year = day_fraction * 365.25

    # 使用纯 Python 计算 JD
    # 首先计算该年1月1日的JD
    date = datetime(full_year, 1, 1)
    jd_start = _datetime_to_jd(date)
    # 加上天数
    return jd_start + days_into_year


def _parse_mpcorb_line(line: str) -> Optional[AsteroidOrbit]:
    """解析 MPCORB 单行数据 (MPC 格式)"""
    if len(line) < 160:
        return None

    try:
        designation = line[0:7].strip()
        abs_mag = float(line[8:13].strip())
        slope = float(line[14:19].strip()) if line[14:19].strip() else 0.15
        epoch_packed = line[20:25].strip()
        mean_anomaly = float(line[26:35].strip())
        arg_peri = float(line[37:46].strip())
        asc_node = float(line[48:57].strip())
        incl = float(line[59:68].strip())
        ecc = float(line[70:79].strip())
        # mean_daily_motion = float(line[80:91].strip())  # 不用
        semi_a = float(line[92:103].strip())

        # 解包 epoch
        epoch = _unpack_packed_epoch(epoch_packed)

        return AsteroidOrbit(
            designation=designation,
            epoch=epoch,
            mean_anomaly=mean_anomaly,
            arg_perihelion=arg_peri,
            ascending_node=asc_node,
            inclination=incl,
            eccentricity=ecc,
            semi_major_axis=semi_a,
            abs_magnitude=abs_mag,
            slope_param=slope,
        )
    except (ValueError, IndexError):
        return None


def filter_by_magnitude(
    asteroids: List[AsteroidOrbit],
    limit_mag: float,
) -> List[AsteroidOrbit]:
    """按极限星等过滤小行星

    Args:
        asteroids: 小行星列表
        limit_mag: 极限星等 (暗于此值的将被过滤掉)

    Returns:
        过滤后的列表
    """
    return [a for a in asteroids if a.abs_magnitude <= limit_mag]


def _solve_kepler_equation(M: float, e: float, tolerance: float = 1e-10) -> float:
    """求解开普勒方程 M = E - e*sin(E)

    使用牛顿-拉夫森迭代法求解偏近点角 E。

    Args:
        M: 平近点角（弧度）
        e: 偏心率
        tolerance: 收敛容差

    Returns:
        偏近点角 E（弧度）
    """
    import math

    # 初始猜测
    if e < 0.8:
        E = M  # 小偏心率
    else:
        E = math.pi  # 大偏心率

    # 牛顿-拉夫森迭代
    max_iter = 100
    for _ in range(max_iter):
        f = E - e * math.sin(E) - M
        f_prime = 1.0 - e * math.cos(E)

        delta = f / f_prime
        E -= delta

        if abs(delta) < tolerance:
            return E

    # 未收敛，返回当前值
    return E


def _compute_true_anomaly(E: float, e: float) -> float:
    """计算真近点角

    Args:
        E: 偏近点角（弧度）
        e: 偏心率

    Returns:
        真近点角（弧度）
    """
    import math

    sqrt_e = math.sqrt(1.0 - e * e)
    cos_E = math.cos(E)
    sin_E = math.sin(E)

    # tan^2(nu/2) = (1+e)/(1-e) * tan^2(E/2)
    tan_nu_2 = sqrt_e / (1.0 + e) * math.tan(E / 2.0)

    return 2.0 * math.atan(tan_nu_2)


def _orbital_to_equatorial(
    x: float,
    y: float,
    z: float,
    i: float,
    om: float,
    w: float,
) -> tuple:
    """轨道平面坐标转换到赤道坐标

    Args:
        x, y, z: 轨道平面坐标（AU）
        i: 轨道倾角（弧度）
        om: 升交点经度（弧度）
        w: 近日点幅角（弧度）

    Returns:
        (X, Y, Z) 赤道坐标（AU）
    """
    import math

    cos_om = math.cos(om)
    sin_om = math.sin(om)
    cos_w = math.cos(w)
    sin_w = math.sin(w)
    cos_i = math.cos(i)
    sin_i = math.sin(i)

    # 旋转矩阵组合
    P11 = cos_om * cos_w - sin_om * sin_w * cos_i
    P12 = -cos_om * sin_w - sin_om * cos_w * cos_i
    P13 = sin_om * sin_i

    P21 = sin_om * cos_w + cos_om * sin_w * cos_i
    P22 = -sin_om * sin_w + cos_om * cos_w * cos_i
    P23 = -cos_om * sin_i

    P31 = sin_w * sin_i
    P32 = cos_w * sin_i
    P33 = cos_i

    X = P11 * x + P12 * y + P13 * z
    Y = P21 * x + P22 * y + P23 * z
    Z = P31 * x + P32 * y + P33 * z

    return X, Y, Z


def _equatorial_to_spherical(X: float, Y: float, Z: float) -> tuple:
    """赤道坐标转换到球面坐标

    Args:
        X, Y, Z: 赤道坐标（AU）

    Returns:
        (ra, dec) 赤经和赤纬（弧度）
    """
    import math

    # 赤纬
    r = math.sqrt(X * X + Y * Y + Z * Z)
    dec = math.asin(Z / r) if r > 0 else 0.0

    # 赤经
    ra = math.atan2(Y, X)
    if ra < 0:
        ra += 2.0 * math.pi

    return ra, dec


def compute_apparent_positions(
    asteroids: List[AsteroidOrbit],
    obs_datetime: datetime,
    observatory: ObservatoryConfig,
) -> List[SkyPosition]:
    """计算小行星在指定时刻的视位置

    Args:
        asteroids: 小行星列表
        obs_datetime: 观测时刻
        observatory: 天文台参数（目前未使用，可扩展）

    Returns:
        天球坐标列表
    """
    import math

    positions = []

    # 计算观测时间的 JD
    obs_jd = _datetime_to_jd(obs_datetime)

    # 计算从 epoch 到观测的时间（天数）
    for asteroid in asteroids:
        if asteroid.epoch == 0:
            # 无效 epoch
            positions.append(SkyPosition(
                ra=0.0,
                dec=0.0,
                mag=asteroid.abs_magnitude,
                name=asteroid.designation,
            ))
            continue

        # 时间差（天）
        dt = obs_jd - asteroid.epoch

        # 轨道要素转换为弧度
        i = math.radians(asteroid.inclination)
        om = math.radians(asteroid.ascending_node)
        w = math.radians(asteroid.arg_perihelion)

        # 计算平近点角
        # n = sqrt(mu / a^3) 是平均运动
        # 这里简化：假设轨道要素在 epoch
        # 平近点角随时间变化
        n = 0.9856076686 / (asteroid.semi_major_axis ** 1.5)  # 度/天
        M0 = math.radians(asteroid.mean_anomaly)
        M = M0 + math.radians(n * dt)

        # 求解开普勒方程
        E = _solve_kepler_equation(M, asteroid.eccentricity)

        # 计算真近点角
        nu = _compute_true_anomaly(E, asteroid.eccentricity)

        # 计算距离（椭圆轨道）
        r = asteroid.semi_major_axis * (1.0 - asteroid.eccentricity ** 2) / (1.0 + asteroid.eccentricity * math.cos(nu))

        # 轨道平面坐标
        x_orb = r * math.cos(nu)
        y_orb = r * math.sin(nu)
        z_orb = 0.0

        # 转换到赤道坐标
        X, Y, Z = _orbital_to_equatorial(x_orb, y_orb, z_orb, i, om, w)

        # 转换到球面坐标
        ra_rad, dec_rad = _equatorial_to_spherical(X, Y, Z)

        # 转换为度
        ra_deg = math.degrees(ra_rad)
        dec_deg = math.degrees(dec_rad)

        positions.append(SkyPosition(
            ra=ra_deg % 360.0,  # 确保在 0-360 范围
            dec=dec_deg,
            mag=asteroid.abs_magnitude,
            name=asteroid.designation,
        ))

    return positions
