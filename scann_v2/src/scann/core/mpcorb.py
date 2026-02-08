"""MPCORB 已知小行星处理模块

职责:
- 加载和解析 MPCORB 数据文件
- 计算指定时刻的小行星位置
- 按极限星等过滤
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from scann.core.models import ObservatoryConfig, SkyPosition


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

        return AsteroidOrbit(
            designation=designation,
            epoch=0.0,  # TODO: unpack epoch
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


def compute_apparent_positions(
    asteroids: List[AsteroidOrbit],
    obs_datetime: datetime,
    observatory: ObservatoryConfig,
) -> List[SkyPosition]:
    """计算小行星在指定时刻的视位置

    Args:
        asteroids: 小行星列表
        obs_datetime: 观测时刻
        observatory: 天文台参数

    Returns:
        天球坐标列表
    """
    # TODO: 实现轨道计算（使用 astropy 或自行实现开普勒方程求解）
    # 这里是占位实现
    positions = []
    for asteroid in asteroids:
        # 简化：返回空位置，实际需要完整的天体力学计算
        positions.append(SkyPosition(
            ra=0.0,
            dec=0.0,
            mag=asteroid.abs_magnitude,
            name=asteroid.designation,
        ))
    return positions
