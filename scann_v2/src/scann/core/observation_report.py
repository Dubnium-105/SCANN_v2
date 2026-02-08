"""MPC 观测报告生成模块

职责:
- 生成 MPC 标准 80 列格式观测报告
- 格式化单行观测数据
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List

from scann.core.astrometry import format_dec_dms, format_ra_hms


@dataclass
class Observation:
    """单次观测数据"""
    designation: str      # 临时/永久编号
    discovery: bool       # 是否为发现观测
    obs_datetime: datetime
    ra_deg: float         # 赤经 (度)
    dec_deg: float        # 赤纬 (度)
    magnitude: float      # 视星等
    mag_band: str         # 星等波段 (V, R, C, etc.)
    observatory_code: str # MPC 天文台代码 (3字符)


def format_80col_line(obs: Observation) -> str:
    """格式化单条观测为 MPC 80 列格式

    MPC 80 列格式规范:
    - Col 1-12: 编号/临时编号
    - Col 13: 发现标记 ('*' = 发现)
    - Col 14: 注释代码
    - Col 15: 观测类型 ('C' = CCD)
    - Col 16-32: 观测日期时间 (YYYY MM DD.ddddd)
    - Col 33-44: 赤经 (HH MM SS.ss)
    - Col 45-56: 赤纬 (±DD MM SS.s)
    - Col 57-65: 空格
    - Col 66-70: 视星等
    - Col 71: 星等波段
    - Col 72-77: 空格
    - Col 78-80: 天文台代码

    Args:
        obs: 观测数据

    Returns:
        80 字符的格式化字符串
    """
    # 编号 (12 chars, 左对齐)
    designation = obs.designation.ljust(12)[:12]

    # 发现标记
    discovery = "*" if obs.discovery else " "

    # 观测类型
    note = " "
    obs_type = "C"  # CCD

    # 日期时间 (YYYY MM DD.ddddd) - 17 chars (col 16-32)
    dt = obs.obs_datetime
    day_fraction = (dt.hour * 3600 + dt.minute * 60 + dt.second) / 86400.0
    date_str = f"{dt.year:4d} {dt.month:02d} {dt.day + day_fraction:08.5f}"
    date_str = date_str.rjust(17)

    # 赤经赤纬 - 各 12 chars (col 33-44, 45-56)
    ra_str = format_ra_hms(obs.ra_deg).rjust(12)
    dec_str = format_dec_dms(obs.dec_deg).rjust(12)

    # 星等
    mag_str = f"{obs.magnitude:5.1f}" if obs.magnitude > 0 else "     "
    band = obs.mag_band if obs.mag_band else " "

    # 天文台代码
    code = obs.observatory_code.ljust(3)[:3]

    # 拼接 80 列
    line = (
        f"{designation}"      # 1-12
        f"{discovery}"        # 13
        f"{note}"             # 14
        f"{obs_type}"         # 15
        f"{date_str}"         # 16-32
        f"{ra_str}"           # 33-44
        f"{dec_str}"          # 45-56
        f"         "          # 57-65
        f"{mag_str}"          # 66-70
        f"{band}"             # 71
        f"      "             # 72-77
        f"{code}"             # 78-80
    )

    return line[:80]


def generate_mpc_report(
    observations: List[Observation],
    observatory_code: str = "",
) -> str:
    """生成完整的 MPC 观测报告

    Args:
        observations: 观测列表
        observatory_code: 默认天文台代码

    Returns:
        多行 80 列报告字符串
    """
    lines = []
    for obs in observations:
        if not obs.observatory_code and observatory_code:
            obs.observatory_code = observatory_code
        lines.append(format_80col_line(obs))
    return "\n".join(lines)
