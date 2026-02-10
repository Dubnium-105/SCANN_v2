"""配置管理模块

职责:
- 加载/保存应用配置 (JSON)
- 提供默认值
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

from scann.core.models import (
    AppConfig,
    BitDepth,
    ObservatoryConfig,
    TelescopeConfig,
)

DEFAULT_CONFIG_FILENAME = "scann_v2_config.json"


def get_default_config_path() -> Path:
    """获取默认配置文件路径 (与脚本同目录)"""
    import sys
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).parent
    else:
        base = Path(__file__).resolve().parent.parent.parent.parent
    return base / DEFAULT_CONFIG_FILENAME


def load_config(
    path: Optional[Union[str, Path]] = None,
) -> AppConfig:
    """加载配置文件

    Args:
        path: 配置文件路径 (None=默认位置)

    Returns:
        AppConfig 实例
    """
    if path is None:
        path = get_default_config_path()
    path = Path(path)

    config = AppConfig()

    if not path.exists():
        return config

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return config

    # 映射 JSON -> AppConfig
    config.new_folder = data.get("new_folder", "")
    config.old_folder = data.get("old_folder", "")
    config.save_folder = data.get("save_folder", "")

    # 望远镜参数
    tel = data.get("telescope", {})
    config.telescope = TelescopeConfig(
        pixel_size_um=tel.get("pixel_size_um", 9.0),
        pixel_scale_arcsec=tel.get("pixel_scale_arcsec", 0.0),
        focal_length_mm=tel.get("focal_length_mm", 0.0),
        camera_rotation_deg=tel.get("camera_rotation_deg", 0.0),
    )

    # 天文台参数
    obs = data.get("observatory", {})
    config.observatory = ObservatoryConfig(
        code=obs.get("code", ""),
        name=obs.get("name", ""),
        longitude=obs.get("longitude", 0.0),
        latitude=obs.get("latitude", 0.0),
        altitude=obs.get("altitude", 0.0),
    )

    # 检测参数
    config.thresh = data.get("thresh", 80)
    config.min_area = data.get("min_area", 6)
    config.sharpness = data.get("sharpness", 1.2)
    config.max_sharpness = data.get("max_sharpness", 5.0)
    config.contrast = data.get("contrast", 15)
    config.edge_margin = data.get("edge_margin", 10)
    config.dynamic_thresh = data.get("dynamic_thresh", False)
    config.kill_flat = data.get("kill_flat", True)
    config.kill_dipole = data.get("kill_dipole", True)

    # AI 参数
    config.model_path = data.get("model_path", "")
    config.model_format = data.get("model_format", "auto")
    config.crowd_high_score = data.get("crowd_high_score", 0.85)
    config.crowd_high_count = data.get("crowd_high_count", 10)
    config.crowd_high_penalty = data.get("crowd_high_penalty", 0.50)

    # 保存参数
    bit = data.get("save_bit_depth", 16)
    config.save_bit_depth = BitDepth.INT32 if bit == 32 else BitDepth.INT16

    # 闪烁
    config.blink_speed_ms = data.get("blink_speed_ms", 500)

    # MPCORB
    config.mpcorb_path = data.get("mpcorb_path", "")
    config.limit_magnitude = data.get("limit_magnitude", 20.0)

    return config


def save_config(
    config: AppConfig,
    path: Optional[Union[str, Path]] = None,
) -> Path:
    """保存配置到 JSON 文件

    Args:
        config: 配置对象
        path: 保存路径 (None=默认位置)

    Returns:
        保存的文件路径
    """
    if path is None:
        path = get_default_config_path()
    path = Path(path)

    data = {
        "new_folder": config.new_folder,
        "old_folder": config.old_folder,
        "save_folder": config.save_folder,
        "telescope": {
            "pixel_size_um": config.telescope.pixel_size_um,
            "pixel_scale_arcsec": config.telescope.pixel_scale_arcsec,
            "focal_length_mm": config.telescope.focal_length_mm,
            "camera_rotation_deg": config.telescope.camera_rotation_deg,
        },
        "observatory": {
            "code": config.observatory.code,
            "name": config.observatory.name,
            "longitude": config.observatory.longitude,
            "latitude": config.observatory.latitude,
            "altitude": config.observatory.altitude,
        },
        "thresh": config.thresh,
        "min_area": config.min_area,
        "sharpness": config.sharpness,
        "max_sharpness": config.max_sharpness,
        "contrast": config.contrast,
        "edge_margin": config.edge_margin,
        "dynamic_thresh": config.dynamic_thresh,
        "kill_flat": config.kill_flat,
        "kill_dipole": config.kill_dipole,
        "model_path": config.model_path,
        "model_format": config.model_format,
        "crowd_high_score": config.crowd_high_score,
        "crowd_high_count": config.crowd_high_count,
        "crowd_high_penalty": config.crowd_high_penalty,
        "save_bit_depth": config.save_bit_depth.value,
        "blink_speed_ms": config.blink_speed_ms,
        "mpcorb_path": config.mpcorb_path,
        "limit_magnitude": config.limit_magnitude,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return path
