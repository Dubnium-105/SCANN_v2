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
    config.telescope_name = data.get("telescope_name", "")

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
    config.max_area = data.get("max_area", 600)
    config.sharpness = data.get("sharpness", 1.2)
    config.max_sharpness = data.get("max_sharpness", 5.0)
    config.contrast = data.get("contrast", 15)
    config.edge_margin = data.get("edge_margin", 10)
    config.exclude_edge = data.get("exclude_edge", True)
    config.dynamic_thresh = data.get("dynamic_thresh", False)
    config.nms_radius = data.get("nms_radius", 5.0)
    config.aspect_ratio_max = data.get("aspect_ratio_max", 3.0)
    config.extent_max = data.get("extent_max", 0.90)
    config.topk = data.get("topk", 20)
    config.kill_flat = data.get("kill_flat", True)
    config.kill_dipole = data.get("kill_dipole", True)

    # AI 参数
    config.model_path = data.get("model_path", "")
    config.model_format = data.get("model_format", "auto")
    config.ai_confidence = data.get("ai_confidence", 0.50)
    config.slice_size = data.get("slice_size", 80)
    config.batch_size = data.get("batch_size", 64)
    config.compute_device = data.get("compute_device", "auto")
    config.crowd_high_score = data.get("crowd_high_score", 0.85)
    config.crowd_high_count = data.get("crowd_high_count", 10)
    config.crowd_high_penalty = data.get("crowd_high_penalty", 0.50)

    # 保存参数
    bit = data.get("save_bit_depth", 16)
    config.save_bit_depth = BitDepth.INT32 if bit == 32 else BitDepth.INT16
    config.save_format = data.get("save_format", "FITS (16-bit)")
    config.database_path = data.get("database_path", "")

    # 闪烁
    config.blink_speed_ms = data.get("blink_speed_ms", 500)

    # MPCORB
    config.mpcorb_path = data.get("mpcorb_path", "")
    config.limit_magnitude = data.get("limit_magnitude", 20.0)

    # 最近打开
    config.recent_folders = data.get("recent_folders", [])
    config.max_recent_count = data.get("max_recent_count", 10)

    # 高级/UI 选项
    config.max_threads = data.get("max_threads", 4)
    config.auto_save_annotations = data.get("auto_save_annotations", False)
    config.auto_collapse_sidebar = data.get("auto_collapse_sidebar", True)
    config.confirm_before_close = data.get("confirm_before_close", True)

    # 直方图拉伸参数
    config.stretch_black_point = data.get("stretch_black_point", 0.0)
    config.stretch_white_point = data.get("stretch_white_point", 65535.0)
    config.stretch_mode = data.get("stretch_mode", "线性")

    # 视图开关
    config.show_markers = data.get("show_markers", True)
    config.show_mpcorb = data.get("show_mpcorb", True)
    config.show_known_objects = data.get("show_known_objects", True)
    config.histogram_visible = data.get("histogram_visible", False)
    config.sidebar_collapsed = data.get("sidebar_collapsed", False)

    # 窗口几何
    config.window_width = data.get("window_width", 1600)
    config.window_height = data.get("window_height", 1000)

    # 标注工具选项
    config.ann_mode = data.get("ann_mode", "v1")
    config.ann_dataset_path = data.get("ann_dataset_path", "")
    config.ann_auto_advance = data.get("ann_auto_advance", True)
    config.ann_filter = data.get("ann_filter", "all")
    config.ann_sort = data.get("ann_sort", "默认")
    config.ann_bbox_width = data.get("ann_bbox_width", 2)
    config.ann_invert = data.get("ann_invert", False)
    config.ann_splitter_sizes = data.get("ann_splitter_sizes", [])
    config.ann_window_width = data.get("ann_window_width", 1000)
    config.ann_window_height = data.get("ann_window_height", 700)
    config.ann_stretch_black = data.get("ann_stretch_black", 0.0)
    config.ann_stretch_white = data.get("ann_stretch_white", 65535.0)
    config.ann_stretch_mode = data.get("ann_stretch_mode", "线性")
    config.ann_histogram_visible = data.get("ann_histogram_visible", False)

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
        "telescope_name": config.telescope_name,
        "observatory": {
            "code": config.observatory.code,
            "name": config.observatory.name,
            "longitude": config.observatory.longitude,
            "latitude": config.observatory.latitude,
            "altitude": config.observatory.altitude,
        },
        "thresh": config.thresh,
        "min_area": config.min_area,
        "max_area": config.max_area,
        "sharpness": config.sharpness,
        "max_sharpness": config.max_sharpness,
        "contrast": config.contrast,
        "edge_margin": config.edge_margin,
        "exclude_edge": config.exclude_edge,
        "dynamic_thresh": config.dynamic_thresh,
        "nms_radius": config.nms_radius,
        "aspect_ratio_max": config.aspect_ratio_max,
        "extent_max": config.extent_max,
        "topk": config.topk,
        "kill_flat": config.kill_flat,
        "kill_dipole": config.kill_dipole,
        "model_path": config.model_path,
        "model_format": config.model_format,
        "ai_confidence": config.ai_confidence,
        "slice_size": config.slice_size,
        "batch_size": config.batch_size,
        "compute_device": config.compute_device,
        "crowd_high_score": config.crowd_high_score,
        "crowd_high_count": config.crowd_high_count,
        "crowd_high_penalty": config.crowd_high_penalty,
        "save_bit_depth": config.save_bit_depth.value,
        "save_format": config.save_format,
        "database_path": config.database_path,
        "blink_speed_ms": config.blink_speed_ms,
        "mpcorb_path": config.mpcorb_path,
        "limit_magnitude": config.limit_magnitude,
        "recent_folders": config.recent_folders,
        "max_recent_count": config.max_recent_count,
        "max_threads": config.max_threads,
        "auto_save_annotations": config.auto_save_annotations,
        "auto_collapse_sidebar": config.auto_collapse_sidebar,
        "confirm_before_close": config.confirm_before_close,
        "stretch_black_point": config.stretch_black_point,
        "stretch_white_point": config.stretch_white_point,
        "stretch_mode": config.stretch_mode,
        "show_markers": config.show_markers,
        "show_mpcorb": config.show_mpcorb,
        "show_known_objects": config.show_known_objects,
        "histogram_visible": config.histogram_visible,
        "sidebar_collapsed": config.sidebar_collapsed,
        "window_width": config.window_width,
        "window_height": config.window_height,
        "ann_mode": config.ann_mode,
        "ann_dataset_path": config.ann_dataset_path,
        "ann_auto_advance": config.ann_auto_advance,
        "ann_filter": config.ann_filter,
        "ann_sort": config.ann_sort,
        "ann_bbox_width": config.ann_bbox_width,
        "ann_invert": config.ann_invert,
        "ann_splitter_sizes": config.ann_splitter_sizes,
        "ann_window_width": config.ann_window_width,
        "ann_window_height": config.ann_window_height,
        "ann_stretch_black": config.ann_stretch_black,
        "ann_stretch_white": config.ann_stretch_white,
        "ann_stretch_mode": config.ann_stretch_mode,
        "ann_histogram_visible": config.ann_histogram_visible,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return path
