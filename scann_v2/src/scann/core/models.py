"""Core data models for SCANN v2.

所有数据模型使用 dataclass 定义，确保不可变性和类型安全。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional

import numpy as np


# ─────────────────────── Enums ───────────────────────


class BitDepth(Enum):
    """FITS 保存位深度 - 仅整数"""
    INT16 = 16
    INT32 = 32


class MarkerType(Enum):
    """标记类型"""
    BOUNDING_BOX = auto()
    CROSSHAIR = auto()


class TargetVerdict(Enum):
    """目标判决"""
    REAL = "real"
    BOGUS = "bogus"
    UNKNOWN = "unknown"


# ─────────────────────── FITS 相关 ───────────────────────


@dataclass(frozen=True)
class FitsHeader:
    """FITS 文件头信息的结构化表示"""
    raw: dict[str, Any]  # 原始头信息键值对

    @property
    def observation_datetime(self) -> Optional[datetime]:
        """提取观测日期时间 (DATE-OBS)"""
        date_obs = self.raw.get("DATE-OBS")
        if date_obs is None:
            return None
        if isinstance(date_obs, str):
            # 尝试多种常见格式
            for fmt in ["%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]:
                try:
                    return datetime.strptime(date_obs, fmt)
                except ValueError:
                    continue
        return None

    @property
    def exposure_time(self) -> Optional[float]:
        """曝光时间 (EXPTIME)"""
        val = self.raw.get("EXPTIME") or self.raw.get("EXPOSURE")
        return float(val) if val is not None else None

    @property
    def object_name(self) -> Optional[str]:
        """目标名称 (OBJECT)"""
        return self.raw.get("OBJECT")

    @property
    def ra(self) -> Optional[float]:
        """赤经 (RA, degrees)"""
        val = self.raw.get("RA") or self.raw.get("CRVAL1")
        return float(val) if val is not None else None

    @property
    def dec(self) -> Optional[float]:
        """赤纬 (DEC, degrees)"""
        val = self.raw.get("DEC") or self.raw.get("CRVAL2")
        return float(val) if val is not None else None


@dataclass
class FitsImage:
    """FITS 图像数据 + 头信息

    Attributes:
        data: 像素数据 (numpy array)
        header: FITS 文件头
        path: 文件路径 (可选)
    """
    data: np.ndarray
    header: FitsHeader
    path: Optional[Path] = None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype


# ─────────────────────── 图像对齐 ───────────────────────


@dataclass(frozen=True)
class AlignResult:
    """图像对齐结果

    Attributes:
        aligned_old: 对齐后的旧图数据
        dx: X 方向偏移量 (像素)
        dy: Y 方向偏移量 (像素)
        rotation: 旋转角 (度)
        success: 是否成功
        error_message: 失败原因
    """
    aligned_old: Optional[np.ndarray]
    dx: float = 0.0
    dy: float = 0.0
    rotation: float = 0.0
    success: bool = True
    error_message: str = ""


# ─────────────────────── 图像对 ───────────────────────


@dataclass
class ImagePair:
    """新旧图像配对

    Attributes:
        name: 配对名称 (通常是文件名主干)
        new_image: 新图
        old_image: 旧图 (可能已对齐)
        aligned: 是否已完成对齐
    """
    name: str
    new_image: FitsImage
    old_image: FitsImage
    aligned: bool = False


# ─────────────────────── 候选体检测 ───────────────────────


@dataclass
class CandidateFeatures:
    """候选体的量化特征"""
    peak: float = 0.0         # 峰值亮度
    mean: float = 0.0         # 平均亮度
    sharpness: float = 0.0    # 锐度 (peak / mean)
    contrast: float = 0.0     # 对比度 (peak - median)
    area: float = 0.0         # 面积 (像素)
    rise: float = 0.0         # 增亮 (new - old)
    val_new: float = 0.0      # 新图局部最大值
    val_old: float = 0.0      # 旧图局部最大值
    extent: float = 0.0       # 填充率
    aspect_ratio: float = 1.0 # 长宽比


@dataclass
class Candidate:
    """单个候选目标

    Attributes:
        x: 像素坐标 X (在对齐后的图像中)
        y: 像素坐标 Y
        features: 量化特征
        ai_score: AI 评分 (0~1)
        verdict: 人工判决
        is_manual: 是否手动添加
        is_known: 是否为已知天体
        known_id: 已知天体 ID
    """
    x: int
    y: int
    features: CandidateFeatures = field(default_factory=CandidateFeatures)
    ai_score: float = 0.0
    verdict: TargetVerdict = TargetVerdict.UNKNOWN
    is_manual: bool = False
    is_known: bool = False
    known_id: str = ""


# ─────────────────────── AI 检测 ───────────────────────


@dataclass(frozen=True)
class Detection:
    """AI 全图检测结果"""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    marker_type: MarkerType = MarkerType.BOUNDING_BOX


# ─────────────────────── 天文坐标 ───────────────────────


@dataclass(frozen=True)
class SkyPosition:
    """天球坐标位置"""
    ra: float   # 赤经 (degrees)
    dec: float  # 赤纬 (degrees)
    mag: Optional[float] = None  # 星等
    name: str = ""


# ─────────────────────── 配置 ───────────────────────


@dataclass
class TelescopeConfig:
    """望远镜/相机参数"""
    pixel_size_um: float = 9.0          # 像素大小 (μm)
    pixel_scale_arcsec: float = 0.0     # 像素分辨率 (arcsec/pix)，由焦距和像素大小推算
    focal_length_mm: float = 0.0        # 焦距 (mm)
    camera_rotation_deg: float = 0.0    # 相机旋转角 (度)

    def compute_pixel_scale(self) -> float:
        """由像素大小和焦距计算像素分辨率"""
        if self.focal_length_mm <= 0:
            return 0.0
        # arcsec/pix = 206265 * pixel_size_mm / focal_length_mm
        return 206.265 * self.pixel_size_um / self.focal_length_mm


@dataclass
class ObservatoryConfig:
    """天文台参数"""
    code: str = ""              # MPC 天文台代码
    name: str = ""
    longitude: float = 0.0     # 经度 (度)
    latitude: float = 0.0      # 纬度 (度)
    altitude: float = 0.0      # 海拔 (米)


@dataclass
class AppConfig:
    """应用程序完整配置"""
    # 路径
    new_folder: str = ""        # 新图文件夹
    old_folder: str = ""        # 旧图文件夹
    save_folder: str = ""       # 保存文件夹

    # 望远镜参数
    telescope: TelescopeConfig = field(default_factory=TelescopeConfig)
    telescope_name: str = ""    # 望远镜名称描述
    observatory: ObservatoryConfig = field(default_factory=ObservatoryConfig)

    # 检测参数
    thresh: int = 80
    min_area: int = 6
    max_area: int = 600         # 最大面积 (px)
    sharpness: float = 1.2
    max_sharpness: float = 5.0
    contrast: int = 15
    edge_margin: int = 10
    exclude_edge: bool = True   # 是否排除边缘区域
    dynamic_thresh: bool = False
    nms_radius: float = 5.0     # NMS 非极大值抑制半径 (px)
    aspect_ratio_max: float = 3.0
    extent_max: float = 0.90
    topk: int = 20              # Top-K 候选体上限

    # 过滤器
    kill_flat: bool = True
    kill_dipole: bool = True

    # AI 参数
    model_path: str = ""
    model_format: str = "auto"  # "auto", "v1_classifier", "v2_classifier"
    ai_confidence: float = 0.50 # AI 置信度阈值
    slice_size: int = 64        # 切片大小 (px)
    batch_size: int = 64        # 推理批量大小
    compute_device: str = "auto"  # "auto", "cpu", "cuda"
    crowd_high_score: float = 0.85
    crowd_high_count: int = 10
    crowd_high_penalty: float = 0.50

    # 保存参数
    save_bit_depth: BitDepth = BitDepth.INT16
    save_format: str = "FITS (16-bit)"  # "FITS (16-bit)", "FITS (32-bit)", "PNG (8-bit)"
    database_path: str = ""     # 数据库路径

    # 闪烁
    blink_speed_ms: int = 500

    # MPCORB
    mpcorb_path: str = ""
    limit_magnitude: float = 20.0

    # 最近打开
    recent_folders: list = field(default_factory=list)  # 最近打开的文件夹列表
    max_recent_count: int = 10  # 最近打开列表最大数量

    # 高级/UI 选项
    max_threads: int = 4        # 最大线程数
    auto_save_annotations: bool = False  # 退出时自动保存标记
    auto_collapse_sidebar: bool = True   # 窗口 < 1200px 时自动折叠侧边栏
    confirm_before_close: bool = True    # 关闭前确认

    # 直方图拉伸参数 (显示用，不改变原始数据)
    stretch_black_point: float = 0.0     # 黑点值 (原始像素值)
    stretch_white_point: float = 65535.0 # 白点值 (原始像素值)
    stretch_mode: str = "线性"           # 拉伸预设: "线性","对数","平方根","Asinh","自动拉伸"

    # 视图开关 (菜单栏 "视图" 中的可勾选项)
    show_markers: bool = True            # 显示候选标记
    show_mpcorb: bool = True             # 显示 MPCORB 叠加
    show_known_objects: bool = True       # 显示已知天体
    histogram_visible: bool = False       # 直方图面板是否可见
    sidebar_collapsed: bool = False       # 侧边栏是否折叠

    # 窗口几何
    window_width: int = 1600
    window_height: int = 1000

    # 标注工具选项
    ann_mode: str = "v1"                    # 标注模式: "v1" 三联图 / "v2" FITS
    ann_dataset_path: str = ""              # 上次使用的数据集路径
    ann_auto_advance: bool = True           # 标注后自动下一个
    ann_filter: str = "all"                 # 筛选: "all","unlabeled","real","bogus"
    ann_sort: str = "默认"                   # 排序方式
    ann_bbox_width: int = 2                 # 边框粗细 (1-5)
    ann_invert: bool = False                # 反色显示
    ann_splitter_sizes: list = field(default_factory=list)  # 分割面板比例
    ann_window_width: int = 1000            # 标注窗口宽度
    ann_window_height: int = 700            # 标注窗口高度
    ann_stretch_black: float = 0.0          # 标注工具直方图黑点
    ann_stretch_white: float = 65535.0      # 标注工具直方图白点
    ann_stretch_mode: str = "线性"           # 标注工具拉伸预设
    ann_histogram_visible: bool = False     # 标注工具直方图面板是否可见
