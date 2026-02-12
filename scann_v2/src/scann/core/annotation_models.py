"""标注系统数据模型

定义标注工具系统所需的所有数据结构，兼容 v1 三联图分类和 v2 FITS 全图检测两种模式。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─────────────────────── 枚举常量 ───────────────────────


class AnnotationLabel(str, Enum):
    """标注大类标签"""
    REAL = "real"
    BOGUS = "bogus"


class DetailType(str, Enum):
    """标注细分类型

    A. 真类 (REAL):
        asteroid       - 小行星、彗星等移动天体
        supernova      - 暂现源新星
        variable_star  - 周期/非周期变星

    B. 假类 (BOGUS):
        satellite_trail     - 人造卫星划过轨迹
        noise               - 图像噪点/热点
        diffraction_spike   - 衍射芒
        cmos_condensation   - 传感器结霜伪影
        corresponding       - 新旧图均有，亮度差异大/对齐后旋转等
    """
    # A. 真类
    ASTEROID = "asteroid"
    SUPERNOVA = "supernova"
    VARIABLE_STAR = "variable_star"

    # B. 假类
    SATELLITE_TRAIL = "satellite_trail"
    NOISE = "noise"
    DIFFRACTION_SPIKE = "diffraction_spike"
    CMOS_CONDENSATION = "cmos_condensation"
    CORRESPONDING = "corresponding"


# 类型→大类标签映射
DETAIL_TYPE_TO_LABEL: dict[DetailType, AnnotationLabel] = {
    DetailType.ASTEROID: AnnotationLabel.REAL,
    DetailType.SUPERNOVA: AnnotationLabel.REAL,
    DetailType.VARIABLE_STAR: AnnotationLabel.REAL,
    DetailType.SATELLITE_TRAIL: AnnotationLabel.BOGUS,
    DetailType.NOISE: AnnotationLabel.BOGUS,
    DetailType.DIFFRACTION_SPIKE: AnnotationLabel.BOGUS,
    DetailType.CMOS_CONDENSATION: AnnotationLabel.BOGUS,
    DetailType.CORRESPONDING: AnnotationLabel.BOGUS,
}

# 详细类型显示文本
DETAIL_TYPE_DISPLAY: dict[DetailType, str] = {
    DetailType.ASTEROID: "小行星 ★",
    DetailType.SUPERNOVA: "超新星 💥",
    DetailType.VARIABLE_STAR: "变星 ✦",
    DetailType.SATELLITE_TRAIL: "卫星线 🛰️",
    DetailType.NOISE: "噪点 ⚡",
    DetailType.DIFFRACTION_SPIKE: "星芒 ✨",
    DetailType.CMOS_CONDENSATION: "CMOS结霜 ❄️",
    DetailType.CORRESPONDING: "有对应 🔀",
}

# 快捷键映射: Y1-Y3 → 真子类型, N1-N5 → 假子类型
SHORTCUT_TO_DETAIL_TYPE: dict[str, DetailType] = {
    "Y1": DetailType.ASTEROID,
    "Y2": DetailType.SUPERNOVA,
    "Y3": DetailType.VARIABLE_STAR,
    "N1": DetailType.SATELLITE_TRAIL,
    "N2": DetailType.NOISE,
    "N3": DetailType.DIFFRACTION_SPIKE,
    "N4": DetailType.CMOS_CONDENSATION,
    "N5": DetailType.CORRESPONDING,
}

# 详细类型颜色映射
DETAIL_TYPE_COLOR: dict[DetailType, str] = {
    # A. 真类 - 绿色系
    DetailType.ASTEROID: "#2E7D32",         # 深绿
    DetailType.SUPERNOVA: "#00E676",        # 鲜绿
    DetailType.VARIABLE_STAR: "#69F0AE",    # 浅绿
    # B. 假类 - 红色/橙色/灰色系
    DetailType.SATELLITE_TRAIL: "#C62828",  # 深红
    DetailType.NOISE: "#EF5350",             # 橙红
    DetailType.DIFFRACTION_SPIKE: "#FF9800",# 橙色
    DetailType.CMOS_CONDENSATION: "#FFB74D",# 灰橙
    DetailType.CORRESPONDING: "#BDBDBD",     # 灰色
}

# 默认颜色（未标注）
DEFAULT_BBOX_COLOR = "#FFEB3B"  # 黄色
SELECTED_BBOX_COLOR = "#9C27B0"  # 紫色选中


# ─────────────────────── 数据类 ───────────────────────


@dataclass
class BBox:
    """边界框 (v2 FITS 全图标注模式使用)

    Attributes:
        x: 左上角 X 坐标 (像素)
        y: 左上角 Y 坐标 (像素)
        width: 宽度 (像素)
        height: 高度 (像素)
        label: 大类标签 (real/bogus)
        confidence: 置信度 (人工标注=1.0, AI预标注=模型输出)
        detail_type: 详细标注类型
    """
    x: int
    y: int
    width: int
    height: int
    label: str = "real"
    confidence: float = 1.0
    detail_type: Optional[str] = None

    @property
    def center(self) -> tuple[int, int]:
        """返回边界框中心坐标"""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        """返回边界框面积"""
        return self.width * self.height

    def contains(self, px: int, py: int) -> bool:
        """判断点 (px, py) 是否在边界框内"""
        return (self.x <= px < self.x + self.width and
                self.y <= py < self.y + self.height)

    def to_dict(self) -> dict:
        """序列化为字典"""
        d = {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "label": self.label,
            "confidence": self.confidence,
        }
        if self.detail_type is not None:
            d["detail_type"] = self.detail_type
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "BBox":
        """从字典反序列化"""
        return cls(
            x=data["x"],
            y=data["y"],
            width=data["width"],
            height=data["height"],
            label=data.get("label", "real"),
            confidence=data.get("confidence", 1.0),
            detail_type=data.get("detail_type"),
        )


@dataclass
class AnnotationSample:
    """单个标注样本

    Attributes:
        id: 唯一标识 (通常为文件名或序号)
        source_path: 原始文件路径
        display_name: 显示名称
        label: 当前标签 (real/bogus/None=未标注)
        detail_type: 详细类型 (asteroid/noise/...)
        bboxes: 边界框列表 (v2 模式)
        ai_suggestion: AI 预标注建议标签
        ai_confidence: AI 预标注置信度
        metadata: 扩展元数据
    """
    id: str
    source_path: str
    display_name: str
    label: Optional[str] = None
    detail_type: Optional[str] = None
    bboxes: list[BBox] = field(default_factory=list)
    ai_suggestion: Optional[str] = None
    ai_confidence: Optional[float] = None
    metadata: dict = field(default_factory=dict)

    @property
    def is_labeled(self) -> bool:
        """是否已标注"""
        return self.label is not None

    @property
    def label_display(self) -> str:
        """获取标签的显示文本"""
        if self.detail_type is not None:
            try:
                dt = DetailType(self.detail_type)
                return DETAIL_TYPE_DISPLAY.get(dt, self.detail_type)
            except ValueError:
                return self.detail_type
        if self.label == "real":
            return "A.真"
        if self.label == "bogus":
            return "B.假"
        return "未标注"

    def to_dict(self) -> dict:
        """序列化为字典"""
        d = {
            "id": self.id,
            "source_path": self.source_path,
            "display_name": self.display_name,
        }
        if self.label is not None:
            d["label"] = self.label
        if self.detail_type is not None:
            d["detail_type"] = self.detail_type
        if self.bboxes:
            d["bboxes"] = [b.to_dict() for b in self.bboxes]
        if self.ai_suggestion is not None:
            d["ai_suggestion"] = self.ai_suggestion
        if self.ai_confidence is not None:
            d["ai_confidence"] = self.ai_confidence
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "AnnotationSample":
        """从字典反序列化"""
        bboxes = [BBox.from_dict(b) for b in data.get("bboxes", [])]
        return cls(
            id=data["id"],
            source_path=data["source_path"],
            display_name=data["display_name"],
            label=data.get("label"),
            detail_type=data.get("detail_type"),
            bboxes=bboxes,
            ai_suggestion=data.get("ai_suggestion"),
            ai_confidence=data.get("ai_confidence"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AnnotationStats:
    """标注统计信息

    Attributes:
        total: 总样本数
        labeled: 已标注数
        unlabeled: 未标注数
        label_counts: 各标签计数 {"asteroid": 32, "noise": 15, ...}
        progress_percent: 进度百分比 (0.0~100.0)
    """
    total: int = 0
    labeled: int = 0
    unlabeled: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)
    progress_percent: float = 0.0

    def update_from_samples(self, samples: list[AnnotationSample]) -> None:
        """从样本列表计算统计信息"""
        self.total = len(samples)
        self.labeled = sum(1 for s in samples if s.is_labeled)
        self.unlabeled = self.total - self.labeled
        self.progress_percent = (self.labeled / self.total * 100.0) if self.total > 0 else 0.0

        self.label_counts.clear()
        for s in samples:
            if s.detail_type is not None:
                self.label_counts[s.detail_type] = self.label_counts.get(s.detail_type, 0) + 1
            elif s.label is not None:
                self.label_counts[s.label] = self.label_counts.get(s.label, 0) + 1

    @property
    def real_count(self) -> int:
        """真类总数"""
        real_types = {dt.value for dt, lbl in DETAIL_TYPE_TO_LABEL.items()
                      if lbl == AnnotationLabel.REAL}
        count = sum(self.label_counts.get(t, 0) for t in real_types)
        count += self.label_counts.get("real", 0)
        return count

    @property
    def bogus_count(self) -> int:
        """假类总数"""
        bogus_types = {dt.value for dt, lbl in DETAIL_TYPE_TO_LABEL.items()
                       if lbl == AnnotationLabel.BOGUS}
        count = sum(self.label_counts.get(t, 0) for t in bogus_types)
        count += self.label_counts.get("bogus", 0)
        return count


@dataclass
class ExportResult:
    """数据集导出结果

    Attributes:
        success: 是否成功
        output_dir: 输出目录
        total_exported: 导出样本总数
        train_count: 训练集数量
        val_count: 验证集数量
        format: 导出格式
        error_message: 错误信息
    """
    success: bool = True
    output_dir: str = ""
    total_exported: int = 0
    train_count: int = 0
    val_count: int = 0
    format: str = "native"
    error_message: str = ""


@dataclass
class AnnotationAction:
    """标注操作记录 (用于撤销/重做)

    Attributes:
        action_type: 操作类型 (label/bbox_add/bbox_remove/bbox_edit/move_file)
        sample_id: 样本ID
        old_value: 操作前的值
        new_value: 操作后的值
    """
    action_type: str
    sample_id: str
    old_value: Optional[dict] = None
    new_value: Optional[dict] = None
