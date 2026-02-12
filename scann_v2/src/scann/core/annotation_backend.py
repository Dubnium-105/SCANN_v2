"""标注后端抽象接口

采用策略模式 (Strategy Pattern) 将数据加载/保存与 UI 解耦，
兼容 v1 三联图分类和 v2 FITS 全图检测两种标注模式。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
from PIL import Image

from scann.core.annotation_models import (
    AnnotationAction,
    AnnotationSample,
    AnnotationStats,
    BBox,
    ExportResult,
)

# 图像数据类型: numpy 数组 (FITS) 或 PIL Image (三联图 PNG)
ImageData = Union[np.ndarray, Image.Image]


class AnnotationBackend(ABC):
    """标注后端抽象基类 — 隔离数据格式差异

    子类需实现以下抽象方法:
    - load_samples: 加载样本列表
    - save_annotation: 保存单个标注
    - get_image_data: 获取图像数据
    - get_display_info: 获取显示信息
    - export_dataset: 导出数据集
    - get_statistics: 获取统计信息
    - supports_bbox: 是否支持边界框

    默认提供基于操作栈的 undo/redo 实现。
    """

    def __init__(self) -> None:
        self._samples: list[AnnotationSample] = []
        self._sample_index: dict[str, int] = {}  # id → index
        self._undo_stack: list[AnnotationAction] = []
        self._redo_stack: list[AnnotationAction] = []
        self._max_undo_size: int = 100

    # ─── 样本管理 ───

    @property
    def samples(self) -> list[AnnotationSample]:
        """返回当前所有样本"""
        return self._samples

    def get_sample(self, sample_id: str) -> Optional[AnnotationSample]:
        """根据 ID 获取样本"""
        idx = self._sample_index.get(sample_id)
        if idx is not None and idx < len(self._samples):
            return self._samples[idx]
        return None

    def _rebuild_index(self) -> None:
        """重建样本索引"""
        self._sample_index = {s.id: i for i, s in enumerate(self._samples)}

    # ─── 抽象方法 ───

    @abstractmethod
    def load_samples(self, path: str, filter: str = "all") -> list[AnnotationSample]:
        """加载样本列表

        Args:
            path: 数据集路径 (文件夹或文件)
            filter: 筛选条件 ("all" / "unlabeled" / "real" / "bogus")

        Returns:
            加载的样本列表
        """
        ...

    @abstractmethod
    def save_annotation(
        self,
        sample_id: str,
        label: str,
        bbox: Optional[BBox] = None,
        confidence: float = 1.0,
        detail_type: Optional[str] = None,
    ) -> None:
        """保存单个标注

        Args:
            sample_id: 样本 ID
            label: 标签 (real/bogus)
            bbox: 边界框 (仅 v2 模式)
            confidence: 置信度
            detail_type: 详细子类型
        """
        ...

    @abstractmethod
    def get_image_data(self, sample: AnnotationSample) -> ImageData:
        """获取样本的图像数据

        Args:
            sample: 标注样本

        Returns:
            numpy 数组 (FITS) 或 PIL Image (PNG)
        """
        ...

    @abstractmethod
    def get_display_info(self, sample: AnnotationSample) -> dict:
        """获取样本的显示信息

        Returns:
            dict: {
                "file_name": str,
                "label": Optional[str],
                "detail_type": Optional[str],
                "label_display": str,
                "has_new_image": bool,
                "has_old_image": bool,
            }
        """
        ...

    @abstractmethod
    def export_dataset(
        self, output_dir: str, format: str = "native",
        include_unlabeled: bool = False,
        val_split: float = 0.0,
    ) -> ExportResult:
        """导出标注数据集

        Args:
            output_dir: 输出目录
            format: 导出格式 ("native" / "csv" / "coco" / "yolo" / "voc")
            include_unlabeled: 是否包含未标注样本
            val_split: 验证集比例 (0 则不拆分)

        Returns:
            导出结果
        """
        ...

    @abstractmethod
    def get_statistics(self) -> AnnotationStats:
        """获取当前标注统计信息"""
        ...

    @abstractmethod
    def supports_bbox(self) -> bool:
        """是否支持边界框标注

        Returns:
            v1 三联图=False, v2 FITS=True
        """
        ...

    # ─── 撤销/重做 (基于操作栈默认实现) ───

    def _push_undo(self, action: AnnotationAction) -> None:
        """记录操作到撤销栈"""
        self._undo_stack.append(action)
        if len(self._undo_stack) > self._max_undo_size:
            self._undo_stack.pop(0)
        # 新操作清空重做栈
        self._redo_stack.clear()

    def undo(self) -> bool:
        """撤销上一步操作

        Returns:
            是否成功撤销
        """
        if not self._undo_stack:
            return False
        action = self._undo_stack.pop()
        self._apply_undo(action)
        self._redo_stack.append(action)
        return True

    def redo(self) -> bool:
        """重做已撤销的操作

        Returns:
            是否成功重做
        """
        if not self._redo_stack:
            return False
        action = self._redo_stack.pop()
        self._apply_redo(action)
        self._undo_stack.append(action)
        return True

    def _apply_undo(self, action: AnnotationAction) -> None:
        """应用撤销 — 将样本恢复到操作前状态"""
        sample = self.get_sample(action.sample_id)
        if sample is None:
            return
        if action.old_value is not None:
            if "label" in action.old_value:
                sample.label = action.old_value["label"]
            if "detail_type" in action.old_value:
                sample.detail_type = action.old_value["detail_type"]
            if "bboxes" in action.old_value:
                sample.bboxes = [BBox.from_dict(b) for b in action.old_value["bboxes"]]

    def _apply_redo(self, action: AnnotationAction) -> None:
        """应用重做 — 将样本恢复到操作后状态"""
        sample = self.get_sample(action.sample_id)
        if sample is None:
            return
        if action.new_value is not None:
            if "label" in action.new_value:
                sample.label = action.new_value["label"]
            if "detail_type" in action.new_value:
                sample.detail_type = action.new_value["detail_type"]
            if "bboxes" in action.new_value:
                sample.bboxes = [BBox.from_dict(b) for b in action.new_value["bboxes"]]

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    # ─── 筛选 ───

    def get_filtered_samples(self, filter: str = "all") -> list[AnnotationSample]:
        """按条件筛选样本

        Args:
            filter: "all" / "unlabeled" / "real" / "bogus"

        Returns:
            筛选后的样本列表
        """
        if filter == "all":
            return list(self._samples)
        elif filter == "unlabeled":
            return [s for s in self._samples if not s.is_labeled]
        elif filter == "real":
            return [s for s in self._samples if s.label == "real"]
        elif filter == "bogus":
            return [s for s in self._samples if s.label == "bogus"]
        return list(self._samples)
