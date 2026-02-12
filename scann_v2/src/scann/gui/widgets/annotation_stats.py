"""标注统计面板

实时显示标注进度、各类别计数等信息。
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtWidgets import (
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from scann.core.annotation_models import AnnotationStats


class AnnotationStatsPanel(QWidget):
    """标注统计面板

    显示:
    - 总计 / 已标注 / 未标注
    - 真/假类计数
    - 进度条
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self._lbl_total = QLabel("总计: 0")
        self._lbl_labeled = QLabel("已标注: 0 (0%)")
        self._lbl_real = QLabel("  A.真: 0")
        self._lbl_bogus = QLabel("  B.假: 0")
        self._lbl_unlabeled = QLabel("未标注: 0")
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFixedHeight(16)

        for w in [self._lbl_total, self._lbl_labeled, self._lbl_real,
                  self._lbl_bogus, self._lbl_unlabeled, self._progress]:
            layout.addWidget(w)

    def update_stats(self, stats: AnnotationStats) -> None:
        """更新统计显示"""
        self._lbl_total.setText(f"总计: {stats.total}")
        self._lbl_labeled.setText(
            f"已标注: {stats.labeled} ({stats.progress_percent:.1f}%)"
        )
        self._lbl_real.setText(f"  A.真: {stats.real_count}")
        self._lbl_bogus.setText(f"  B.假: {stats.bogus_count}")
        self._lbl_unlabeled.setText(f"未标注: {stats.unlabeled}")
        self._progress.setValue(int(stats.progress_percent))
