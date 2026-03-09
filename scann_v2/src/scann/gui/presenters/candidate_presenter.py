"""候选体表格与 marker 展示器。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from scann.core.models import Candidate

if TYPE_CHECKING:
    from scann.gui.image_viewer import FitsImageViewer
    from scann.gui.widgets.suspect_table import SuspectTableWidget


class CandidatePresenter:
    """统一封装候选体列表与图像标记刷新。"""

    def __init__(self, suspect_table: SuspectTableWidget, image_viewer: FitsImageViewer) -> None:
        self._suspect_table = suspect_table
        self._image_viewer = image_viewer

    def set_candidates(self, candidates: Sequence[Candidate]) -> None:
        """将候选体列表同步到表格。"""
        self._suspect_table.set_candidates(list(candidates))

    def refresh_markers(
        self,
        candidates: Sequence[Candidate],
        *,
        selected_idx: int = -1,
        show_markers: bool = True,
    ) -> None:
        """按当前显示配置刷新图像上的候选体标记。"""
        self._image_viewer.draw_markers(
            list(candidates),
            selected_idx=selected_idx,
            hide_all=not show_markers,
        )