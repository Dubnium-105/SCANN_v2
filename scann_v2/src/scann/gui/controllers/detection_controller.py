"""检测流程控制器。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scann.core.candidate_detector import DetectionParams
    from scann.gui.main_window import MainWindow


class DetectionController:
    """集中主窗口中的检测相关事件入口。"""

    def __init__(self, window: MainWindow) -> None:
        self._window = window

    def batch_align(self) -> None:
        self._window._on_batch_align_impl()

    def batch_process(self) -> None:
        self._window._on_batch_process_impl()

    def run_batch_process(self, params: dict) -> None:
        self._window._run_batch_process_impl(params)

    def build_detection_params(self) -> DetectionParams:
        return self._window._build_detection_params_impl()

    def batch_detect(self) -> None:
        self._window._on_batch_detect_impl()

    def mark_real(self) -> None:
        self._window._on_mark_real_impl()

    def mark_bogus(self) -> None:
        self._window._on_mark_bogus_impl()

    def next_candidate(self) -> None:
        self._window._on_next_candidate_impl()

    def candidate_selected(self, index: int) -> None:
        self._window._on_candidate_selected_impl(index)

    def candidate_double_clicked(self, index: int) -> None:
        self._window._on_candidate_double_clicked_impl(index)

    def focus_candidate(self, index: int) -> None:
        self._window._focus_candidate_impl(index)