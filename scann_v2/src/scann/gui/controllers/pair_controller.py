"""配对流程控制器。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scann.services.pair_service import PairService

if TYPE_CHECKING:
    from scann.gui.main_window import MainWindow


class PairController:
    """集中主窗口中的配对流程事件入口。

    当前提交只完成事件接线，不迁移配对加载细节；
    具体逻辑仍暂时保留在 MainWindow 的内部实现方法中。
    """

    def __init__(self, window: MainWindow, pair_service: PairService) -> None:
        self._window = window
        self._pair_service = pair_service

    @property
    def pair_service(self) -> PairService:
        """暴露配对服务，供后续提交逐步迁移逻辑。"""
        return self._pair_service

    def open_new_folder(self) -> None:
        self._window._open_new_folder_impl()

    def open_old_folder(self) -> None:
        self._window._open_old_folder_impl()

    def add_recent_folder(self, folder: str) -> None:
        self._window._add_recent_folder_impl(folder)

    def update_recent_menu(self) -> None:
        self._window._update_recent_menu_impl()

    def open_recent_folder(self, folder: str) -> None:
        self._window._open_recent_folder_impl(folder)

    def prev_pair(self) -> None:
        self._window._prev_pair_impl()

    def next_pair(self) -> None:
        self._window._next_pair_impl()

    def select_pair(self, index: int) -> None:
        self._window._select_pair_impl(index)