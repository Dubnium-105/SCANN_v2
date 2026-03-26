"""配对流程控制器。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PyQt5.QtWidgets import QFileDialog

from scann.services.pair_service import PairService

if TYPE_CHECKING:
    from scann.gui.main_window import MainWindow


class PairController:
    """集中主窗口中的配对流程事件入口。

    当前阶段由控制器负责编排配对扫描、切换和加载流程，
    窗口对象仍保留 GUI 持有状态，供后续 ImageSessionController 继续收口。
    """

    def __init__(self, window: MainWindow, pair_service: PairService) -> None:
        self._window = window
        self._pair_service = pair_service

    @property
    def pair_service(self) -> PairService:
        """暴露配对服务，供其他流程复用。"""
        return self._pair_service

    def open_new_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self._window, "选择新图文件夹")
        if not folder:
            return

        files = self._pair_service.scan_new_folder(folder)
        self._window._new_folder = folder
        self._window._config.new_folder = folder

        self._window.file_list.clear()
        self._window._image_pairs = []
        self._window._current_pair_idx = -1
        self._window._current_pair_using_aligned = False
        self._window._candidates_cache.clear()

        for file_info in files:
            self._window.file_list.addItem(file_info.stem)

        if files:
            try:
                first_image = self._pair_service.read_image(files[0].path)
                self._window.set_image_data(first_image.data, None)
                self._window._new_fits_header = first_image.header
                self._window._old_fits_header = None
            except Exception as exc:
                self._window._show_message(f"加载失败: {exc}", 5000, level="ERROR")
                return

        self._window._show_message(f"已加载新图文件夹: {folder} ({len(files)} 个文件)")
        self.add_recent_folder(folder)

    def open_old_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self._window, "选择旧图文件夹")
        if not folder:
            return

        self._window._old_folder = folder
        self._window._config.old_folder = folder
        self.add_recent_folder(folder)
        old_files = self._pair_service.scan_old_folder(folder)

        if self._window._new_folder:
            pairs, only_new, only_old = self._pair_service.match_pairs(
                self._window._new_folder,
                folder,
            )
            self._window._image_pairs = pairs
            self._window._current_pair_idx = -1
            self._window._current_pair_using_aligned = False
            self._window._candidates_cache.clear()

            self._window.file_list.clear()
            for pair in pairs:
                self._window.file_list.addItem(f"✅ {pair.name}")
            for name in only_new:
                self._window.file_list.addItem(f"🆕 {name} (仅新图)")
            for name in only_old:
                self._window.file_list.addItem(f"📁 {name} (仅旧图)")

            if pairs:
                self.load_pair(0)

            self._window._show_message(
                f"已配对: {len(pairs)} 对, 仅新图: {len(only_new)}, 仅旧图: {len(only_old)}",
                5000,
            )
            return

        self._window._show_message(f"已选择旧图文件夹: {folder} ({len(old_files)} 个文件)")

    def add_recent_folder(self, folder: str) -> None:
        recent_folders = self._window._config.recent_folders
        if folder in recent_folders:
            recent_folders.remove(folder)
        recent_folders.insert(0, folder)
        max_count = self._window._config.max_recent_count
        self._window._config.recent_folders = recent_folders[:max_count]
        self.update_recent_menu()

    def update_recent_menu(self) -> None:
        self._window.menu_recent.clear()
        recent = self._window._config.recent_folders
        if not recent:
            self._window.menu_recent.addAction("(无最近打开)")
            return

        for folder in recent:
            action = self._window.menu_recent.addAction(folder)
            action.triggered.connect(
                lambda checked, current_folder=folder: self.open_recent_folder(current_folder)
            )

    def open_recent_folder(self, folder: str) -> None:
        if not Path(folder).exists():
            self._window._show_message(f"文件夹不存在: {folder}", 5000, level="WARNING")
            return

        files = self._pair_service.scan_new_folder(folder)
        self._window._new_folder = folder
        self._window._config.new_folder = folder
        self._window.file_list.clear()
        self._window._image_pairs = []
        self._window._current_pair_idx = -1
        self._window._current_pair_using_aligned = False
        self._window._candidates_cache.clear()

        for file_info in files:
            self._window.file_list.addItem(file_info.stem)

        if files:
            try:
                first_image = self._pair_service.read_image(files[0].path)
                self._window.set_image_data(first_image.data, None)
                self._window._new_fits_header = first_image.header
                self._window._old_fits_header = None
            except Exception as exc:
                self._window._show_message(f"加载失败: {exc}", 5000, level="ERROR")
                return

        self._window._show_message(f"已加载: {folder} ({len(files)} 个文件)")

    def prev_pair(self) -> None:
        current = self._window.file_list.currentRow()
        if current > 0:
            self._window.file_list.setCurrentRow(current - 1)

    def next_pair(self) -> None:
        current = self._window.file_list.currentRow()
        if current < self._window.file_list.count() - 1:
            self._window.file_list.setCurrentRow(current + 1)

    def select_pair(self, index: int) -> None:
        if index < 0 or index >= len(self._window._image_pairs):
            return
        self.load_pair(index)

    def load_pair(self, index: int) -> None:
        """加载指定索引的图像配对。"""
        if index < 0 or index >= len(self._window._image_pairs):
            return

        pair = self._window._image_pairs[index]
        self._window._current_pair_idx = index
        self._window._candidates = []
        self._window._current_candidate_idx = -1
        self._window._update_markers()
        self._window.candidate_presenter.set_candidates([])

        try:
            image_pair = self._pair_service.load_pair(pair)
            self._window._new_image_data = image_pair.new_image.data
            self._window._old_image_data = image_pair.old_image.data
            self._window._new_fits_header = image_pair.new_image.header
            self._window._old_fits_header = image_pair.old_image.header
            self._window._current_pair_using_aligned = bool(image_pair.aligned)
            if image_pair.aligned and self._window._old_image_data is not None:
                bounds = self._pair_service.calc_nonzero_valid_bounds(self._window._old_image_data)
                if bounds is not None and self._window._new_image_data is not None:
                    x0, x1, y0, y1 = bounds
                    self._window._new_image_data = self._window._new_image_data[y0:y1, x0:x1]
                    self._window._old_image_data = self._window._old_image_data[y0:y1, x0:x1]

            # 注意：已对齐的图像（__aligned_crop.fts）已经是裁剪后的结果，
            # 不需要再次调用 calc_nonzero_valid_bounds 进行裁剪，
            # 否则可能导致错误的结果。

            self._window.set_image_data(
                self._window._new_image_data,
                self._window._old_image_data,
            )

            if image_pair.aligned:
                self._window._logger.info("加载已对齐裁剪图像: %s", pair.name)

            if index in self._window._candidates_cache:
                self._window.set_candidates(self._window._candidates_cache[index])
        except Exception as exc:
            self._window._show_message(f"加载失败: {exc}", 5000, level="ERROR")

    def aligned_artifact_paths(self, pair) -> tuple[Path, Path, Path, Path]:
        return self._pair_service.aligned_artifact_paths(pair)

    def pair_has_aligned_artifacts(self, pair) -> bool:
        return self._pair_service.pair_has_aligned_artifacts(pair)

    def resolve_pair_image_paths(self, pair) -> tuple[Path, Path, bool]:
        return self._pair_service.resolve_pair_image_paths(pair)

    def calc_nonzero_valid_bounds(self, image):
        return self._pair_service.calc_nonzero_valid_bounds(image)

    def calc_overlap_crop_bounds(self, w: int, h: int, dx: float, dy: float, aligned_old=None, new_image=None):
        return self._pair_service.calc_overlap_crop_bounds(
            w=w,
            h=h,
            dx=dx,
            dy=dy,
            aligned_old=aligned_old,
            new_image=new_image,
        )
