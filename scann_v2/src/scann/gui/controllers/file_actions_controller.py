"""文件保存相关辅助动作控制器。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt5.QtWidgets import QFileDialog

from scann.core.fits_io import write_fits

if TYPE_CHECKING:
    from scann.gui.main_window import MainWindow


class FileActionsController:
    """集中主窗口中的图像保存辅助动作。"""

    def __init__(self, window: MainWindow) -> None:
        self._window = window

    def save_image(self) -> None:
        data = self._window._new_image_data
        if data is None:
            self._window._show_message("无图像数据可保存")
            return

        path, _ = QFileDialog.getSaveFileName(
            self._window,
            "保存图像",
            "",
            "FITS (*.fits);;所有文件 (*)",
        )
        if not path:
            return

        try:
            write_fits(
                path,
                data,
                header=self._window._new_fits_header,
            )
            self._window._show_message(f"已保存: {path}")
        except Exception as exc:
            self._window._show_message(f"保存失败: {exc}", 5000, level="ERROR")

    def save_marked_image(self) -> None:
        if self._window._new_image_data is None:
            self._window._show_message("无图像数据可保存")
            return

        path, _ = QFileDialog.getSaveFileName(
            self._window,
            "另存为标记图",
            "",
            "PNG (*.png);;FITS (*.fits)",
        )
        if not path:
            return

        try:
            pixmap = self._window.image_viewer.grab()
            pixmap.save(path)
            self._window._show_message(f"已保存标记图: {path}")
        except Exception as exc:
            self._window._show_message(f"保存失败: {exc}", 5000, level="ERROR")