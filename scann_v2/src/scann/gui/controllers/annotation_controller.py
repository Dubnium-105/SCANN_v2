"""标注工具入口控制器。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scann.gui.main_window import MainWindow


class AnnotationController:
    """管理标注对话框入口，保持主窗口仅做转发。"""

    def __init__(self, window: MainWindow) -> None:
        self._window = window

    def open_annotation(self) -> None:
        from scann.gui.dialogs.annotation_dialog import AnnotationDialog

        dialog = AnnotationDialog(self._window, config=self._window._config)
        self._window._annotation_dialog = dialog
        dialog.show()
        dialog.showMaximized()