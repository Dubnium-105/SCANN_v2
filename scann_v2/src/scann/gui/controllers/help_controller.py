"""帮助与说明类辅助动作控制器。"""

from __future__ import annotations

from typing import TYPE_CHECKING
import webbrowser

from PyQt5.QtWidgets import QMessageBox

if TYPE_CHECKING:
    from scann.gui.main_window import MainWindow


class HelpController:
    """集中帮助菜单和轻量占位动作。"""

    DOCS_URL = "https://github.com/Dubnium-105/SCANN_v2/wiki"

    def __init__(self, window: MainWindow) -> None:
        self._window = window

    def open_shortcut_help(self) -> None:
        from scann.gui.dialogs.shortcut_help_dialog import ShortcutHelpDialog

        dialog = ShortcutHelpDialog(self._window)
        dialog.exec_()

    def open_docs(self) -> None:
        webbrowser.open(self.DOCS_URL)

    def open_about(self) -> None:
        QMessageBox.about(
            self._window,
            "关于 SCANN v2",
            "<h3>SCANN v2</h3>"
            "<p>Star/Source Classification and Analysis Neural Network</p>"
            "<p>版本: 2.0.0-dev</p>"
            "<p>基于深度学习的天文瞬变源自动检测工具</p>",
        )

    def open_scheduler(self) -> None:
        self._window._show_message("当前版本未提供计划任务功能")
