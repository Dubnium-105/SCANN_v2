"""首选项与配置持久化控制器。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt5.QtWidgets import QFileDialog, QMessageBox

from scann.services.config_service import ConfigService

if TYPE_CHECKING:
    from PyQt5.QtGui import QCloseEvent
    from scann.gui.main_window import MainWindow


class PreferencesController:
    """处理设置对话框、运行态配置同步和关闭前保存。"""

    def __init__(self, window: MainWindow, config_service: ConfigService) -> None:
        self._window = window
        self._config_service = config_service

    def open_preferences(self) -> None:
        """打开设置对话框并在确认后持久化配置。"""
        from scann.gui.dialogs.settings_dialog import SettingsDialog

        dialog = SettingsDialog(self._window._config, parent=self._window)
        if not dialog.exec_():
            return

        try:
            self._config_service.save_config(self._window._config)
        except Exception as exc:
            self._window._logger.error(f"保存配置失败: {exc}")

        self._window.blink_service.speed_ms = self._window._config.blink_speed_ms
        self._window.model_controller.apply_runtime_config()
        self._window._show_message("设置已保存")

    def select_mpcorb_file(self) -> None:
        """选择 MPCORB 文件并立即保存路径。"""
        path, _ = QFileDialog.getOpenFileName(
            self._window,
            "选择 MPCORB 文件",
            "",
            "DAT 文件 (*.dat);;所有文件 (*)",
        )
        if not path:
            return

        self._window._config.mpcorb_path = path
        try:
            self._config_service.save_config(self._window._config)
        except Exception:
            pass

        try:
            from scann.core.mpcorb import MpcorbParser

            parser = MpcorbParser(path)
            count = parser.load()
            self._window._show_message(f"已加载 MPCORB: {count} 个小行星", 5000)
        except Exception as exc:
            self._window._show_message(f"MPCORB 加载失败: {exc}", 5000, level="ERROR")

    def save_runtime_state(self) -> None:
        """将窗口运行态同步回配置对象。"""
        config = self._window._config
        config.new_folder = self._window._new_folder
        config.old_folder = self._window._old_folder
        config.blink_speed_ms = self._window.blink_service.speed_ms

        config.stretch_black_point = self._window.histogram_panel.black_point
        config.stretch_white_point = self._window.histogram_panel.white_point
        mode_names = ["线性", "对数", "平方根", "Asinh", "自动拉伸"]
        mode_idx = self._window.histogram_panel.combo_mode.currentIndex()
        if 0 <= mode_idx < len(mode_names):
            config.stretch_mode = mode_names[mode_idx]

        config.show_markers = self._window.act_show_markers.isChecked()
        config.show_mpcorb = self._window.act_show_mpcorb.isChecked()
        config.show_known_objects = self._window.act_show_known.isChecked()
        config.histogram_visible = self._window.histogram_panel.isVisible()
        config.sidebar_collapsed = self._window.sidebar.is_collapsed
        config.sidebar_width = self._window.sidebar.preferred_width

        config.window_width = self._window.width()
        config.window_height = self._window.height()

    def restore_ui_state(self) -> None:
        """从配置恢复窗口 UI 状态。"""
        config = self._window._config
        self._window.blink_speed.speed_ms = config.blink_speed_ms
        self._window.blink_service.speed_ms = config.blink_speed_ms

        self._window.act_show_markers.setChecked(config.show_markers)
        self._window.act_show_mpcorb.setChecked(config.show_mpcorb)
        self._window.act_show_known.setChecked(config.show_known_objects)

        self._window.histogram_panel.setVisible(config.histogram_visible)

        mode_names = ["线性", "对数", "平方根", "Asinh", "自动拉伸"]
        if config.stretch_mode in mode_names:
            self._window.histogram_panel.combo_mode.setCurrentIndex(
                mode_names.index(config.stretch_mode)
            )

        self._window.sidebar.set_preferred_width(config.sidebar_width)
        if hasattr(self._window, "main_splitter"):
            self._window.main_splitter.setSizes(
                [config.sidebar_width, max(1, self._window.width() - config.sidebar_width)]
            )
        if config.sidebar_collapsed:
            self._window.sidebar.collapse()

    def handle_close_event(self, event: QCloseEvent) -> bool:
        """在窗口关闭前确认并保存配置。"""
        if self._window._config.confirm_before_close:
            reply = QMessageBox.question(
                self._window,
                "确认退出",
                "确定要退出 SCANN v2 吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return False

        self.save_runtime_state()

        try:
            self._config_service.save_config(self._window._config)
            self._window._logger.info("配置已自动保存")
        except Exception as exc:
            self._window._logger.error(f"退出时保存配置失败: {exc}")

        return True

    def handle_resize_event(self) -> None:
        """窗口大小变化后同步侧边栏和浮层位置。"""
        if self._window._config.auto_collapse_sidebar:
            self._window.sidebar.auto_collapse_check(self._window.width())

        self._window.overlay_state.move(10, 10)
        viewer_width = self._window.image_viewer.width()
        viewer_height = self._window.image_viewer.height()
        self._window.overlay_inv.move(viewer_width - 60, viewer_height - 36)
        self._window.overlay_blink.move(viewer_width - 100, viewer_height - 36)