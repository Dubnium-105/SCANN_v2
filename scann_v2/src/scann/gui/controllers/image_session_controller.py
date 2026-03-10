"""图像会话控制器。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from scann.core.astrometry import pixel_to_wcs, format_ra_hms, format_dec_dms
from scann.core.image_processor import histogram_stretch
from scann.services.blink_service import BlinkState

if TYPE_CHECKING:
    from scann.core.models import FitsHeader
    from scann.gui.main_window import MainWindow


class ImageSessionController:
    """集中主窗口中的图像显示状态与会话交互。"""

    def __init__(self, window: MainWindow) -> None:
        self._window = window

    def toggle_blink(self) -> None:
        """切换闪烁显示。"""
        running = self._window.blink_service.toggle()
        self._window.btn_blink.setChecked(running)
        if running:
            self._window.blink_timer.setInterval(self._window.blink_service.speed_ms)
            self._window.blink_timer.start()
        else:
            self._window.blink_timer.stop()

    def blink_tick(self) -> None:
        """响应闪烁定时器节拍。"""
        state = self._window.blink_service.tick()
        if state == BlinkState.NEW:
            self.show_image("new")
        else:
            self.show_image("old")

    def set_blink_speed(self, speed_ms: int) -> None:
        """更新闪烁速度。"""
        self._window.blink_service.speed_ms = speed_ms
        self._window._config.blink_speed_ms = speed_ms
        if self._window.blink_service.is_running:
            self._window.blink_timer.setInterval(speed_ms)

    def toggle_invert(self) -> None:
        """切换反色并刷新当前显示。"""
        inverted = self._window.blink_service.toggle_invert()
        self._window.btn_invert.setChecked(inverted)

        if inverted:
            self._window.overlay_inv.show_label()
        else:
            self._window.overlay_inv.hide_label()

        current = (
            "new"
            if self._window.blink_service.current_state == BlinkState.NEW
            else "old"
        )
        self.show_image(current)

    def show_new(self) -> None:
        """显示新图。"""
        self._window.btn_show_new.setChecked(True)
        self._window.btn_show_old.setChecked(False)
        self._window.blink_service.set_state(BlinkState.NEW)
        self.show_image("new")

    def show_old(self) -> None:
        """显示旧图。"""
        self._window.btn_show_new.setChecked(False)
        self._window.btn_show_old.setChecked(True)
        self._window.blink_service.set_state(BlinkState.OLD)
        self.show_image("old")

    def show_image(self, which: str) -> None:
        """统一的图像显示逻辑。"""
        if which == "new":
            data = self._window._new_image_data
            label = "NEW"
            color = "new"
        else:
            data = self._window._old_image_data
            label = "OLD"
            color = "old"

        if data is None:
            self._window.overlay_state.setText(f"无{label}")
            return

        self._window.histogram_panel.set_image_data(data)

        black = self._window.histogram_panel.black_point
        white = self._window.histogram_panel.white_point
        stretched = histogram_stretch(data, black_point=black, white_point=white)

        self._window.image_viewer.set_image_data(
            stretched,
            inverted=self._window.blink_service.is_inverted,
        )
        self._window.overlay_state.setText(label)
        self._window.overlay_state.set_state(color)
        self._window.status_image_type.setText(f"当前: {label}")

    def toggle_histogram(self) -> None:
        """切换直方图面板。"""
        visible = not self._window.histogram_panel.isVisible()
        self._window.histogram_panel.setVisible(visible)

    def stretch_changed(self, black: float, white: float) -> None:
        """响应直方图拉伸参数变化。"""
        if self._window.blink_service.current_state == BlinkState.NEW:
            data = self._window._new_image_data
        else:
            data = self._window._old_image_data

        if data is None:
            return

        stretched = histogram_stretch(data, black_point=black, white_point=white)
        self._window.image_viewer.set_image_data(
            stretched,
            inverted=self._window.blink_service.is_inverted,
        )

    def mouse_moved(self, x: int, y: int) -> None:
        """更新像素坐标和可用时的 WCS 坐标。"""
        self._window.status_pixel_coord.set_pixel_coordinates(x, y)

        header = self._window._new_fits_header
        if header is not None:
            sky = pixel_to_wcs(x, y, header)
            if sky:
                self._window.status_wcs_coord.set_wcs_coordinates(
                    format_ra_hms(sky.ra),
                    format_dec_dms(sky.dec),
                )

    def zoom_changed(self, zoom_pct: float) -> None:
        """更新缩放状态栏文本。"""
        self._window.status_zoom.setText(f"{zoom_pct:.0f}%")

    def set_image_data(
        self,
        new_data: Optional[np.ndarray],
        old_data: Optional[np.ndarray],
    ) -> None:
        """设置当前图像配对数据并刷新显示。"""
        self._window._new_image_data = new_data
        self._window._old_image_data = old_data
        self.show_new()

        if new_data is not None:
            self._window.histogram_panel.set_image_data(new_data)