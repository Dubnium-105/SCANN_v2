"""图像会话控制器。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from scann.core.astrometry import pixel_to_wcs
from scann.core.image_processor import histogram_stretch
from scann.services.blink_service import BlinkState
from scann.services.siril_astrometry import ResolvedSkyCoordinate

if TYPE_CHECKING:
    from scann.core.models import FitsHeader
    from scann.gui.main_window import MainWindow


class ImageSessionController:
    """集中管理主窗口图像显示与会话状态。"""

    def __init__(self, window: MainWindow) -> None:
        self._window = window

    def _current_view_name(self) -> str:
        state = self._window.blink_service.current_state
        if state == BlinkState.MARKED:
            return "new_marked"
        if state == BlinkState.OLD:
            return "old"
        return "new"

    def _image_data_for_view(self, which: str) -> Optional[np.ndarray]:
        if which == "new_marked":
            return self._window._new_marked_image_data
        if which == "old":
            return self._window._old_image_data
        return self._window._new_image_data

    def _header_for_view(self, which: str) -> FitsHeader | None:
        if which == "old":
            return self._window._old_fits_header or self._window._new_fits_header
        return self._window._new_fits_header

    def _configure_blink_sequence(self) -> None:
        sequence = (
            (BlinkState.NEW, BlinkState.MARKED, BlinkState.OLD)
            if self._window._new_marked_image_data is not None
            else (BlinkState.NEW, BlinkState.OLD)
        )
        self._window.blink_service.set_sequence(sequence)
        self._window.btn_show_new_marked.setEnabled(
            self._window._new_marked_image_data is not None
        )

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
        """响应闪烁定时器。"""
        state = self._window.blink_service.tick()
        if state == BlinkState.MARKED:
            self.show_image("new_marked")
        elif state == BlinkState.OLD:
            self.show_image("old")
        else:
            self.show_image("new")

    def set_blink_speed(self, speed_ms: int) -> None:
        """更新闪烁速度。"""
        self._window.blink_service.speed_ms = speed_ms
        self._window._config.blink_speed_ms = speed_ms
        if self._window.blink_service.is_running:
            self._window.blink_timer.setInterval(speed_ms)

    def toggle_invert(self) -> None:
        """切换反色并刷新当前视图。"""
        inverted = self._window.blink_service.toggle_invert()
        self._window.btn_invert.setChecked(inverted)

        if inverted:
            self._window.overlay_inv.show_label()
        else:
            self._window.overlay_inv.hide_label()

        self.show_image(self._current_view_name())

    def show_new_marked(self) -> None:
        """显示带标记新图。"""
        if self._window._new_marked_image_data is None:
            self.show_new()
            return

        self._window.btn_show_new_marked.setChecked(True)
        self._window.btn_show_new.setChecked(False)
        self._window.btn_show_old.setChecked(False)
        self._window.blink_service.set_state(BlinkState.MARKED)
        self.show_image("new_marked")

    def show_new(self) -> None:
        """显示新图。"""
        self._window.btn_show_new_marked.setChecked(False)
        self._window.btn_show_new.setChecked(True)
        self._window.btn_show_old.setChecked(False)
        self._window.blink_service.set_state(BlinkState.NEW)
        self.show_image("new")

    def show_old(self) -> None:
        """显示旧图。"""
        self._window.btn_show_new_marked.setChecked(False)
        self._window.btn_show_new.setChecked(False)
        self._window.btn_show_old.setChecked(True)
        self._window.blink_service.set_state(BlinkState.OLD)
        self.show_image("old")

    def show_image(self, which: str) -> None:
        """统一图像显示逻辑。"""
        if which == "new_marked" and self._window._new_marked_image_data is None:
            which = "new"

        if which == "new_marked":
            data = self._window._new_marked_image_data
            label = "MARKED"
            color = "marked"
            self._window.btn_show_new_marked.setChecked(True)
            self._window.btn_show_new.setChecked(False)
            self._window.btn_show_old.setChecked(False)
        elif which == "old":
            data = self._window._old_image_data
            label = "OLD"
            color = "old"
            self._window.btn_show_new_marked.setChecked(False)
            self._window.btn_show_new.setChecked(False)
            self._window.btn_show_old.setChecked(True)
        else:
            data = self._window._new_image_data
            label = "NEW"
            color = "new"
            self._window.btn_show_new_marked.setChecked(False)
            self._window.btn_show_new.setChecked(True)
            self._window.btn_show_old.setChecked(False)

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
        data = self._image_data_for_view(self._current_view_name())
        if data is None:
            return

        stretched = histogram_stretch(data, black_point=black, white_point=white)
        self._window.image_viewer.set_image_data(
            stretched,
            inverted=self._window.blink_service.is_inverted,
        )

    def mouse_moved(self, x: int, y: int) -> None:
        """更新像素坐标和可用的 WCS 坐标。"""
        self._window.status_pixel_coord.set_pixel_coordinates(x, y)

        header = self._header_for_view(self._current_view_name())
        if header is not None:
            sky = pixel_to_wcs(x, y, header)
            if sky:
                resolved = ResolvedSkyCoordinate.from_decimal_degrees(
                    sky.ra,
                    sky.dec,
                )
                self._window.status_wcs_coord.set_coordinate_text(
                    resolved.normalized_coordinate,
                )

    def zoom_changed(self, zoom_pct: float) -> None:
        """更新缩放状态栏文本。"""
        self._window.status_zoom.setText(f"{zoom_pct:.0f}%")

    def set_image_data(
        self,
        new_data: Optional[np.ndarray],
        old_data: Optional[np.ndarray],
        new_marked_data: Optional[np.ndarray] = None,
    ) -> None:
        """设置当前图像数据并刷新显示。"""
        self._window._new_image_data = new_data
        self._window._old_image_data = old_data
        self._window._new_marked_image_data = new_marked_data
        self._configure_blink_sequence()
        self.show_new()

        if new_data is not None:
            self._window.histogram_panel.set_image_data(new_data)
