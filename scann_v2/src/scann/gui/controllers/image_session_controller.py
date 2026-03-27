"""Image session controller."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from scann.core.astrometry import pixel_to_wcs
from scann.core.brightness_match import (
    compute_brightness_match_interval,
    infer_match_positions_from_target_interval,
)
from scann.core.image_processor import histogram_stretch
from scann.services.blink_service import BlinkState
from scann.services.siril_astrometry import ResolvedSkyCoordinate

if TYPE_CHECKING:
    from scann.core.models import FitsHeader
    from scann.gui.main_window import MainWindow


class ImageSessionController:
    """Manage main-window image display and per-view stretch state."""

    _MATCH_MAX_SAMPLES = 200000
    _MATCH_HIGH_PERCENTILE = 99.9
    _MATCH_HIGHLIGHT_SIGMA = 5.0
    _MATCH_BACKGROUND_POSITION = 0.10
    _MATCH_HIGHLIGHT_POSITION = 0.98
    _MATCH_ADAPTIVE_HIGH_PERCENTILE = False

    def __init__(self, window: MainWindow) -> None:
        self._window = window
        if not hasattr(self._window, "_stretch_state_by_view"):
            self._window._stretch_state_by_view = {}

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

    def _build_stretch_state(self, data: np.ndarray, black: float, white: float) -> dict[str, float]:
        finite = np.asarray(data, dtype=np.float32)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return {
                "range_min": 0.0,
                "range_max": 1.0,
                "black_point": 0.0,
                "white_point": 1.0,
            }

        range_min = float(np.min(finite))
        range_max = float(np.max(finite))
        safe_black = float(min(black, white))
        safe_white = float(max(black, white))
        if safe_white <= safe_black:
            if range_max > range_min:
                safe_black = range_min
                safe_white = range_max
            else:
                safe_black = range_min
                safe_white = range_min + 1.0
        return {
            "range_min": range_min,
            "range_max": range_max,
            "black_point": safe_black,
            "white_point": safe_white,
        }

    def _compute_default_match_stretch_state(self, data: np.ndarray) -> dict[str, float]:
        try:
            interval = compute_brightness_match_interval(
                data,
                max_samples=self._MATCH_MAX_SAMPLES,
                high_percentile=self._MATCH_HIGH_PERCENTILE,
                highlight_sigma=self._MATCH_HIGHLIGHT_SIGMA,
                background_position=self._MATCH_BACKGROUND_POSITION,
                highlight_position=self._MATCH_HIGHLIGHT_POSITION,
                adaptive_high_percentile=self._MATCH_ADAPTIVE_HIGH_PERCENTILE,
            )
            return self._build_stretch_state(data, interval.display_min, interval.display_max)
        except Exception:
            finite = np.asarray(data, dtype=np.float32)
            finite = finite[np.isfinite(finite)]
            if finite.size == 0:
                return {
                    "range_min": 0.0,
                    "range_max": 1.0,
                    "black_point": 0.0,
                    "white_point": 1.0,
                }
            return self._build_stretch_state(data, float(np.min(finite)), float(np.max(finite)))

    def _state_for_view(self, which: str, data: np.ndarray) -> dict[str, float]:
        state = self._window._stretch_state_by_view.get(which)
        if state is None:
            state = self._compute_default_match_stretch_state(data)
            self._window._stretch_state_by_view[which] = state
        return state

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
        running = self._window.blink_service.toggle()
        self._window.btn_blink.setChecked(running)
        if running:
            self._window.blink_timer.setInterval(self._window.blink_service.speed_ms)
            self._window.blink_timer.start()
        else:
            self._window.blink_timer.stop()

    def blink_tick(self) -> None:
        state = self._window.blink_service.tick()
        if state == BlinkState.MARKED:
            self.show_image("new_marked")
        elif state == BlinkState.OLD:
            self.show_image("old")
        else:
            self.show_image("new")

    def set_blink_speed(self, speed_ms: int) -> None:
        self._window.blink_service.speed_ms = speed_ms
        self._window._config.blink_speed_ms = speed_ms
        if self._window.blink_service.is_running:
            self._window.blink_timer.setInterval(speed_ms)

    def toggle_invert(self) -> None:
        inverted = self._window.blink_service.toggle_invert()
        self._window.btn_invert.setChecked(inverted)

        if inverted:
            self._window.overlay_inv.show_label()
        else:
            self._window.overlay_inv.hide_label()

        self.show_image(self._current_view_name())

    def show_new_marked(self) -> None:
        if self._window._new_marked_image_data is None:
            self.show_new()
            return

        self._window.btn_show_new_marked.setChecked(True)
        self._window.btn_show_new.setChecked(False)
        self._window.btn_show_old.setChecked(False)
        self._window.blink_service.set_state(BlinkState.MARKED)
        self.show_image("new_marked")

    def show_new(self) -> None:
        self._window.btn_show_new_marked.setChecked(False)
        self._window.btn_show_new.setChecked(True)
        self._window.btn_show_old.setChecked(False)
        self._window.blink_service.set_state(BlinkState.NEW)
        self.show_image("new")

    def show_old(self) -> None:
        self._window.btn_show_new_marked.setChecked(False)
        self._window.btn_show_new.setChecked(False)
        self._window.btn_show_old.setChecked(True)
        self._window.blink_service.set_state(BlinkState.OLD)
        self.show_image("old")

    def show_image(self, which: str) -> None:
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

        state = self._state_for_view(which, data)
        self._window.histogram_panel.set_image_data(
            data,
            black_point=state["black_point"],
            white_point=state["white_point"],
        )

        stretched = histogram_stretch(
            data,
            black_point=state["black_point"],
            white_point=state["white_point"],
        )
        self._window.image_viewer.set_image_data(
            stretched,
            inverted=self._window.blink_service.is_inverted,
        )
        self._window.overlay_state.setText(label)
        self._window.overlay_state.set_state(color)
        self._window.status_image_type.setText(f"当前: {label}")

    def toggle_histogram(self) -> None:
        visible = not self._window.histogram_panel.isVisible()
        self._window.histogram_panel.setVisible(visible)

    def stretch_changed(self, black: float, white: float) -> None:
        current_view = self._current_view_name()
        data = self._image_data_for_view(current_view)
        if data is None:
            return

        self._window._stretch_state_by_view[current_view] = self._build_stretch_state(data, black, white)
        stretched = histogram_stretch(data, black_point=black, white_point=white)
        self._window.image_viewer.set_image_data(
            stretched,
            inverted=self._window.blink_service.is_inverted,
        )

    def match_current_stretch_to_other_views(self) -> None:
        current_view = self._current_view_name()
        data = self._image_data_for_view(current_view)
        if data is None:
            return

        state = self._window._stretch_state_by_view.get(current_view)
        if state is None:
            state = self._compute_default_match_stretch_state(data)
            self._window._stretch_state_by_view[current_view] = state

        try:
            inferred = infer_match_positions_from_target_interval(
                data,
                target_min=state["black_point"],
                target_max=state["white_point"],
                max_samples=self._MATCH_MAX_SAMPLES,
                high_percentile=self._MATCH_HIGH_PERCENTILE,
                highlight_sigma=self._MATCH_HIGHLIGHT_SIGMA,
                adaptive_high_percentile=self._MATCH_ADAPTIVE_HIGH_PERCENTILE,
            )
        except Exception as exc:
            self._window._show_message(f"亮度匹配失败: {exc}", level="WARNING")
            return

        updated_views: list[str] = []
        for view in ("new", "new_marked", "old"):
            if view == current_view:
                continue
            other_data = self._image_data_for_view(view)
            if other_data is None:
                continue
            try:
                interval = compute_brightness_match_interval(
                    other_data,
                    max_samples=self._MATCH_MAX_SAMPLES,
                    high_percentile=self._MATCH_HIGH_PERCENTILE,
                    highlight_sigma=self._MATCH_HIGHLIGHT_SIGMA,
                    background_position=inferred.background_position,
                    highlight_position=inferred.highlight_position,
                    adaptive_high_percentile=self._MATCH_ADAPTIVE_HIGH_PERCENTILE,
                )
            except Exception:
                continue

            self._window._stretch_state_by_view[view] = self._build_stretch_state(
                other_data,
                interval.display_min,
                interval.display_max,
            )
            updated_views.append(view)

        self.show_image(current_view)
        if updated_views:
            self._window._show_message(
                f"已将当前亮度匹配同步到: {', '.join(updated_views)}",
                level="INFO",
            )

    def mouse_moved(self, x: int, y: int) -> None:
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
        self._window.status_zoom.setText(f"{zoom_pct:.0f}%")

    def set_image_data(
        self,
        new_data: Optional[np.ndarray],
        old_data: Optional[np.ndarray],
        new_marked_data: Optional[np.ndarray] = None,
    ) -> None:
        self._window._new_image_data = new_data
        self._window._old_image_data = old_data
        self._window._new_marked_image_data = new_marked_data
        self._window._stretch_state_by_view = {}
        self._configure_blink_sequence()
        self.show_new()

        if new_data is not None:
            self._window.histogram_panel.set_image_data(new_data)
