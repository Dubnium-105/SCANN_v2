from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow

from scann.gui.composition import MainWindowWiring


class DummySignal:
    def __init__(self) -> None:
        self._callbacks = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self, *args, **kwargs) -> None:
        for callback in list(self._callbacks):
            callback(*args, **kwargs)


class DummyButton:
    def __init__(self) -> None:
        self.clicked = DummySignal()


class DummyAction:
    def __init__(self) -> None:
        self.triggered = DummySignal()
        self.toggled = DummySignal()


class DummyBlinkSpeed:
    def __init__(self) -> None:
        self.speed_changed = DummySignal()


class DummySuspectTable:
    def __init__(self) -> None:
        self.candidate_selected = DummySignal()
        self.candidate_double_clicked = DummySignal()


class DummyFileList:
    def __init__(self) -> None:
        self.currentRowChanged = DummySignal()


class DummyImageViewer:
    def __init__(self) -> None:
        self.point_clicked = DummySignal()
        self.right_click = DummySignal()
        self.mouse_moved = DummySignal()
        self.zoom_changed = DummySignal()
        self.fit_in_view = Mock()


class DummyHistogramPanel:
    def __init__(self) -> None:
        self.stretch_changed = DummySignal()
        self.apply_match_requested = DummySignal()


class WiringWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.btn_show_new_marked = DummyButton()
        self.btn_show_new = DummyButton()
        self.btn_show_old = DummyButton()
        self.btn_blink = DummyButton()
        self.btn_invert = DummyButton()
        self.btn_mark_real = DummyButton()
        self.btn_mark_bogus = DummyButton()
        self.btn_next_candidate = DummyButton()
        self.btn_histogram = DummyButton()
        self.btn_new_folder = DummyButton()
        self.btn_old_folder = DummyButton()
        self.btn_align = DummyButton()
        self.btn_detect = DummyButton()

        self.blink_speed = DummyBlinkSpeed()
        self.suspect_table = DummySuspectTable()
        self.file_list = DummyFileList()
        self.image_viewer = DummyImageViewer()
        self.histogram_panel = DummyHistogramPanel()
        self.sidebar = SimpleNamespace(toggle=Mock())

        action_names = [
            "act_open_new",
            "act_open_old",
            "act_save",
            "act_save_marked",
            "act_exit",
            "act_align",
            "act_batch_process",
            "act_histogram",
            "act_detect",
            "act_train",
            "act_load_model",
            "act_model_info",
            "act_annotation",
            "act_query_vsx",
            "act_query_mpc",
            "act_query_simbad",
            "act_query_tns",
            "act_query_satellite",
            "act_mpc_report",
            "act_toggle_sidebar",
            "act_fit_view",
            "act_zoom_actual",
            "act_zoom_in",
            "act_zoom_out",
            "act_show_markers",
            "act_show_mpcorb",
            "act_show_known",
            "act_preferences",
            "act_mpcorb_file",
            "act_scheduler",
            "act_shortcut_help",
            "act_docs",
            "act_about",
        ]
        for name in action_names:
            setattr(self, name, DummyAction())

        method_names = [
            "_on_show_new",
            "_on_show_new_marked",
            "_on_show_old",
            "_on_blink_toggle",
            "_on_invert_toggle",
            "_on_mark_real",
            "_on_mark_bogus",
            "_on_next_candidate",
            "_on_toggle_histogram",
            "_on_blink_speed_changed",
            "_on_open_dataset",
            "_on_open_new_folder",
            "_on_open_old_folder",
            "_on_save_image",
            "_on_save_marked_image",
            "_on_batch_align",
            "_on_batch_process",
            "_on_batch_detect",
            "_on_open_training",
            "_on_load_model",
            "_on_model_info",
            "_on_open_annotation",
            "_on_menu_query",
            "_on_mpc_report",
            "_on_zoom_actual",
            "_on_zoom_in",
            "_on_zoom_out",
            "_update_markers",
            "_on_toggle_mpcorb",
            "_on_toggle_known",
            "_on_open_preferences",
            "_on_select_mpcorb_file",
            "_on_open_scheduler",
            "_on_shortcut_help",
            "_on_open_docs",
            "_on_about",
            "_on_candidate_selected",
            "_on_candidate_double_clicked",
            "_on_pair_selected",
            "_on_image_clicked",
            "_on_image_right_click",
            "_on_mouse_moved",
            "_on_zoom_changed",
            "_on_stretch_changed",
            "_on_match_current_stretch_to_other_views",
            "_on_prev_pair",
            "_on_next_pair",
        ]
        for name in method_names:
            setattr(self, name, Mock())

        self.close = Mock()


def test_connect_signals_wires_controls_and_menus(qapp):
    window = WiringWindow()
    wiring = MainWindowWiring(window)

    try:
        wiring.connect_signals()

        window.btn_show_new_marked.clicked.emit()
        window.btn_show_new.clicked.emit()
        window.btn_align.clicked.emit()
        window.btn_new_folder.clicked.emit()
        window.act_open_new.triggered.emit()
        window.act_save_marked.triggered.emit()
        window.act_exit.triggered.emit()
        window.act_histogram.triggered.emit()

        window._on_show_new_marked.assert_called_once_with()
        window._on_show_new.assert_called_once_with()
        window._on_batch_align.assert_called_once_with()
        assert window._on_open_dataset.call_count == 2
        window._on_save_marked_image.assert_called_once_with()
        window.close.assert_called_once_with()
        window._on_toggle_histogram.assert_called_once_with()
    finally:
        window.close()


def test_connect_signals_wires_queries_view_and_widget_events(qapp):
    window = WiringWindow()
    wiring = MainWindowWiring(window)

    try:
        wiring.connect_signals()

        window.act_query_vsx.triggered.emit()
        window.act_query_satellite.triggered.emit()
        window.act_show_markers.toggled.emit(True)
        window.act_show_mpcorb.toggled.emit(False)
        window.act_shortcut_help.triggered.emit()
        window.suspect_table.candidate_selected.emit(5)
        window.file_list.currentRowChanged.emit(3)
        window.image_viewer.mouse_moved.emit(10, 20)
        window.histogram_panel.stretch_changed.emit(0.1, 0.9)
        window.histogram_panel.apply_match_requested.emit()

        assert window._on_menu_query.call_args_list == [(("vsx",), {}), (("satellite",), {})]
        window._update_markers.assert_called_once_with()
        window._on_toggle_mpcorb.assert_called_once_with(False)
        window._on_shortcut_help.assert_called_once_with()
        window._on_candidate_selected.assert_called_once_with(5)
        window._on_pair_selected.assert_called_once_with(3)
        window._on_mouse_moved.assert_called_once_with(10, 20)
        window._on_stretch_changed.assert_called_once_with(0.1, 0.9)
        window._on_match_current_stretch_to_other_views.assert_called_once_with()
    finally:
        window.close()


def test_init_shortcuts_registers_window_scoped_actions(qapp):
    window = WiringWindow()
    wiring = MainWindowWiring(window)

    try:
        wiring.init_shortcuts()

        assert len(window._shortcut_actions) == 12

        actions_by_shortcut = {
            action.shortcut().toString(): action for action in window._shortcut_actions
        }
        assert set(actions_by_shortcut) == {
            "R",
            "I",
            "Y",
            "N",
            "M",
            "3",
            "1",
            "2",
            "F",
            "Space",
            "Left",
            "Right",
        }

        for action in window._shortcut_actions:
            assert action.shortcutContext() == Qt.WindowShortcut

        actions_by_shortcut["R"].trigger()
        actions_by_shortcut["F"].trigger()
        actions_by_shortcut["M"].trigger()
        actions_by_shortcut["Left"].trigger()

        window._on_blink_toggle.assert_called_once_with()
        window.image_viewer.fit_in_view.assert_called_once_with()
        window._on_match_current_stretch_to_other_views.assert_called_once_with()
        window._on_prev_pair.assert_called_once_with()
    finally:
        window.close()
