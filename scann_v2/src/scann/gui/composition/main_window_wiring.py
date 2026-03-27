"""Main window signal and shortcut wiring helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QAction

if TYPE_CHECKING:
    from scann.gui.main_window import MainWindow


class MainWindowWiring:
    """Encapsulate signal wiring and shortcut registration for MainWindow."""

    def __init__(self, window: MainWindow):
        self._window = window

    def wire(self) -> None:
        self.connect_signals()
        self.init_shortcuts()

    def connect_signals(self) -> None:
        window = self._window

        window.btn_show_new_marked.clicked.connect(window._on_show_new_marked)
        window.btn_show_new.clicked.connect(window._on_show_new)
        window.btn_show_old.clicked.connect(window._on_show_old)
        window.btn_blink.clicked.connect(window._on_blink_toggle)
        window.btn_invert.clicked.connect(window._on_invert_toggle)
        window.btn_mark_real.clicked.connect(window._on_mark_real)
        window.btn_mark_bogus.clicked.connect(window._on_mark_bogus)
        window.btn_next_candidate.clicked.connect(window._on_next_candidate)
        window.btn_histogram.clicked.connect(window._on_toggle_histogram)

        window.blink_speed.speed_changed.connect(window._on_blink_speed_changed)

        window.btn_new_folder.clicked.connect(window._on_open_dataset)
        window.btn_old_folder.clicked.connect(window._on_open_dataset)
        window.btn_align.clicked.connect(window._on_batch_align)
        window.btn_detect.clicked.connect(window._on_batch_detect)

        window.act_open_new.triggered.connect(window._on_open_dataset)
        window.act_open_old.triggered.connect(window._on_open_dataset)
        window.act_save.triggered.connect(window._on_save_image)
        window.act_save_marked.triggered.connect(window._on_save_marked_image)
        window.act_exit.triggered.connect(window.close)

        window.act_align.triggered.connect(window._on_batch_align)
        window.act_batch_process.triggered.connect(window._on_batch_process)
        window.act_histogram.triggered.connect(window._on_toggle_histogram)

        window.act_detect.triggered.connect(window._on_batch_detect)
        window.act_train.triggered.connect(window._on_open_training)
        window.act_load_model.triggered.connect(window._on_load_model)
        window.act_model_info.triggered.connect(window._on_model_info)
        window.act_annotation.triggered.connect(window._on_open_annotation)

        query_actions = {
            window.act_query_vsx: "vsx",
            window.act_query_mpc: "mpc",
            window.act_query_simbad: "simbad",
            window.act_query_tns: "tns",
            window.act_query_satellite: "satellite",
        }
        for action, query_type in query_actions.items():
            action.triggered.connect(
                lambda checked=False, current_query=query_type: window._on_menu_query(current_query)
            )
        window.act_mpc_report.triggered.connect(window._on_mpc_report)

        window.act_toggle_sidebar.triggered.connect(window.sidebar.toggle)
        window.act_fit_view.triggered.connect(window.image_viewer.fit_in_view)
        window.act_zoom_actual.triggered.connect(window._on_zoom_actual)
        window.act_zoom_in.triggered.connect(window._on_zoom_in)
        window.act_zoom_out.triggered.connect(window._on_zoom_out)
        window.act_show_markers.toggled.connect(lambda _checked: window._update_markers())
        window.act_show_mpcorb.toggled.connect(window._on_toggle_mpcorb)
        window.act_show_known.toggled.connect(window._on_toggle_known)

        window.act_preferences.triggered.connect(window._on_open_preferences)
        window.act_mpcorb_file.triggered.connect(window._on_select_mpcorb_file)
        window.act_scheduler.triggered.connect(window._on_open_scheduler)

        window.act_shortcut_help.triggered.connect(window._on_shortcut_help)
        window.act_docs.triggered.connect(window._on_open_docs)
        window.act_about.triggered.connect(window._on_about)

        window.suspect_table.candidate_selected.connect(window._on_candidate_selected)
        window.suspect_table.candidate_double_clicked.connect(window._on_candidate_double_clicked)

        window.file_list.currentRowChanged.connect(window._on_pair_selected)

        window.image_viewer.point_clicked.connect(window._on_image_clicked)
        window.image_viewer.right_click.connect(window._on_image_right_click)
        window.image_viewer.mouse_moved.connect(window._on_mouse_moved)
        window.image_viewer.zoom_changed.connect(window._on_zoom_changed)

        window.histogram_panel.stretch_changed.connect(window._on_stretch_changed)

    def init_shortcuts(self) -> None:
        window = self._window
        existing_actions = getattr(window, "_shortcut_actions", [])
        for action in existing_actions:
            window.removeAction(action)

        shortcuts: dict[str, Callable[[], None]] = {
            "R": window._on_blink_toggle,
            "I": window._on_invert_toggle,
            "Y": window._on_mark_real,
            "N": window._on_mark_bogus,
            "3": window._on_show_new_marked,
            "1": window._on_show_new,
            "2": window._on_show_old,
            "F": window.image_viewer.fit_in_view,
            "Space": window._on_next_candidate,
            "Left": window._on_prev_pair,
            "Right": window._on_next_pair,
        }

        shortcut_actions: list[QAction] = []
        for key, handler in shortcuts.items():
            action = QAction(window)
            action.setShortcut(key)
            action.setShortcutContext(Qt.WindowShortcut)
            action.triggered.connect(
                lambda checked=False, current_handler=handler: current_handler()
            )
            window.addAction(action)
            shortcut_actions.append(action)

        window._shortcut_actions = shortcut_actions
