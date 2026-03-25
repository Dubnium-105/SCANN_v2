from PyQt5.QtWidgets import QMainWindow

from scann.gui.composition import MainWindowBuilder


def test_builder_returns_expected_ui_parts(qapp):
    window = QMainWindow()

    try:
        parts = MainWindowBuilder(window).build()

        assert parts.central.main_splitter.count() == 2
        assert parts.central.main_splitter.widget(0) is parts.central.sidebar
        assert parts.menu.menu_recent.title() == "最近打开"
        assert parts.status.status_zoom.text() == "100%"
        assert parts.histogram_panel.isVisible() is False
    finally:
        window.close()


def test_builder_attach_sets_main_window_attributes(qapp):
    window = QMainWindow()

    try:
        parts = MainWindowBuilder(window).build()
        parts.attach(window)

        assert window.main_splitter is parts.central.main_splitter
        assert window.sidebar is parts.central.sidebar
        assert window.menu_recent is parts.menu.menu_recent
        assert window.status_image_type is parts.status.status_image_type
        assert window.histogram_panel is parts.histogram_panel
    finally:
        window.close()


def test_builder_hides_scheduler_placeholder_entry(qapp):
    window = QMainWindow()

    try:
        parts = MainWindowBuilder(window).build()

        assert parts.menu.act_scheduler.isVisible() is False
        assert parts.menu.act_scheduler.isEnabled() is False
    finally:
        window.close()
