from __future__ import annotations

from unittest.mock import Mock, patch

from scann.gui.controllers.training_controller import TrainingController


def test_open_worker_console_creates_and_reuses_dialog() -> None:
    window = Mock()
    controller = TrainingController(window)

    with patch("scann.gui.dialogs.worker_console_dialog.WorkerConsoleDialog") as mock_dialog_cls:
        dialog = Mock()
        mock_dialog_cls.return_value = dialog

        controller.open_worker_console()

        mock_dialog_cls.assert_called_once_with(window)
        dialog.setModal.assert_called_once_with(False)
        dialog.show.assert_called_once()
        dialog.raise_.assert_called_once()
        dialog.activateWindow.assert_called_once()

        dialog.show.reset_mock()
        dialog.raise_.reset_mock()
        dialog.activateWindow.reset_mock()

        controller.open_worker_console()

        dialog.show.assert_not_called()
        dialog.raise_.assert_called_once()
        dialog.activateWindow.assert_called_once()
