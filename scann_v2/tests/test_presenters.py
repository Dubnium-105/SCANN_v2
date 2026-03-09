from unittest.mock import Mock

from scann.core.models import Candidate
from scann.gui.presenters import CandidatePresenter, StatusPresenter


def test_status_presenter_show_message_updates_status_bar_and_logs() -> None:
    status_bar = Mock()
    logger = Mock()
    presenter = StatusPresenter(status_bar, logger)

    presenter.show_message("ready", timeout=1200, level="warning")

    status_bar.showMessage.assert_called_once_with("ready", 1200)
    logger.log.assert_called_once()
    log_level, message = logger.log.call_args.args
    assert message == "ready"
    assert log_level > 0


def test_candidate_presenter_sets_candidates_and_refreshes_markers() -> None:
    suspect_table = Mock()
    image_viewer = Mock()
    presenter = CandidatePresenter(suspect_table, image_viewer)
    candidates = [Candidate(x=10, y=20), Candidate(x=30, y=40)]

    presenter.set_candidates(candidates)
    presenter.refresh_markers(candidates, selected_idx=1, show_markers=False)

    suspect_table.set_candidates.assert_called_once_with(candidates)
    image_viewer.draw_markers.assert_called_once_with(
        candidates,
        selected_idx=1,
        hide_all=True,
    )