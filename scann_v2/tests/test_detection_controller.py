"""DetectionController 骨架测试。"""

from unittest.mock import Mock

from scann.gui.controllers import DetectionController


def _make_window() -> Mock:
    window = Mock()
    window._on_batch_align_impl = Mock()
    window._on_batch_process_impl = Mock()
    window._run_batch_process_impl = Mock()
    window._build_detection_params_impl = Mock(return_value="params")
    window._on_batch_detect_impl = Mock()
    window._on_mark_real_impl = Mock()
    window._on_mark_bogus_impl = Mock()
    window._on_next_candidate_impl = Mock()
    window._on_candidate_selected_impl = Mock()
    window._on_candidate_double_clicked_impl = Mock()
    window._focus_candidate_impl = Mock()
    return window


def test_batch_actions_delegate_to_window_impls() -> None:
    window = _make_window()
    controller = DetectionController(window)

    controller.batch_align()
    controller.batch_process()
    controller.run_batch_process({"input_dir": "/tmp/in"})
    controller.batch_detect()

    window._on_batch_align_impl.assert_called_once_with()
    window._on_batch_process_impl.assert_called_once_with()
    window._run_batch_process_impl.assert_called_once_with({"input_dir": "/tmp/in"})
    window._on_batch_detect_impl.assert_called_once_with()


def test_candidate_actions_delegate_to_window_impls() -> None:
    window = _make_window()
    controller = DetectionController(window)

    controller.mark_real()
    controller.mark_bogus()
    controller.next_candidate()
    controller.candidate_selected(2)
    controller.candidate_double_clicked(3)
    controller.focus_candidate(4)

    window._on_mark_real_impl.assert_called_once_with()
    window._on_mark_bogus_impl.assert_called_once_with()
    window._on_next_candidate_impl.assert_called_once_with()
    window._on_candidate_selected_impl.assert_called_once_with(2)
    window._on_candidate_double_clicked_impl.assert_called_once_with(3)
    window._focus_candidate_impl.assert_called_once_with(4)


def test_build_detection_params_delegates_return_value() -> None:
    window = _make_window()
    controller = DetectionController(window)

    result = controller.build_detection_params()

    assert result == "params"
    window._build_detection_params_impl.assert_called_once_with()