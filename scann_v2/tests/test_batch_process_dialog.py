"""BatchProcessDialog 批量处理对话框 单元测试

TDD 测试:
1. 初始化 → 控件默认值
2. _on_start → 验证输入, 发射 process_started(params)
3. update_progress → 进度条 + 状态
4. processing_finished → 恢复按钮
"""

import pytest

from scann.gui.dialogs.batch_process_dialog import BatchProcessDialog


@pytest.fixture
def dialog(qapp):
    return BatchProcessDialog()


class TestBatchProcessDialogInit:
    """测试初始化"""

    def test_window_title(self, dialog):
        assert "批量" in dialog.windowTitle()

    def test_denoise_checked_by_default(self, dialog):
        assert dialog.grp_denoise.isChecked()

    def test_flat_unchecked_by_default(self, dialog):
        assert not dialog.grp_flat.isChecked()

    def test_kernel_size_default(self, dialog):
        assert dialog.spin_kernel_size.value() == 3

    def test_flat_sigma_default(self, dialog):
        assert dialog.spin_flat_sigma.value() == pytest.approx(100.0)

    def test_denoise_methods(self, dialog):
        items = [dialog.combo_denoise_method.itemText(i)
                 for i in range(dialog.combo_denoise_method.count())]
        assert "中值滤波" in items
        assert "高斯滤波" in items

    def test_bit_depth_options(self, dialog):
        assert dialog.combo_bit_depth.count() >= 2

    def test_progress_bar_hidden(self, dialog):
        assert not dialog.progress_bar.isVisible()


class TestOnStart:
    """测试开始处理"""

    def test_start_without_input_shows_warning(self, dialog):
        dialog._on_start()
        assert "⚠" in dialog.lbl_status.text()

    def test_start_emits_signal(self, dialog):
        received = []
        dialog.process_started.connect(lambda p: received.append(p))
        dialog.edit_input_dir.setText("/some/input")
        dialog._on_start()
        assert len(received) == 1
        params = received[0]
        assert params["input_dir"] == "/some/input"
        assert params["denoise"] is True
        assert "kernel_size" in params

    def test_start_disables_button(self, dialog):
        dialog.edit_input_dir.setText("/input")
        dialog._on_start()
        assert not dialog.btn_start.isEnabled()

    def test_start_includes_flat_params(self, dialog):
        received = []
        dialog.process_started.connect(lambda p: received.append(p))
        dialog.edit_input_dir.setText("/input")
        dialog.grp_flat.setChecked(True)
        dialog._on_start()
        assert received[0]["flat_field"] is True

    def test_start_includes_overwrite(self, dialog):
        received = []
        dialog.process_started.connect(lambda p: received.append(p))
        dialog.edit_input_dir.setText("/input")
        dialog.chk_overwrite.setChecked(True)
        dialog._on_start()
        assert received[0]["overwrite"] is True


class TestUpdateProgress:
    """测试进度更新"""

    def test_progress_bar_shown(self, dialog):
        dialog.show()
        dialog.update_progress(1, 10, "test.fits")
        assert dialog.progress_bar.isVisible()
        dialog.close()

    def test_progress_values(self, dialog):
        dialog.update_progress(5, 20, "image.fits")
        assert dialog.progress_bar.value() == 5
        assert dialog.progress_bar.maximum() == 20

    def test_status_text(self, dialog):
        dialog.update_progress(3, 10, "field_001.fits")
        assert "field_001.fits" in dialog.lbl_status.text()


class TestProcessingFinished:
    """测试处理完成"""

    def test_restores_button(self, dialog):
        dialog.btn_start.setEnabled(False)
        dialog.processing_finished()
        assert dialog.btn_start.isEnabled()

    def test_shows_completion(self, dialog):
        dialog.processing_finished()
        assert "✅" in dialog.lbl_status.text()
