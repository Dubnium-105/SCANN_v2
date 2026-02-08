"""TrainingDialog AI 训练对话框 单元测试

TDD 测试:
1. 初始化 → 控件默认值
2. _on_start → 验证文件夹, 发射 training_started(params)
3. _on_stop → 发射 training_stopped
4. update_progress → 更新进度条和标签
5. training_finished → 恢复按钮状态
"""

import pytest
from unittest.mock import Mock

from scann.gui.dialogs.training_dialog import TrainingDialog


@pytest.fixture
def dialog(qapp):
    return TrainingDialog()


class TestTrainingDialogInit:
    """测试初始化"""

    def test_window_title(self, dialog):
        assert "训练" in dialog.windowTitle()

    def test_not_training(self, dialog):
        assert dialog._is_training is False

    def test_default_epochs(self, dialog):
        assert dialog.spin_epochs.value() == 50

    def test_default_batch(self, dialog):
        assert dialog.spin_batch.value() == 32

    def test_default_lr(self, dialog):
        assert dialog.spin_lr.value() == pytest.approx(0.001, abs=1e-6)

    def test_default_val_split(self, dialog):
        assert dialog.spin_val_split.value() == pytest.approx(0.2)

    def test_augment_checked(self, dialog):
        assert dialog.chk_augment.isChecked()

    def test_early_stop_checked(self, dialog):
        assert dialog.chk_early_stop.isChecked()

    def test_optimizer_options(self, dialog):
        items = [dialog.combo_optimizer.itemText(i) for i in range(dialog.combo_optimizer.count())]
        assert "Adam" in items
        assert "SGD" in items

    def test_backbone_options(self, dialog):
        items = [dialog.combo_backbone.itemText(i) for i in range(dialog.combo_backbone.count())]
        assert "ResNet18" in items


class TestOnStart:
    """测试开始训练"""

    def test_start_without_dirs_shows_warning(self, dialog):
        dialog._on_start()
        # 应该显示警告
        assert "⚠" in dialog.log_text.toPlainText()
        assert dialog._is_training is False

    def test_start_emits_signal(self, dialog):
        received = []
        dialog.training_started.connect(lambda p: received.append(p))
        dialog.edit_pos_dir.setText("/some/pos")
        dialog.edit_neg_dir.setText("/some/neg")
        dialog._on_start()
        assert len(received) == 1
        params = received[0]
        assert params["pos_dir"] == "/some/pos"
        assert params["neg_dir"] == "/some/neg"
        assert params["epochs"] == 50
        assert params["batch_size"] == 32

    def test_start_disables_button(self, dialog):
        dialog.edit_pos_dir.setText("/pos")
        dialog.edit_neg_dir.setText("/neg")
        dialog._on_start()
        assert not dialog.btn_start.isEnabled()
        assert dialog.btn_stop.isEnabled()

    def test_start_sets_training_flag(self, dialog):
        dialog.edit_pos_dir.setText("/pos")
        dialog.edit_neg_dir.setText("/neg")
        dialog._on_start()
        assert dialog._is_training is True


class TestOnStop:
    """测试停止训练"""

    def test_stop_emits_signal(self, dialog):
        received = []
        dialog.training_stopped.connect(lambda: received.append(True))
        dialog._on_stop()
        assert len(received) == 1

    def test_stop_restores_buttons(self, dialog):
        dialog._on_stop()
        assert dialog.btn_start.isEnabled()
        assert not dialog.btn_stop.isEnabled()

    def test_stop_clears_training_flag(self, dialog):
        dialog._is_training = True
        dialog._on_stop()
        assert dialog._is_training is False


class TestUpdateProgress:
    """测试进度更新"""

    def test_update_sets_bar(self, dialog):
        dialog.update_progress(epoch=5, total=50, loss=0.1234, val_loss=0.2345)
        assert dialog.progress_bar.value() == 5
        assert dialog.progress_bar.maximum() == 50

    def test_update_sets_label(self, dialog):
        dialog.update_progress(epoch=10, total=100, loss=0.05, val_loss=0.08)
        text = dialog.lbl_epoch_info.text()
        assert "10" in text
        assert "100" in text

    def test_update_appends_log(self, dialog):
        dialog.update_progress(epoch=1, total=10, loss=0.5, val_loss=0.6)
        assert "Epoch" in dialog.log_text.toPlainText()


class TestTrainingFinished:
    """测试训练完成"""

    def test_finished_restores_buttons(self, dialog):
        dialog._is_training = True
        dialog.btn_start.setEnabled(False)
        dialog.training_finished("/model.pth")
        assert dialog.btn_start.isEnabled()
        assert dialog._is_training is False

    def test_finished_logs_message(self, dialog):
        dialog.training_finished("/model.pth")
        assert "✅" in dialog.log_text.toPlainText()
        assert "/model.pth" in dialog.log_text.toPlainText()
