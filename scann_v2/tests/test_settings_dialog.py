"""SettingsDialog 首选项设置对话框 单元测试

TDD 测试:
1. 初始化 → 5 个标签页
2. _load_from_config → 从 Config 加载
3. _save_to_config → 写回 Config
4. settings_changed 信号
5. 各标签页控件存在
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from PyQt5.QtWidgets import QDialogButtonBox

from scann.core.models import AppConfig, ObservatoryConfig, TelescopeConfig


@pytest.fixture
def mock_config():
    """创建真实的 AppConfig 对象用于测试"""
    cfg = AppConfig()
    cfg.observatory = ObservatoryConfig(code="C42", name="Test Obs")
    cfg.thresh = 5
    cfg.min_area = 3
    cfg.ai_confidence = 0.5
    return cfg


@pytest.fixture
def dialog(qapp, mock_config):
    """创建 SettingsDialog 实例"""
    from scann.gui.dialogs.settings_dialog import SettingsDialog
    dlg = SettingsDialog(mock_config)
    return dlg


class TestSettingsDialogInit:
    """测试初始化"""

    def test_window_title(self, dialog):
        assert "首选项" in dialog.windowTitle() or "设置" in dialog.windowTitle()

    def test_minimum_size(self, dialog):
        assert dialog.minimumWidth() >= 600
        assert dialog.minimumHeight() >= 500

    def test_has_tabs(self, dialog):
        assert dialog.tabs is not None

    def test_five_tabs(self, dialog):
        assert dialog.tabs.count() == 5

    def test_config_stored(self, dialog, mock_config):
        assert dialog.config is mock_config


class TestObservatoryTab:
    """测试望远镜/天文台标签页"""

    def test_obs_code_widget(self, dialog):
        assert dialog.edit_obs_code is not None

    def test_obs_name_widget(self, dialog):
        assert dialog.edit_obs_name is not None

    def test_lat_spin(self, dialog):
        assert dialog.spin_lat.minimum() == -90
        assert dialog.spin_lat.maximum() == 90

    def test_lon_spin(self, dialog):
        assert dialog.spin_lon.minimum() == -180
        assert dialog.spin_lon.maximum() == 180

    def test_alt_spin(self, dialog):
        assert dialog.spin_alt.maximum() == 10000

    def test_pixel_scale_spin(self, dialog):
        assert dialog.spin_pixel_scale.minimum() == 0.01


class TestDetectionTab:
    """测试检测参数标签页"""

    def test_sigma_spin(self, dialog):
        assert dialog.spin_sigma.minimum() == 1.0
        assert dialog.spin_sigma.maximum() == 20.0

    def test_min_area_spin(self, dialog):
        assert dialog.spin_min_area.minimum() == 1

    def test_nms_radius_spin(self, dialog):
        assert dialog.spin_nms_radius is not None


class TestAITab:
    """测试 AI 模型标签页"""

    def test_model_path_widget(self, dialog):
        assert dialog.edit_model_path.isReadOnly()

    def test_confidence_spin(self, dialog):
        assert dialog.spin_confidence.minimum() == 0.0
        assert dialog.spin_confidence.maximum() == 1.0

    def test_device_combo(self, dialog):
        items = [dialog.combo_device.itemText(i) for i in range(dialog.combo_device.count())]
        assert "auto" in items
        assert "cpu" in items
        assert "cuda" in items


class TestPathsTab:
    """测试保存/路径标签页"""

    def test_save_dir_widget(self, dialog):
        assert dialog.edit_save_dir is not None

    def test_mpcorb_path_widget(self, dialog):
        assert dialog.edit_mpcorb_path is not None

    def test_save_format_combo(self, dialog):
        assert dialog.combo_save_format.count() >= 2


class TestAdvancedTab:
    """测试高级标签页"""

    def test_max_threads(self, dialog):
        assert dialog.spin_max_threads.minimum() == 1
        assert dialog.spin_max_threads.maximum() == 32

    def test_auto_collapse_default(self, dialog):
        assert dialog.chk_auto_collapse.isChecked()


class TestLoadFromConfig:
    """测试从 Config 加载"""

    def test_obs_code_loaded(self, dialog, mock_config):
        assert dialog.edit_obs_code.text() == "C42"

    def test_obs_name_loaded(self, dialog, mock_config):
        assert dialog.edit_obs_name.text() == "Test Obs"

    def test_sigma_loaded(self, dialog, mock_config):
        assert dialog.spin_sigma.value() == 5.0


class TestSaveToConfig:
    """测试写回 Config"""

    def test_save_updates_config(self, dialog, mock_config):
        dialog.edit_obs_code.setText("X99")
        dialog._save_to_config()
        assert mock_config.observatory.code == "X99"

    def test_save_updates_ai_confidence(self, dialog, mock_config):
        dialog.spin_confidence.setValue(0.75)
        dialog._save_to_config()
        assert mock_config.ai_confidence == 0.75


class TestSettingsChangedSignal:
    """测试 settings_changed 信号"""

    def test_on_apply_emits_signal(self, dialog):
        received = []
        dialog.settings_changed.connect(lambda: received.append(True))
        dialog._on_apply()
        assert len(received) == 1
