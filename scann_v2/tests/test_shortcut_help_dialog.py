"""ShortcutHelpDialog 快捷键帮助对话框 单元测试

TDD 测试:
1. 初始化 → 表格行数正确
2. 5 个分组标题
3. 快捷键内容完整
"""

import pytest

from scann.gui.dialogs.shortcut_help_dialog import ShortcutHelpDialog, SHORTCUT_GROUPS


@pytest.fixture
def dialog(qapp):
    return ShortcutHelpDialog()


class TestShortcutHelpDialogInit:
    """测试初始化"""

    def test_window_title(self, dialog):
        assert "快捷键" in dialog.windowTitle()

    def test_minimum_size(self, dialog):
        assert dialog.minimumWidth() >= 500
        assert dialog.minimumHeight() >= 500

    def test_table_exists(self, dialog):
        assert dialog.table is not None

    def test_two_columns(self, dialog):
        assert dialog.table.columnCount() == 2


class TestTableContent:
    """测试表格内容"""

    def test_total_row_count(self, dialog):
        """总行数 = 每组标题行(1) + 快捷键行"""
        expected = sum(len(shortcuts) + 1 for _, shortcuts in SHORTCUT_GROUPS)
        assert dialog.table.rowCount() == expected

    def test_five_groups(self):
        assert len(SHORTCUT_GROUPS) == 5

    def test_group_names(self):
        names = [g[0] for g in SHORTCUT_GROUPS]
        assert "图像操作" in names
        assert "候选操作" in names
        assert "界面" in names
        assert "文件" in names
        assert "检测/查询" in names

    def test_image_shortcuts_include_blink(self):
        image_group = SHORTCUT_GROUPS[0]
        keys = [k for k, _ in image_group[1]]
        assert "R" in keys  # 切换闪烁
        assert "I" in keys  # 切换反色

    def test_candidate_shortcuts_include_mark(self):
        candidate_group = SHORTCUT_GROUPS[1]
        keys = [k for k, _ in candidate_group[1]]
        assert "Y" in keys  # 标记为真
        assert "N" in keys  # 标记为假
        assert "Space" in keys  # 下一个候选

    def test_table_not_editable(self, dialog):
        from PyQt5.QtWidgets import QTableWidget
        assert dialog.table.editTriggers() == QTableWidget.NoEditTriggers

    def test_no_row_header(self, dialog):
        assert not dialog.table.verticalHeader().isVisible()

    def test_headers(self, dialog):
        h0 = dialog.table.horizontalHeaderItem(0).text()
        h1 = dialog.table.horizontalHeaderItem(1).text()
        assert "快捷键" in h0
        assert "功能" in h1


class TestGroupContent:
    """测试分组内容完整性"""

    @pytest.mark.parametrize(
        "group_idx, expected_key",
        [
            (0, "F"),       # 图像操作 → 适配窗口
            (0, "1"),       # 图像操作 → 显示新图
            (0, "2"),       # 图像操作 → 显示旧图
            (1, "←"),       # 候选操作 → 上一组配对
            (1, "→"),       # 候选操作 → 下一组配对
            (2, "Ctrl+B"),  # 界面 → 切换侧边栏
            (3, "Ctrl+O"),  # 文件 → 打开新图文件夹
            (4, "F5"),      # 检测/查询 → 批量检测
        ],
    )
    def test_key_present_in_group(self, group_idx, expected_key):
        group_name, shortcuts = SHORTCUT_GROUPS[group_idx]
        keys = [k for k, _ in shortcuts]
        assert expected_key in keys, f"'{expected_key}' 不在 '{group_name}' 分组中"
