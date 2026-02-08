"""MpcReportDialog MPC 80列报告对话框 单元测试

TDD 测试:
1. 初始化 → 空报告
2. set_report → 文本显示 + 字符统计
3. 80列验证 → 全部80列 ✅ / 非标准 ⚠
4. _on_copy → 复制到剪贴板
5. report_exported 信号
"""

import pytest
from unittest.mock import patch

from scann.gui.dialogs.mpc_report_dialog import MpcReportDialog


@pytest.fixture
def dialog(qapp):
    return MpcReportDialog()


# 标准 MPC 80列报告行 (恰好80字符)
SAMPLE_80_COL = "     2024 AB1   C2024 01 15.12345 12 34 56.78 +12 34 56.7          15.2 R      C42"
SAMPLE_SHORT = "short line"


class TestMpcReportDialogInit:
    """测试初始化"""

    def test_window_title(self, dialog):
        assert "MPC" in dialog.windowTitle() or "80" in dialog.windowTitle()

    def test_minimum_size(self, dialog):
        assert dialog.minimumWidth() >= 700

    def test_empty_report(self, dialog):
        assert dialog._report_text == ""

    def test_text_readonly(self, dialog):
        assert dialog.text_report.isReadOnly()

    def test_no_wrap(self, dialog):
        from PyQt5.QtWidgets import QPlainTextEdit
        assert dialog.text_report.lineWrapMode() == QPlainTextEdit.NoWrap


class TestSetReport:
    """测试设置报告内容"""

    def test_set_report_displays(self, dialog):
        dialog.set_report("test content")
        assert dialog.text_report.toPlainText() == "test content"

    def test_set_report_stores_text(self, dialog):
        dialog.set_report("hello")
        assert dialog._report_text == "hello"


class TestCharCount:
    """测试字符统计"""

    def test_80_col_all_pass(self, dialog):
        # 创建恰好80字符的行
        line = "A" * 80
        dialog.set_report(line)
        assert "✅" in dialog.lbl_char_count.text()
        assert "80" in dialog.lbl_char_count.text()

    def test_non_standard_width(self, dialog):
        dialog.set_report("short")
        assert "⚠" in dialog.lbl_char_count.text()

    def test_mixed_widths(self, dialog):
        lines = "A" * 80 + "\n" + "B" * 60
        dialog.set_report(lines)
        assert "⚠" in dialog.lbl_char_count.text()

    def test_empty_report_count(self, dialog):
        dialog.set_report("")
        assert "0" in dialog.lbl_char_count.text()

    def test_multiple_lines_count(self, dialog):
        lines = "\n".join(["A" * 80] * 5)
        dialog.set_report(lines)
        assert "5" in dialog.lbl_char_count.text()


class TestCopy:
    """测试复制功能"""

    def test_copy_to_clipboard(self, dialog):
        dialog.set_report("copy me")
        dialog._on_copy()
        from PyQt5.QtWidgets import QApplication
        assert QApplication.clipboard().text() == "copy me"

    def test_copy_shows_feedback(self, dialog):
        dialog.set_report("test")
        dialog._on_copy()
        assert "已复制" in dialog.lbl_char_count.text()


class TestReportExportedSignal:
    """测试 report_exported 信号"""

    def test_signal_exists(self, dialog):
        received = []
        dialog.report_exported.connect(lambda p: received.append(p))
        dialog.report_exported.emit("/test.txt")
        assert received == ["/test.txt"]
