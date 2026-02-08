"""QueryResultPopup 查询结果浮窗 单元测试

TDD 测试:
1. 初始化 → 非模态, WindowStaysOnTopHint
2. set_content → 文本/标题/坐标
3. set_loading → 加载状态
4. set_error → 错误状态
5. set_success → 成功状态
6. _on_copy → 复制文本
"""

import pytest

from PyQt5.QtCore import Qt

from scann.gui.dialogs.query_result_popup import QueryResultPopup


@pytest.fixture
def popup(qapp):
    return QueryResultPopup(title="测试查询")


class TestQueryResultPopupInit:
    """测试初始化"""

    def test_window_title(self, popup):
        assert popup.windowTitle() == "测试查询"

    def test_default_title(self, qapp):
        p = QueryResultPopup()
        assert "查询结果" in p.windowTitle()

    def test_window_flags_tool(self, popup):
        flags = popup.windowFlags()
        assert flags & Qt.Tool

    def test_window_flags_stays_on_top(self, popup):
        flags = popup.windowFlags()
        assert flags & Qt.WindowStaysOnTopHint

    def test_minimum_size(self, popup):
        assert popup.minimumWidth() >= 300
        assert popup.minimumHeight() >= 200

    def test_text_readonly(self, popup):
        assert popup.text_result.isReadOnly()


class TestSetContent:
    """测试设置内容"""

    def test_set_content_text(self, popup):
        popup.set_content("Name: V1234\nType: EA")
        assert "V1234" in popup.text_result.toPlainText()

    def test_set_content_with_title(self, popup):
        popup.set_content("data", title="VSX 结果")
        assert popup.lbl_title.text() == "VSX 结果"

    def test_set_content_with_coords(self, popup):
        popup.set_content("data", coords="RA: 12h 34m")
        assert "12h 34m" in popup.lbl_coords.text()

    def test_set_content_no_title_keeps_old(self, popup):
        popup.lbl_title.setText("old")
        popup.set_content("new data")
        assert popup.lbl_title.text() == "old"


class TestSetLoading:
    """测试加载状态"""

    def test_loading_text(self, popup):
        popup.set_loading("正在查询 VSX...")
        assert "正在查询" in popup.text_result.toPlainText()

    def test_loading_status(self, popup):
        popup.set_loading()
        assert "⏳" in popup.lbl_status.text()


class TestSetError:
    """测试错误状态"""

    def test_error_text(self, popup):
        popup.set_error("网络超时")
        assert "网络超时" in popup.text_result.toPlainText()
        assert "❌" in popup.text_result.toPlainText()

    def test_error_status(self, popup):
        popup.set_error("fail")
        assert "❌" in popup.lbl_status.text()


class TestSetSuccess:
    """测试成功状态"""

    def test_success_with_count(self, popup):
        popup.set_success(count=5)
        assert "5" in popup.lbl_status.text()
        assert "✅" in popup.lbl_status.text()

    def test_success_no_count(self, popup):
        popup.set_success()
        assert "✅" in popup.lbl_status.text()


class TestCopy:
    """测试复制"""

    def test_copy_to_clipboard(self, popup):
        popup.set_content("copy this text")
        popup._on_copy()
        from PyQt5.QtWidgets import QApplication
        assert QApplication.clipboard().text() == "copy this text"

    def test_copy_shows_feedback(self, popup):
        popup.set_content("test")
        popup._on_copy()
        assert "剪贴板" in popup.lbl_status.text()
