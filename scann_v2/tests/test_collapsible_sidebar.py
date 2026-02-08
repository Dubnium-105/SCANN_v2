"""CollapsibleSidebar 可折叠侧边栏 单元测试

TDD 测试:
1. 初始化 → 展开状态, 240px 宽度
2. toggle → 折叠/展开切换
3. collapse/expand → 状态变化
4. auto_collapse_check → 窗口宽度 < 1200 折叠, >= 1200 展开
5. content_layout → 可访问布局
6. collapsed_changed 信号
"""

import pytest

from scann.gui.widgets.collapsible_sidebar import CollapsibleSidebar


@pytest.fixture
def sidebar(qapp):
    """创建 CollapsibleSidebar 实例"""
    return CollapsibleSidebar()


class TestCollapsibleSidebarInit:
    """测试初始化"""

    def test_not_collapsed(self, sidebar):
        assert sidebar.is_collapsed is False

    def test_default_width(self, sidebar):
        assert sidebar.DEFAULT_WIDTH == 240

    def test_min_width(self, sidebar):
        assert sidebar.MIN_WIDTH == 200

    def test_content_layout_accessible(self, sidebar):
        layout = sidebar.content_layout
        assert layout is not None


class TestCollapseExpand:
    """测试折叠/展开"""

    def test_collapse_sets_state(self, sidebar):
        sidebar.collapse()
        assert sidebar.is_collapsed is True

    def test_expand_sets_state(self, sidebar):
        sidebar.collapse()
        sidebar.expand()
        assert sidebar.is_collapsed is False

    def test_double_collapse_no_op(self, sidebar):
        sidebar.collapse()
        sidebar.collapse()  # 不应再次触发动画
        assert sidebar.is_collapsed is True

    def test_double_expand_no_op(self, sidebar):
        sidebar.expand()  # 已经展开
        assert sidebar.is_collapsed is False

    def test_toggle_from_expanded(self, sidebar):
        sidebar.toggle()
        assert sidebar.is_collapsed is True

    def test_toggle_from_collapsed(self, sidebar):
        sidebar.collapse()
        sidebar.toggle()
        assert sidebar.is_collapsed is False

    def test_toggle_twice_returns(self, sidebar):
        sidebar.toggle()
        sidebar.toggle()
        assert sidebar.is_collapsed is False


class TestAutoCollapse:
    """测试自动折叠"""

    def test_narrow_window_collapses(self, sidebar):
        sidebar.auto_collapse_check(1000)
        assert sidebar.is_collapsed is True

    def test_wide_window_stays_expanded(self, sidebar):
        sidebar.auto_collapse_check(1400)
        assert sidebar.is_collapsed is False

    def test_boundary_1200_stays_expanded(self, sidebar):
        sidebar.auto_collapse_check(1200)
        assert sidebar.is_collapsed is False

    def test_narrow_then_wide_expands(self, sidebar):
        sidebar.auto_collapse_check(800)
        assert sidebar.is_collapsed is True
        sidebar.auto_collapse_check(1400)
        assert sidebar.is_collapsed is False

    def test_wide_then_narrow_collapses(self, sidebar):
        sidebar.auto_collapse_check(1500)
        assert sidebar.is_collapsed is False
        sidebar.auto_collapse_check(900)
        assert sidebar.is_collapsed is True


class TestCollapsedChangedSignal:
    """测试 collapsed_changed 信号"""

    def test_collapse_emits_true(self, sidebar):
        received = []
        sidebar.collapsed_changed.connect(lambda v: received.append(v))
        sidebar.collapse()
        assert received == [True]

    def test_expand_emits_false(self, sidebar):
        sidebar.collapse()
        received = []
        sidebar.collapsed_changed.connect(lambda v: received.append(v))
        sidebar.expand()
        assert received == [False]

    def test_toggle_emits_signal(self, sidebar):
        received = []
        sidebar.collapsed_changed.connect(lambda v: received.append(v))
        sidebar.toggle()
        assert received == [True]
