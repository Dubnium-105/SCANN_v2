"""NoScrollSpinBox Widget 单元测试

需求: 所有输入窗口禁用滚轮调整数字大小
"""

import pytest


class TestNoScrollSpinBox:
    """测试禁用滚轮的 SpinBox"""

    def test_import(self):
        from scann.gui.widgets.no_scroll_spinbox import (
            NoScrollDoubleSpinBox,
            NoScrollSpinBox,
        )
        assert NoScrollSpinBox is not None
        assert NoScrollDoubleSpinBox is not None

    def test_instantiate(self):
        """需要 QApplication 实例才能测试 Qt widget"""
        PyQt5 = pytest.importorskip("PyQt5")
        from PyQt5.QtWidgets import QApplication
        import sys

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        from scann.gui.widgets.no_scroll_spinbox import NoScrollSpinBox

        sb = NoScrollSpinBox()
        sb.setValue(10)
        assert sb.value() == 10
