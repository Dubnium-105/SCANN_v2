"""查询结果浮窗

功能:
- 显示天体查询结果 (VSX/MPC/SIMBAD/TNS/卫星)
- 无模态浮窗，可同时打开多个
- 支持文本复制
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class QueryResultPopup(QWidget):
    """查询结果浮窗 (非模态)

    用法:
        popup = QueryResultPopup(title="VSX 查询结果")
        popup.set_content("Name: V1234 Sgr\\nType: EA\\n...")
        popup.show()
    """

    def __init__(self, title: str = "查询结果", parent=None):
        super().__init__(parent, Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setWindowTitle(title)
        self.resize(400, 300)
        self.setMinimumSize(300, 200)

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # 标题
        self.lbl_title = QLabel("")
        self.lbl_title.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #2196F3;"
        )
        layout.addWidget(self.lbl_title)

        # 坐标信息
        self.lbl_coords = QLabel("")
        self.lbl_coords.setStyleSheet("color: #888;")
        layout.addWidget(self.lbl_coords)

        # 结果文本
        self.text_result = QPlainTextEdit()
        self.text_result.setReadOnly(True)
        self.text_result.setFont(QFont("Consolas", 10))
        self.text_result.setStyleSheet(
            "background-color: #1E1E1E; color: #D4D4D4; "
            "border: 1px solid #3C3C3C;"
        )
        layout.addWidget(self.text_result, 1)

        # 状态
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.lbl_status)

        # 按钮
        btn_layout = QHBoxLayout()
        btn_copy = QPushButton("📋 复制")
        btn_copy.clicked.connect(self._on_copy)
        btn_layout.addWidget(btn_copy)

        btn_layout.addStretch()

        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.close)
        btn_layout.addWidget(btn_close)

        layout.addLayout(btn_layout)

    # ── 公共 API ──

    def set_content(
        self,
        content: str,
        title: Optional[str] = None,
        coords: Optional[str] = None,
    ) -> None:
        """设置查询结果内容

        Args:
            content: 结果文本
            title: 查询类型标题 (如 "VSX 查询结果")
            coords: 坐标信息
        """
        self.text_result.setPlainText(content)
        if title:
            self.lbl_title.setText(title)
        if coords:
            self.lbl_coords.setText(coords)

    def set_loading(self, message: str = "查询中...") -> None:
        """显示加载状态"""
        self.text_result.setPlainText(message)
        self.lbl_status.setText("⏳ " + message)

    def set_error(self, error: str) -> None:
        """显示错误"""
        self.text_result.setPlainText(f"❌ 查询失败:\n{error}")
        self.lbl_status.setText("❌ 失败")

    def set_success(self, count: int = 0) -> None:
        """显示成功状态"""
        if count > 0:
            self.lbl_status.setText(f"✅ 找到 {count} 条结果")
        else:
            self.lbl_status.setText("✅ 查询完成")

    # ── 事件 ──

    def _on_copy(self) -> None:
        QApplication.clipboard().setText(self.text_result.toPlainText())
        self.lbl_status.setText("📋 已复制到剪贴板")
