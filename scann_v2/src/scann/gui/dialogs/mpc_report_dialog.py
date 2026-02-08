"""MPC 80列报告对话框

功能:
- 显示符合 MPC 80列格式的观测报告
- 预览/复制/导出 (.txt)
- 自动填充天文台编号、观测日期
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)


class MpcReportDialog(QDialog):
    """MPC 80列报告对话框

    信号:
        report_exported: 报告已导出 (文件路径)
    """

    report_exported = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MPC 80列观测报告")
        self.setMinimumSize(700, 500)

        self._report_text: str = ""
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── 说明 ──
        lbl_info = QLabel(
            "以下报告遵循 MPC 80列格式。每行恰好80个字符。\n"
            "请检查内容后复制或导出。"
        )
        lbl_info.setStyleSheet("color: #AAA;")
        layout.addWidget(lbl_info)

        # ── 报告文本 ──
        self.text_report = QPlainTextEdit()
        self.text_report.setReadOnly(True)
        self.text_report.setFont(QFont("Courier New", 11))
        self.text_report.setStyleSheet(
            "background-color: #1E1E1E; color: #D4D4D4; "
            "border: 1px solid #3C3C3C; padding: 8px;"
        )
        self.text_report.setLineWrapMode(QPlainTextEdit.NoWrap)
        layout.addWidget(self.text_report, 1)

        # ── 状态行: 字符数统计 ──
        self.lbl_char_count = QLabel("行数: 0  |  每行字符: --")
        self.lbl_char_count.setStyleSheet("color: #888;")
        layout.addWidget(self.lbl_char_count)

        # ── 按钮 ──
        btn_layout = QHBoxLayout()

        btn_copy = QPushButton("📋 复制到剪贴板")
        btn_copy.clicked.connect(self._on_copy)
        btn_layout.addWidget(btn_copy)

        btn_export = QPushButton("💾 导出为 .txt")
        btn_export.clicked.connect(self._on_export)
        btn_layout.addWidget(btn_export)

        btn_layout.addStretch()

        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.close)
        btn_layout.addWidget(btn_close)

        layout.addLayout(btn_layout)

    # ── 公共 API ──

    def set_report(self, report_text: str) -> None:
        """设置报告内容

        Args:
            report_text: MPC 80列格式的报告字符串
        """
        self._report_text = report_text
        self.text_report.setPlainText(report_text)
        self._update_char_count()

    def _update_char_count(self) -> None:
        """更新字符统计"""
        lines = self._report_text.split("\n")
        non_empty = [l for l in lines if l.strip()]
        if non_empty:
            widths = [len(l) for l in non_empty]
            self.lbl_char_count.setText(
                f"行数: {len(non_empty)}  |  "
                f"每行字符: {min(widths)}~{max(widths)}  "
                f"{'✅ 全部 80 列' if all(w == 80 for w in widths) else '⚠ 非标准宽度'}"
            )
        else:
            self.lbl_char_count.setText("行数: 0")

    # ── 事件 ──

    def _on_copy(self) -> None:
        QApplication.clipboard().setText(self._report_text)
        self.lbl_char_count.setText(self.lbl_char_count.text() + "  → 已复制!")

    def _on_export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 MPC 报告", "mpc_report.txt", "Text Files (*.txt)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._report_text)
            self.report_exported.emit(path)
            self.lbl_char_count.setText(
                self.lbl_char_count.text() + f"  → 已导出: {path}"
            )
