"""快捷键帮助对话框

显示所有可用快捷键的列表。
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


# 快捷键定义 (分组)
SHORTCUT_GROUPS = [
    (
        "图像操作",
        [
            ("1", "显示新图"),
            ("2", "显示旧图"),
            ("R", "切换闪烁"),
            ("I", "切换反色"),
            ("F", "适配窗口 (Fit)"),
            ("Ctrl+0", "实际大小"),
            ("Ctrl++", "放大"),
            ("Ctrl+-", "缩小"),
            ("滚轮", "缩放"),
            ("中键拖拽", "拖动图像"),
        ],
    ),
    (
        "候选操作",
        [
            ("Y", "标记为真目标"),
            ("N", "标记为假目标"),
            ("Space", "下一个候选体"),
            ("←", "上一组配对"),
            ("→", "下一组配对"),
        ],
    ),
    (
        "界面",
        [
            ("Ctrl+B", "切换侧边栏"),
            ("Ctrl+,", "首选项设置"),
        ],
    ),
    (
        "文件",
        [
            ("Ctrl+O", "打开新图文件夹"),
            ("Ctrl+Shift+O", "打开旧图文件夹"),
            ("Ctrl+S", "保存当前图像"),
            ("Ctrl+Shift+S", "另存为标记图"),
        ],
    ),
    (
        "检测/查询",
        [
            ("F5", "批量检测"),
            ("Ctrl+E", "生成 MPC 报告"),
            ("右键菜单", "坐标查询 (VSX/MPC/SIMBAD/TNS)"),
        ],
    ),
]


class ShortcutHelpDialog(QDialog):
    """快捷键帮助对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("快捷键列表")
        self.setMinimumSize(500, 500)

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        lbl_title = QLabel("⌨ SCANN v2 快捷键")
        lbl_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2196F3;")
        layout.addWidget(lbl_title)

        lbl_note = QLabel("所有快捷键仅在程序窗口获得焦点时有效 (非全局)。")
        lbl_note.setStyleSheet("color: #888; margin-bottom: 8px;")
        layout.addWidget(lbl_note)

        # 表格
        # 计算总行数
        total_rows = sum(len(shortcuts) + 1 for _, shortcuts in SHORTCUT_GROUPS)
        self.table = QTableWidget(total_rows, 2)
        self.table.setHorizontalHeaderLabels(["快捷键", "功能"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.setStyleSheet(
            "QTableWidget { background-color: #252526; border: 1px solid #3C3C3C; }"
            "QTableWidget::item { padding: 4px; }"
        )

        bold_font = QFont()
        bold_font.setBold(True)

        row = 0
        for group_name, shortcuts in SHORTCUT_GROUPS:
            # 分组标题行
            group_item = QTableWidgetItem(f"── {group_name} ──")
            group_item.setFont(bold_font)
            group_item.setForeground(Qt.cyan)
            self.table.setItem(row, 0, group_item)
            self.table.setSpan(row, 0, 1, 2)
            row += 1

            for key, desc in shortcuts:
                key_item = QTableWidgetItem(key)
                key_item.setFont(QFont("Consolas", 10, QFont.Bold))
                key_item.setForeground(Qt.yellow)
                self.table.setItem(row, 0, key_item)

                desc_item = QTableWidgetItem(desc)
                self.table.setItem(row, 1, desc_item)
                row += 1

        layout.addWidget(self.table, 1)

        # 关闭按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.close)
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)
