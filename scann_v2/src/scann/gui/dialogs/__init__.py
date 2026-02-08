"""GUI 对话框模块

包含所有弹出对话框:
- SettingsDialog: 首选项设置 (多标签页)
- TrainingDialog: AI 模型训练
- BatchProcessDialog: 批量图像处理
- MpcReportDialog: MPC 80列报告
- QueryResultPopup: 查询结果浮窗
- ShortcutHelpDialog: 快捷键帮助
"""

from scann.gui.dialogs.settings_dialog import SettingsDialog
from scann.gui.dialogs.training_dialog import TrainingDialog
from scann.gui.dialogs.batch_process_dialog import BatchProcessDialog
from scann.gui.dialogs.mpc_report_dialog import MpcReportDialog
from scann.gui.dialogs.query_result_popup import QueryResultPopup
from scann.gui.dialogs.shortcut_help_dialog import ShortcutHelpDialog

__all__ = [
    "SettingsDialog",
    "TrainingDialog",
    "BatchProcessDialog",
    "MpcReportDialog",
    "QueryResultPopup",
    "ShortcutHelpDialog",
]
