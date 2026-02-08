"""GUI layer - PyQt5 based user interface.

模块结构:
- main_window: 主窗口 (菜单栏, 侧边栏, 图像区域, 控制栏, 状态栏)
- image_viewer: FITS 图像查看器 (QGraphicsView)
- widgets/: 自定义控件 (overlay, table, histogram, slider, sidebar, mpcorb)
- dialogs/: 弹出对话框 (settings, training, batch, mpc_report, query, shortcuts)
"""

from scann.gui.main_window import MainWindow
from scann.gui.image_viewer import FitsImageViewer

__all__ = ["MainWindow", "FitsImageViewer"]
