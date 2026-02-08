"""自定义 Widget 集合

所有可复用控件:
- CoordinateLabel: 可复制坐标标签
- NoScrollSpinBox / NoScrollDoubleSpinBox: 禁用滚轮的 SpinBox
- OverlayLabel: 半透明浮层状态标签
- SuspectTableWidget: 可疑目标表格
- HistogramPanel: 直方图拉伸面板 (QDockWidget)
- BlinkSpeedSlider: 闪烁速度滑块
- CollapsibleSidebar: 可折叠侧边栏
- MpcorbOverlay: MPCORB 已知小行星叠加层
"""

from scann.gui.widgets.coordinate_label import CoordinateLabel
from scann.gui.widgets.no_scroll_spinbox import NoScrollSpinBox, NoScrollDoubleSpinBox
from scann.gui.widgets.overlay_label import OverlayLabel
from scann.gui.widgets.suspect_table import SuspectTableWidget
from scann.gui.widgets.histogram_panel import HistogramPanel
from scann.gui.widgets.blink_speed_slider import BlinkSpeedSlider
from scann.gui.widgets.collapsible_sidebar import CollapsibleSidebar
from scann.gui.widgets.mpcorb_overlay import MpcorbOverlay

__all__ = [
    "CoordinateLabel",
    "NoScrollSpinBox",
    "NoScrollDoubleSpinBox",
    "OverlayLabel",
    "SuspectTableWidget",
    "HistogramPanel",
    "BlinkSpeedSlider",
    "CollapsibleSidebar",
    "MpcorbOverlay",
]
