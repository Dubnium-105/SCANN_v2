"""SCANN v2 主窗口

UI/UX 设计实现:
- 菜单栏: 文件 | 处理 | AI | 查询 | 视图 | 设置 | 帮助
- 可折叠侧边栏 (240px, Ctrl+B 切换)
- 图像区域 (最大化, ≥ 75% 窗口面积)
- 浮层状态标签 (NEW/OLD/INV)
- 控制栏 (40px): 切换/闪烁/反色/拉伸/标记
- 状态栏: 当前图类型 | 像素坐标 | 天球坐标 | 缩放百分比
- 快捷键: r=闪烁, n=假, y=真, i=反色, 1/2=新旧图, F=适配,
          Space=下一候选, ←→=上下配对, Ctrl+B=侧边栏
- 快捷键非全局，窗口焦点在程序内才有效
- 暗色主题 (#1E1E1E)
"""

from __future__ import annotations

from typing import Optional
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QMainWindow,
)

from scann.core.fits_io import read_fits
from scann.core.models import (
    Candidate,
    FitsHeader,
)
from scann.logger_config import get_logger
from scann.gui.controllers import (
    AnnotationController,
    DetectionController,
    FileActionsController,
    HelpController,
    ImageSessionController,
    ModelController,
    PairController,
    PreferencesController,
    QueryController,
    TrainingController,
)
from scann.gui.composition import MainWindowBuilder, MainWindowWiring
from scann.gui.presenters import CandidatePresenter, StatusPresenter
from scann.data.file_manager import scan_fits_folder, match_new_old_pairs
from scann.services.config_service import ConfigService
from scann.services.model_service import ModelService
from scann.services.pair_service import PairService
from scann.services.blink_service import BlinkService


# ─── 暗色主题样式表 ───
DARK_THEME_QSS = """
QMainWindow {
    background-color: #1E1E1E;
}
QWidget {
    background-color: #1E1E1E;
    color: #D4D4D4;
    font-size: 12px;
}
QMenuBar {
    background-color: #333333;
    color: #D4D4D4;
    border-bottom: 1px solid #3C3C3C;
}
QMenuBar::item:selected {
    background-color: #094771;
}
QMenu {
    background-color: #252526;
    color: #D4D4D4;
    border: 1px solid #3C3C3C;
}
QMenu::item:selected {
    background-color: #094771;
}
QMenu::separator {
    height: 1px;
    background-color: #3C3C3C;
}
QPushButton {
    background-color: #333333;
    color: #D4D4D4;
    border: 1px solid #3C3C3C;
    border-radius: 3px;
    padding: 4px 8px;
    min-height: 24px;
}
QPushButton:hover {
    background-color: #3C3C3C;
}
QPushButton:pressed {
    background-color: #094771;
}
QPushButton:checked {
    background-color: #094771;
    border-color: #2196F3;
}
QPushButton:disabled {
    background-color: #2A2A2A;
    color: #555555;
}
QListWidget {
    background-color: #252526;
    border: 1px solid #3C3C3C;
    color: #D4D4D4;
}
QListWidget::item:selected {
    background-color: #094771;
}
QProgressBar {
    background-color: #333333;
    border: 1px solid #3C3C3C;
    border-radius: 2px;
    text-align: center;
    color: #D4D4D4;
}
QProgressBar::chunk {
    background-color: #2196F3;
}
QStatusBar {
    background-color: #007ACC;
    color: white;
    font-size: 11px;
}
QSplitter::handle {
    background-color: #3C3C3C;
    width: 2px;
}
QLabel {
    color: #D4D4D4;
}
"""


class MainWindow(QMainWindow):
    """SCANN v2 主窗口

    分区:
    ┌─────────────────────────────────────────────┐
    │ 菜单栏                                       │
    ├──────────┬──────────────────────────────────┤
    │ 侧边栏   │  [OverlayLabel]                   │
    │ (可折叠) │  FitsImageViewer (弹性填充)        │
    │          │  [控制栏 40px]                     │
    ├──────────┴──────────────────────────────────┤
    │ 状态栏                                       │
    └─────────────────────────────────────────────┘
    """

    def __init__(self):
        super().__init__()

        # ── 配置 (最先加载，后续初始化依赖它) ──
        self.config_service = ConfigService()
        self._config = self.config_service.load_config()

        self.setWindowTitle("SCANN v2 - Star/Source Classification and Analysis Neural Network")
        self.resize(self._config.window_width, self._config.window_height)
        self.setMinimumSize(1024, 768)

        # 暗色主题
        self.setStyleSheet(DARK_THEME_QSS)

        # ── 定时器 ──
        self.blink_timer = QTimer(self)
        self.blink_timer.timeout.connect(self._on_blink_tick)

        # ── 数据状态 ──
        self._candidates: list[Candidate] = []
        self._current_candidate_idx: int = -1
        self._new_image_data: Optional[np.ndarray] = None
        self._old_image_data: Optional[np.ndarray] = None

        # ── 文件管理 ──
        self._new_folder: str = ""
        self._old_folder: str = ""
        self._image_pairs: list = []  # FitsImagePair 列表
        self._current_pair_idx: int = -1
        self._current_pair_using_aligned: bool = False
        self._new_fits_header: Optional[FitsHeader] = None
        self._old_fits_header: Optional[FitsHeader] = None
        
        # ── 候选目标缓存：key 是配对索引，value 是候选目标列表
        self._candidates_cache: dict[int, list[Candidate]] = {}

        # ── 日志 ──
        self._logger = get_logger(__name__)

        # ── 用持久化配置初始化服务 ──
        self.blink_service = BlinkService(speed_ms=self._config.blink_speed_ms)

        # ── 构建 UI ──
        self.ui_parts = MainWindowBuilder(self).build()
        self.ui_parts.attach(self)
        self._init_presenters()
        self._init_controllers()
        self.ui_wiring = MainWindowWiring(self)
        self.ui_wiring.wire()

        # ── 从配置恢复文件夹路径 ──
        self._new_folder = self._config.new_folder
        self._old_folder = self._config.old_folder

        # ── 从配置恢复 UI 状态 ──
        self._restore_ui_state()

    # ══════════════════════════════════════════════
    #  日志和消息输出
    # ══════════════════════════════════════════════

    def _init_presenters(self) -> None:
        """初始化主窗口展示职责委托对象。"""
        self.status_presenter = StatusPresenter(self.statusBar(), self._logger)
        self.candidate_presenter = CandidatePresenter(
            self.suspect_table,
            self.image_viewer,
        )

    def _init_controllers(self) -> None:
        """初始化主窗口控制器。"""
        self.image_session_controller = ImageSessionController(self)
        self.file_actions_controller = FileActionsController(self)
        self.annotation_controller = AnnotationController(self)
        self.help_controller = HelpController(self)
        self.model_service = ModelService()
        self.pair_service = PairService(
            scan_folder_fn=scan_fits_folder,
            match_pairs_fn=match_new_old_pairs,
            read_fits_fn=read_fits,
        )
        self.pair_controller = PairController(self, self.pair_service)
        self.model_controller = ModelController(self, self.model_service)
        self.training_controller = TrainingController(self)
        self.detection_controller = DetectionController(self)
        self.query_controller = QueryController(self)
        self.preferences_controller = PreferencesController(self, self.config_service)

    @property
    def _inference_engine(self):
        """兼容访问点，生命周期由 ModelService 托管。"""
        if hasattr(self, "model_service"):
            return self.model_service.inference_engine
        return self.__dict__.get("_legacy_inference_engine")

    @_inference_engine.setter
    def _inference_engine(self, inference_engine) -> None:
        if hasattr(self, "model_service"):
            self.model_service.set_inference_engine(inference_engine)
            return
        self.__dict__["_legacy_inference_engine"] = inference_engine

    @property
    def _training_dialog(self):
        if hasattr(self, "training_controller"):
            return self.training_controller.training_dialog
        return self.__dict__.get("_legacy_training_dialog")

    @_training_dialog.setter
    def _training_dialog(self, dialog) -> None:
        if hasattr(self, "training_controller"):
            self.training_controller._training_dialog = dialog
            return
        self.__dict__["_legacy_training_dialog"] = dialog

    @property
    def _training_worker(self):
        if hasattr(self, "training_controller"):
            return self.training_controller.training_worker
        return self.__dict__.get("_legacy_training_worker")

    @_training_worker.setter
    def _training_worker(self, worker) -> None:
        if hasattr(self, "training_controller"):
            self.training_controller.training_worker = worker
            return
        self.__dict__["_legacy_training_worker"] = worker

    @property
    def _training_params(self):
        if hasattr(self, "training_controller"):
            return self.training_controller.training_params
        return self.__dict__.get("_legacy_training_params", {})

    @_training_params.setter
    def _training_params(self, params: dict) -> None:
        if hasattr(self, "training_controller"):
            self.training_controller.training_params = params
            return
        self.__dict__["_legacy_training_params"] = params

    def _show_message(self, message: str, timeout: int = 3000, level: str = 'INFO') -> None:
        """统一的消息输出方法，同时输出到状态栏和日志

        Args:
            message: 消息内容
            timeout: 状态栏显示超时时间（毫秒）
            level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.status_presenter.show_message(message, timeout=timeout, level=level)

    # ══════════════════════════════════════════════
    #  菜单栏
    # ══════════════════════════════════════════════

    # ══════════════════════════════════════════════
    #  事件处理
    # ══════════════════════════════════════════════

    def _on_main_splitter_moved(self, _pos: int, _index: int) -> None:
        """记录用户通过拖动分割器调整后的侧边栏宽度。"""
        if not hasattr(self, "main_splitter"):
            return
        sizes = self.main_splitter.sizes()
        if len(sizes) >= 2 and sizes[0] > 0 and not self.sidebar.is_collapsed:
            self.sidebar.set_preferred_width(sizes[0])

    def _on_blink_toggle(self) -> None:
        """切换闪烁"""
        self.image_session_controller.toggle_blink()

    def _on_blink_tick(self) -> None:
        """闪烁定时回调"""
        self.image_session_controller.blink_tick()

    def _on_blink_speed_changed(self, speed_ms: int) -> None:
        """闪烁速度变化"""
        self.image_session_controller.set_blink_speed(speed_ms)

    def _on_invert_toggle(self) -> None:
        """切换反色 (持久状态: 切换图片不重置)"""
        self.image_session_controller.toggle_invert()

    def _on_show_new(self) -> None:
        """显示新图"""
        self.image_session_controller.show_new()

    def _on_show_old(self) -> None:
        """显示旧图"""
        self.image_session_controller.show_old()

    def _show_image(self, which: str) -> None:
        """统一的图像显示逻辑

        Args:
            which: "new" 或 "old"
        """
        self.image_session_controller.show_image(which)

    def _on_mark_real(self) -> None:
        """标记当前候选为真目标"""
        self.detection_controller.mark_real()

    def _on_mark_bogus(self) -> None:
        """标记当前候选为假目标"""
        self.detection_controller.mark_bogus()

    def _on_next_candidate(self) -> None:
        """跳转到下一个候选体"""
        self.detection_controller.next_candidate()

    def _on_candidate_selected(self, index: int) -> None:
        """候选表格单击选中"""
        self.detection_controller.candidate_selected(index)

    def _on_candidate_double_clicked(self, index: int) -> None:
        """候选表格双击 → 放大到候选体"""
        self.detection_controller.candidate_double_clicked(index)

    def _focus_candidate(self, index: int) -> None:
        """聚焦某个候选体"""
        self.detection_controller.focus_candidate(index)

    def _update_markers(self) -> None:
        """刷新候选标记"""
        show = self.act_show_markers.isChecked()
        self.candidate_presenter.refresh_markers(
            self._candidates,
            selected_idx=self._current_candidate_idx,
            show_markers=show,
        )

    def _on_toggle_histogram(self) -> None:
        """切换直方图面板"""
        self.image_session_controller.toggle_histogram()

    def _on_stretch_changed(self, black: float, white: float) -> None:
        """直方图拉伸参数变化 (仅影响显示)"""
        self.image_session_controller.stretch_changed(black, white)

    def _on_image_clicked(self, x: int, y: int) -> None:
        """图像左键点击"""
        self.query_controller.image_clicked(x, y)

    def _on_image_right_click(self, x: int, y: int) -> None:
        """图像右键点击 → 上下文查询菜单"""
        self.query_controller.image_right_click(x, y)

    def _do_query(self, query_type: str, x: int, y: int) -> None:
        """执行外部查询"""
        self.query_controller.do_query(query_type, x, y)

    def _on_prev_pair(self) -> None:
        """上一组图像配对"""
        self.pair_controller.prev_pair()

    def _on_next_pair(self) -> None:
        """下一组图像配对"""
        self.pair_controller.next_pair()

    # ══════════════════════════════════════════════
    #  菜单 / 按钮处理方法
    # ══════════════════════════════════════════════

    # ── 文件菜单 ──

    def _on_open_new_folder(self) -> None:
        """打开新图文件夹"""
        self.pair_controller.open_new_folder()

    def _add_recent_folder(self, folder: str) -> None:
        """添加文件夹到最近打开列表"""
        self.pair_controller.add_recent_folder(folder)

    def _on_open_old_folder(self) -> None:
        """打开旧图文件夹"""
        self.pair_controller.open_old_folder()

    def _on_save_image(self) -> None:
        """保存当前图像"""
        self.file_actions_controller.save_image()

    def _on_save_marked_image(self) -> None:
        """另存为带标记的图像"""
        self.file_actions_controller.save_marked_image()

    def _on_update_recent_menu(self) -> None:
        """更新最近打开菜单"""
        self.pair_controller.update_recent_menu()

    def _open_recent_folder(self, folder: str) -> None:
        """从最近打开列表恢复文件夹"""
        self.pair_controller.open_recent_folder(folder)

    # ── 处理菜单 ──

    def _on_batch_align(self) -> None:
        """批量对齐"""
        self.detection_controller.batch_align()

    def _on_batch_process(self) -> None:
        """打开批量处理对话框"""
        self.detection_controller.batch_process()

    def _run_batch_process(self, params: dict) -> None:
        """执行批量处理 (降噪/伪平场)"""
        self.detection_controller.run_batch_process(params)

    def _build_detection_params(self):
        """从 AppConfig 构造 DetectionParams"""
        return self.detection_controller.build_detection_params()

    # ── AI 菜单 ──

    def _on_batch_detect(self) -> None:
        """批量检测"""
        self.detection_controller.batch_detect()

    def _on_open_training(self) -> None:
        """打开训练对话框"""
        self.training_controller.open_training()

    def _on_open_annotation(self) -> None:
        """打开标注工具对话框 (非模态)"""
        self.annotation_controller.open_annotation()

    def _on_training_started(self, params: dict) -> None:
        """训练开始信号处理: 接收超参数并启动训练线程"""
        self.training_controller.training_started(params)

    def _on_training_progress(self, epoch: int, total: int, loss: float, val_loss: float) -> None:
        """训练进度更新"""
        self.training_controller.training_progress(epoch, total, loss, val_loss)

    def _on_training_finished(self, model_path: str, metrics: dict) -> None:
        """训练完成"""
        self.training_controller.training_finished(model_path, metrics)

    def _on_training_error(self, message: str) -> None:
        """训练出错"""
        self.training_controller.training_error(message)

    def _on_training_stopped(self) -> None:
        """训练停止信号处理"""
        self.training_controller.training_stopped()

    def _on_load_model(self) -> None:
        """加载 AI 模型 (支持 v1/v2 格式自动检测)"""
        self.model_controller.load_model()

    def _on_model_info(self) -> None:
        """显示模型信息"""
        self.model_controller.show_model_info()

    # ── 查询菜单 ──

    def _on_menu_query(self, query_type: str) -> None:
        """从菜单栏触发的查询 (无坐标上下文)"""
        self.query_controller.menu_query(query_type)

    def _on_mpc_report(self) -> None:
        """打开 MPC 80列报告对话框"""
        self.query_controller.mpc_report()

    # ── 视图菜单 ──

    def _on_zoom_actual(self) -> None:
        """重置缩放到 100%"""
        self.image_viewer.resetTransform()
        self.image_viewer._zoom_level = 1.0
        self.image_viewer._emit_zoom()

    def _on_zoom_in(self) -> None:
        """放大"""
        factor = self.image_viewer.ZOOM_FACTOR
        self.image_viewer.scale(factor, factor)
        self.image_viewer._zoom_level *= factor
        self.image_viewer._emit_zoom()

    def _on_zoom_out(self) -> None:
        """缩小"""
        factor = 1.0 / self.image_viewer.ZOOM_FACTOR
        self.image_viewer.scale(factor, factor)
        self.image_viewer._zoom_level *= factor
        self.image_viewer._emit_zoom()

    def _on_toggle_mpcorb(self, checked: bool) -> None:
        """切换 MPCORB 叠加显示"""
        self.image_viewer.set_mpcorb_visible(checked)
        self._show_message(f"MPCORB 叠加: {'开启' if checked else '关闭'}", 2000)

    def _on_toggle_known(self, checked: bool) -> None:
        """切换已知天体显示"""
        self.image_viewer.set_known_objects_visible(checked)
        self._show_message(f"已知天体标记: {'开启' if checked else '关闭'}", 2000)

    # ── 设置菜单 ──

    def _on_open_preferences(self) -> None:
        """打开首选项对话框"""
        self.preferences_controller.open_preferences()

    def _on_select_mpcorb_file(self) -> None:
        """选择 MPCORB 数据文件"""
        self.preferences_controller.select_mpcorb_file()

    def _on_open_scheduler(self) -> None:
        """打开计划任务设置"""
        self.help_controller.open_scheduler()

    # ── 帮助菜单 ──

    def _on_shortcut_help(self) -> None:
        """显示快捷键帮助对话框"""
        self.help_controller.open_shortcut_help()

    def _on_open_docs(self) -> None:
        """打开使用文档"""
        self.help_controller.open_docs()

    def _on_about(self) -> None:
        """显示关于对话框"""
        self.help_controller.open_about()

    # ── 图像查看器信号处理 ──

    def _on_mouse_moved(self, x: int, y: int) -> None:
        """鼠标在图像上移动 → 更新状态栏像素坐标"""
        self.image_session_controller.mouse_moved(x, y)

    def _on_zoom_changed(self, zoom_pct: float) -> None:
        """缩放比例变化 → 更新状态栏"""
        self.image_session_controller.zoom_changed(zoom_pct)

    # ── 右键上下文菜单处理 ──

    def _on_context_mpc_report(self, x: int, y: int) -> None:
        """右键菜单 → 生成 MPC 报告"""
        self.query_controller.context_mpc_report(x, y)

    def _on_context_add_candidate(self, x: int, y: int) -> None:
        """右键菜单 → 手动添加候选体"""
        self.query_controller.context_add_candidate(x, y)

    def _on_copy_wcs_coordinates(self, x: int, y: int) -> None:
        """右键菜单 → 复制天球坐标"""
        self.query_controller.copy_wcs_coordinates(x, y)

    # ══════════════════════════════════════════════
    #  图像配对加载
    # ══════════════════════════════════════════════

    def _load_pair(self, index: int) -> None:
        """加载指定索引的图像配对。"""
        self.pair_controller.load_pair(index)

    def _aligned_artifact_paths(self, pair) -> tuple[Path, Path, Path, Path]:
        """返回配对图像的对齐裁剪产物路径。"""
        return self.pair_controller.aligned_artifact_paths(pair)

    def _pair_has_aligned_artifacts(self, pair) -> bool:
        """配对是否已有可复用的对齐裁剪结果。"""
        return self.pair_controller.pair_has_aligned_artifacts(pair)

    def _resolve_pair_image_paths(self, pair) -> tuple[Path, Path, bool]:
        """解析配对应使用的图像路径。"""
        return self.pair_controller.resolve_pair_image_paths(pair)

    def _calc_nonzero_valid_bounds(self, image: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        """估计旧图有效区域边界。"""
        return self.pair_controller.calc_nonzero_valid_bounds(image)

    def _calc_overlap_crop_bounds(
        self,
        w: int,
        h: int,
        dx: float,
        dy: float,
        aligned_old: Optional[np.ndarray] = None,
    ) -> Optional[tuple[int, int, int, int]]:
        """根据平移量和旧图有效区域计算重叠裁剪区域。"""
        return self.pair_controller.calc_overlap_crop_bounds(
            w=w,
            h=h,
            dx=dx,
            dy=dy,
            aligned_old=aligned_old,
        )

    def _on_pair_selected(self, index: int) -> None:
        """配对列表选择事件"""
        self.pair_controller.select_pair(index)

    # ══════════════════════════════════════════════
    #  公共 API
    # ══════════════════════════════════════════════

    def set_image_data(
        self, new_data: Optional[np.ndarray], old_data: Optional[np.ndarray]
    ) -> None:
        """设置当前图像配对数据"""
        self.image_session_controller.set_image_data(new_data, old_data)

    def set_candidates(self, candidates: list[Candidate]) -> None:
        """设置检测到的候选体列表"""
        self._candidates = candidates
        self._current_candidate_idx = 0 if candidates else -1
        self.candidate_presenter.set_candidates(candidates)
        self._update_markers()

    # ══════════════════════════════════════════════
    #  窗口事件
    # ══════════════════════════════════════════════

    def closeEvent(self, event) -> None:
        """窗口关闭 → 自动保存配置"""
        if not self.preferences_controller.handle_close_event(event):
            return
        super().closeEvent(event)

    def _save_runtime_state(self) -> None:
        """将运行时状态同步到配置对象"""
        self.preferences_controller.save_runtime_state()

    def _restore_ui_state(self) -> None:
        """从配置恢复 UI 状态 (在构建 UI 后调用)"""
        self.preferences_controller.restore_ui_state()

    def resizeEvent(self, event) -> None:
        """窗口大小变化 → 自动折叠侧边栏"""
        super().resizeEvent(event)
        self.preferences_controller.handle_resize_event()
