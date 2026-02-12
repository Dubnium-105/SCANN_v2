"""主窗口 TODO 功能测试 (TDD)

覆盖 TODO 文档中的高优先级和中优先级功能:
- 文件加载 (打开新图/旧图文件夹)
- 线性拉伸显示
- 批量对齐
- AI 模型加载
- 批量检测
- 保存图像
- 右键菜单功能 (手动候选体, WCS坐标复制)
- WCS 坐标同步
- 模型信息显示
- 最近打开菜单
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call, PropertyMock

from scann.core.models import (
    Candidate,
    TargetVerdict,
    FitsImage,
    FitsHeader,
    AlignResult,
    Detection,
    MarkerType,
    AppConfig,
)
from scann.services.blink_service import BlinkState
from scann.services.query_service import QueryResult


# ═══════════════════════════════════════════════
#  辅助: 创建增强版 Mock MainWindow
# ═══════════════════════════════════════════════


def _make_mock_window():
    """创建一个跳过 __init__ 的 MainWindow, 手动挂载 Mock 属性"""
    from scann.gui.main_window import MainWindow

    with patch("scann.gui.main_window.QMainWindow.__init__"):
        w = MainWindow.__new__(MainWindow)

    # 图像查看器
    w.image_viewer = Mock()

    # 闪烁服务
    w.blink_service = Mock()
    w.blink_service.is_inverted = False
    w.blink_service.is_running = False
    w.blink_service.speed_ms = 500
    w.blink_service.current_state = BlinkState.NEW

    # 定时器
    w.blink_timer = Mock()

    # 浮层标签
    w.overlay_state = Mock()
    w.overlay_inv = Mock()
    w.overlay_blink = Mock()

    # 控制栏按钮
    w.btn_show_new = Mock()
    w.btn_show_old = Mock()
    w.btn_blink = Mock()
    w.btn_invert = Mock()
    w.btn_mark_real = Mock()
    w.btn_mark_bogus = Mock()
    w.btn_next_candidate = Mock()

    # 侧边栏
    w.sidebar = Mock()

    # 候选表格
    w.suspect_table = Mock()

    # 闪烁速度
    w.blink_speed = Mock()

    # 直方图面板
    w.histogram_panel = Mock()
    w.histogram_panel.black_point = 0.0
    w.histogram_panel.white_point = 1.0

    # 文件列表
    w.file_list = Mock()
    w.file_list.count.return_value = 0

    # 进度条
    w.progress_bar = Mock()

    # 状态栏
    w.status_image_type = Mock()
    w.status_pixel_coord = Mock()
    w.status_wcs_coord = Mock()
    w.status_zoom = Mock()

    # 动作
    w.act_show_markers = Mock()
    w.act_show_markers.isChecked.return_value = True
    w.act_show_mpcorb = Mock()
    w.act_show_known = Mock()

    # 最近打开菜单
    w.menu_recent = Mock()

    # statusBar mock
    w.statusBar = Mock(return_value=Mock())

    # 数据
    w._candidates = []
    w._current_candidate_idx = -1
    w._new_image_data = None
    w._old_image_data = None

    # 新增: 文件管理相关数据
    w._new_folder = ""
    w._old_folder = ""
    w._image_pairs = []
    w._current_pair_idx = -1
    w._new_fits_header = None
    w._old_fits_header = None
    w._inference_engine = None
    w._config = AppConfig()
    w._annotation_dialog = None
    w._training_worker = None

    # logger mock（_show_message 依赖 self._logger）
    w._logger = Mock()

    return w


# ═══════════════════════════════════════════════
#  功能 1: 打开新图/旧图文件夹
# ═══════════════════════════════════════════════


class TestOpenNewFolder:
    """测试打开新图文件夹功能"""

    @patch("scann.gui.main_window.QFileDialog.getExistingDirectory")
    @patch("scann.gui.main_window.scan_fits_folder")
    def test_open_new_folder_loads_files(self, mock_scan, mock_dialog):
        """打开新图文件夹应扫描FITS文件并填充file_list"""
        from scann.data.file_manager import FitsFileInfo
        w = _make_mock_window()
        mock_dialog.return_value = "/path/to/new"
        mock_scan.return_value = [
            FitsFileInfo(path=Path("/path/to/new/img_001.fits"), stem="img_001",
                        size_bytes=1024, modified_time=0.0),
            FitsFileInfo(path=Path("/path/to/new/img_002.fits"), stem="img_002",
                        size_bytes=1024, modified_time=0.0),
        ]

        w._on_open_new_folder()

        assert w._new_folder == "/path/to/new"
        mock_scan.assert_called_once_with("/path/to/new")
        # 应该向 file_list 中添加了项目
        assert w.file_list.addItem.call_count == 2

    @patch("scann.gui.main_window.QFileDialog.getExistingDirectory")
    def test_open_new_folder_cancelled(self, mock_dialog):
        """取消对话框不应改变状态"""
        w = _make_mock_window()
        mock_dialog.return_value = ""  # 取消

        w._on_open_new_folder()

        assert w._new_folder == ""
        w.file_list.clear.assert_not_called()

    @patch("scann.gui.main_window.QFileDialog.getExistingDirectory")
    @patch("scann.gui.main_window.scan_fits_folder")
    def test_open_new_folder_clears_old_list(self, mock_scan, mock_dialog):
        """打开新文件夹应先清空旧列表"""
        w = _make_mock_window()
        mock_dialog.return_value = "/new/path"
        mock_scan.return_value = []

        w._on_open_new_folder()

        w.file_list.clear.assert_called_once()

    @patch("scann.gui.main_window.QFileDialog.getExistingDirectory")
    @patch("scann.gui.main_window.scan_fits_folder")
    @patch("scann.gui.main_window.read_fits")
    def test_open_new_folder_loads_first_image(self, mock_read, mock_scan, mock_dialog):
        """打开文件夹后应自动加载第一张图"""
        from scann.data.file_manager import FitsFileInfo
        w = _make_mock_window()
        mock_dialog.return_value = "/path/to/new"

        test_data = np.zeros((64, 64), dtype=np.uint16)
        test_header = FitsHeader(raw={"OBJECT": "TestField"})
        mock_scan.return_value = [
            FitsFileInfo(path=Path("/path/to/new/img_001.fits"), stem="img_001",
                        size_bytes=1024, modified_time=0.0),
        ]
        mock_read.return_value = FitsImage(
            data=test_data, header=test_header, path=Path("/path/to/new/img_001.fits")
        )

        w._on_open_new_folder()

        assert w._new_image_data is not None
        w.image_viewer.set_image_data.assert_called()


class TestOpenOldFolder:
    """测试打开旧图文件夹功能"""

    @patch("scann.gui.main_window.QFileDialog.getExistingDirectory")
    @patch("scann.gui.main_window.scan_fits_folder")
    def test_open_old_folder_stores_path(self, mock_scan, mock_dialog):
        """打开旧图文件夹应保存路径"""
        w = _make_mock_window()
        mock_dialog.return_value = "/path/to/old"
        mock_scan.return_value = []

        w._on_open_old_folder()

        assert w._old_folder == "/path/to/old"

    @patch("scann.gui.main_window.QFileDialog.getExistingDirectory")
    @patch("scann.gui.main_window.scan_fits_folder")
    @patch("scann.gui.main_window.match_new_old_pairs")
    def test_open_old_folder_triggers_pairing(self, mock_match, mock_scan, mock_dialog):
        """打开旧图文件夹应自动与新图配对"""
        from scann.data.file_manager import FitsImagePair
        w = _make_mock_window()
        w._new_folder = "/path/to/new"
        mock_dialog.return_value = "/path/to/old"
        mock_scan.return_value = []
        mock_match.return_value = (
            [FitsImagePair(name="img_001",
                          new_path=Path("/path/to/new/img_001.fits"),
                          old_path=Path("/path/to/old/img_001.fits"))],
            [],  # only_new
            [],  # only_old
        )

        w._on_open_old_folder()

        mock_match.assert_called_once_with("/path/to/new", "/path/to/old")
        assert len(w._image_pairs) == 1

    @patch("scann.gui.main_window.QFileDialog.getExistingDirectory")
    def test_open_old_folder_cancelled(self, mock_dialog):
        """取消不应改变状态"""
        w = _make_mock_window()
        mock_dialog.return_value = ""

        w._on_open_old_folder()

        assert w._old_folder == ""


# ═══════════════════════════════════════════════
#  功能 2: 线性拉伸显示
# ═══════════════════════════════════════════════


class TestStretchChanged:
    """测试直方图拉伸回调"""

    def test_stretch_with_new_image(self):
        """拉伸参数变化时应通过ImageProcessor处理并刷新显示"""
        w = _make_mock_window()
        w._new_image_data = np.random.random((64, 64)).astype(np.float32) * 65535
        w.blink_service.current_state = BlinkState.NEW

        with patch("scann.gui.main_window.histogram_stretch") as mock_stretch:
            mock_stretch.return_value = np.zeros((64, 64), dtype=np.float32)
            w._on_stretch_changed(100.0, 50000.0)
            mock_stretch.assert_called_once()
            w.image_viewer.set_image_data.assert_called_once()

    def test_stretch_with_no_image(self):
        """无图像数据时拉伸不应崩溃"""
        w = _make_mock_window()
        w._new_image_data = None
        w._old_image_data = None

        # 不应崩溃
        w._on_stretch_changed(0.0, 65535.0)

    def test_stretch_uses_black_white_points(self):
        """拉伸应使用传入的黑白点参数"""
        w = _make_mock_window()
        w._new_image_data = np.ones((32, 32), dtype=np.float32) * 1000
        w.blink_service.current_state = BlinkState.NEW

        with patch("scann.gui.main_window.histogram_stretch") as mock_stretch:
            mock_stretch.return_value = np.zeros((32, 32), dtype=np.float32)
            w._on_stretch_changed(200.0, 800.0)
            args, kwargs = mock_stretch.call_args
            assert kwargs.get("black_point") == 200.0 or args[1] == 200.0


# ═══════════════════════════════════════════════
#  功能 3: 批量对齐
# ═══════════════════════════════════════════════


class TestBatchAlign:
    """测试批量对齐功能"""

    @patch("scann.gui.main_window.read_fits")
    @patch("scann.gui.main_window.align")
    def test_batch_align_processes_pairs(self, mock_align, mock_read):
        """批量对齐应处理所有图像配对"""
        from scann.data.file_manager import FitsImagePair
        w = _make_mock_window()

        pair = FitsImagePair(
            name="img_001",
            new_path=Path("/new/img_001.fits"),
            old_path=Path("/old/img_001.fits"),
        )
        w._image_pairs = [pair]

        new_data = np.zeros((64, 64), dtype=np.float32)
        old_data = np.ones((64, 64), dtype=np.float32)
        aligned_old = np.zeros((64, 64), dtype=np.float32)

        mock_read.side_effect = [
            FitsImage(data=new_data, header=FitsHeader(raw={}), path=pair.new_path),
            FitsImage(data=old_data, header=FitsHeader(raw={}), path=pair.old_path),
        ]
        mock_align.return_value = AlignResult(aligned_old=aligned_old, dx=1.0, dy=2.0, success=True)

        w._on_batch_align()

        mock_align.assert_called_once()

    def test_batch_align_no_pairs_shows_message(self):
        """无配对时应显示提示信息"""
        w = _make_mock_window()
        w._image_pairs = []

        w._on_batch_align()

        w.statusBar().showMessage.assert_called()
        msg = w.statusBar().showMessage.call_args[0][0]
        assert "配对" in msg or "对齐" in msg or "文件" in msg


# ═══════════════════════════════════════════════
#  功能 4: AI 模型加载
# ═══════════════════════════════════════════════


class TestLoadModel:
    """测试加载AI模型功能"""

    @patch("scann.gui.main_window.QFileDialog.getOpenFileName")
    @patch("scann.gui.main_window.InferenceEngine")
    def test_load_model_success(self, mock_engine_cls, mock_dialog):
        """成功加载模型应设置inference_engine"""
        w = _make_mock_window()
        mock_dialog.return_value = ("/path/to/model.pth", "")
        mock_engine = Mock()
        mock_engine.is_ready = True
        mock_engine.threshold = 0.5
        mock_engine_cls.return_value = mock_engine

        w._on_load_model()

        mock_engine_cls.assert_called_once()
        # 验证 model_path 关键字参数
        call_kwargs = mock_engine_cls.call_args[1]
        assert call_kwargs["model_path"] == "/path/to/model.pth"
        assert w._inference_engine is mock_engine
        w.statusBar().showMessage.assert_called()

    @patch("scann.gui.main_window.QFileDialog.getOpenFileName")
    def test_load_model_cancelled(self, mock_dialog):
        """取消不应改变状态"""
        w = _make_mock_window()
        w._inference_engine = None
        mock_dialog.return_value = ("", "")

        w._on_load_model()

        assert w._inference_engine is None

    @patch("scann.gui.main_window.QFileDialog.getOpenFileName")
    @patch("scann.gui.main_window.InferenceEngine")
    def test_load_model_failure_shows_error(self, mock_engine_cls, mock_dialog):
        """加载失败应显示错误信息"""
        w = _make_mock_window()
        mock_dialog.return_value = ("/bad/model.pth", "")
        mock_engine_cls.side_effect = Exception("模型文件损坏")

        w._on_load_model()

        w.statusBar().showMessage.assert_called()
        msg = w.statusBar().showMessage.call_args[0][0]
        assert "失败" in msg or "错误" in msg or "损坏" in msg


# ═══════════════════════════════════════════════
#  功能 5: 批量检测
# ═══════════════════════════════════════════════


class TestBatchDetect:
    """测试批量检测功能"""

    def test_batch_detect_requires_image_data(self):
        """无图像数据时应显示提示"""
        w = _make_mock_window()
        w._new_image_data = None

        w._on_batch_detect()

        w.statusBar().showMessage.assert_called()

    @patch("scann.gui.main_window.DetectionPipeline")
    def test_batch_detect_with_data(self, mock_pipeline_cls):
        """有图像数据时应执行检测管线"""
        from scann.services.detection_service import PipelineResult
        w = _make_mock_window()
        w._new_image_data = np.zeros((64, 64), dtype=np.float32)
        w._old_image_data = np.ones((64, 64), dtype=np.float32)

        mock_pipeline = Mock()
        mock_pipeline.process_pair.return_value = PipelineResult(
            pair_name="test",
            candidates=[Candidate(x=10, y=20, ai_score=0.9)],
        )
        mock_pipeline_cls.return_value = mock_pipeline

        w._on_batch_detect()

        mock_pipeline.process_pair.assert_called_once()
        assert len(w._candidates) > 0
        w.suspect_table.set_candidates.assert_called()

    @patch("scann.gui.main_window.DetectionPipeline")
    def test_batch_detect_updates_candidates(self, mock_pipeline_cls):
        """检测结果应正确设置到界面"""
        from scann.services.detection_service import PipelineResult
        w = _make_mock_window()
        w._new_image_data = np.zeros((64, 64), dtype=np.float32)
        w._old_image_data = np.ones((64, 64), dtype=np.float32)

        cands = [Candidate(x=10, y=20, ai_score=0.9), Candidate(x=30, y=40, ai_score=0.7)]
        mock_pipeline = Mock()
        mock_pipeline.process_pair.return_value = PipelineResult(
            pair_name="test", candidates=cands,
        )
        mock_pipeline_cls.return_value = mock_pipeline

        w._on_batch_detect()

        assert w._candidates == cands
        assert w._current_candidate_idx == 0


# ═══════════════════════════════════════════════
#  功能 6: 保存图像
# ═══════════════════════════════════════════════


class TestSaveImage:
    """测试保存图像功能"""

    @patch("scann.gui.main_window.QFileDialog.getSaveFileName")
    @patch("scann.gui.main_window.write_fits")
    def test_save_image_with_data(self, mock_write, mock_dialog):
        """有数据时应保存"""
        w = _make_mock_window()
        w._new_image_data = np.zeros((64, 64), dtype=np.uint16)
        mock_dialog.return_value = ("/save/test.fits", "")

        w._on_save_image()

        mock_write.assert_called_once()

    def test_save_image_no_data(self):
        """无数据时应提示"""
        w = _make_mock_window()
        w._new_image_data = None

        w._on_save_image()

        w.statusBar().showMessage.assert_called()

    @patch("scann.gui.main_window.QFileDialog.getSaveFileName")
    def test_save_marked_image_creates_file(self, mock_dialog):
        """另存标记图应导出PNG/FITS"""
        w = _make_mock_window()
        w._new_image_data = np.zeros((64, 64), dtype=np.uint16)
        w._candidates = [Candidate(x=10, y=20)]
        mock_dialog.return_value = ("/save/marked.png", "")

        w._on_save_marked_image()

        w.statusBar().showMessage.assert_called()


# ═══════════════════════════════════════════════
#  功能 7: 右键菜单 - 手动添加候选体
# ═══════════════════════════════════════════════


class TestContextAddCandidate:
    """测试右键菜单手动添加候选体"""

    def test_add_candidate_creates_new(self):
        """应在指定坐标创建手动候选体"""
        w = _make_mock_window()
        w._candidates = []

        w._on_context_add_candidate(100, 200)

        assert len(w._candidates) == 1
        assert w._candidates[0].x == 100
        assert w._candidates[0].y == 200
        assert w._candidates[0].is_manual is True

    def test_add_candidate_appends_to_existing(self):
        """应追加到现有列表"""
        w = _make_mock_window()
        w._candidates = [Candidate(x=1, y=1)]

        w._on_context_add_candidate(50, 60)

        assert len(w._candidates) == 2
        assert w._candidates[-1].x == 50

    def test_add_candidate_updates_table(self):
        """添加后应刷新表格"""
        w = _make_mock_window()
        w._candidates = []

        w._on_context_add_candidate(10, 20)

        w.suspect_table.set_candidates.assert_called()


# ═══════════════════════════════════════════════
#  功能 8: WCS 坐标同步
# ═══════════════════════════════════════════════


class TestWCSSync:
    """测试WCS坐标同步更新"""

    def test_mouse_moved_updates_wcs_with_header(self):
        """有WCS头信息时鼠标移动应更新天球坐标"""
        w = _make_mock_window()
        w._new_fits_header = FitsHeader(raw={
            "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
            "CRVAL1": 180.0, "CRVAL2": 45.0,
            "CRPIX1": 64.0, "CRPIX2": 64.0,
            "CD1_1": -0.001, "CD1_2": 0.0,
            "CD2_1": 0.0, "CD2_2": 0.001,
            "NAXIS1": 128, "NAXIS2": 128,
        })

        with patch("scann.gui.main_window.pixel_to_wcs") as mock_p2w:
            from scann.core.models import SkyPosition
            mock_p2w.return_value = SkyPosition(ra=180.5, dec=45.3)

            w._on_mouse_moved(64, 64)

            mock_p2w.assert_called_once()
            w.status_wcs_coord.set_wcs_coordinates.assert_called()

    def test_mouse_moved_no_wcs_header(self):
        """无WCS头信息时不应更新天球坐标"""
        w = _make_mock_window()
        w._new_fits_header = None

        w._on_mouse_moved(64, 64)

        w.status_pixel_coord.set_pixel_coordinates.assert_called_with(64, 64)
        w.status_wcs_coord.set_wcs_coordinates.assert_not_called()


# ═══════════════════════════════════════════════
#  功能 9: 复制天球坐标
# ═══════════════════════════════════════════════


class TestCopyWCSCoordinates:
    """测试复制天球坐标功能"""

    @patch("scann.gui.main_window.QApplication")
    @patch("scann.gui.main_window.pixel_to_wcs")
    def test_copy_wcs_with_header(self, mock_p2w, mock_qapp):
        """有WCS时应复制RA/Dec到剪贴板"""
        from scann.core.models import SkyPosition
        w = _make_mock_window()
        w._new_fits_header = FitsHeader(raw={"CTYPE1": "RA---TAN"})
        mock_p2w.return_value = SkyPosition(ra=180.5, dec=45.3)
        mock_clipboard = Mock()
        mock_qapp.clipboard.return_value = mock_clipboard

        w._on_copy_wcs_coordinates(64, 64)

        mock_clipboard.setText.assert_called_once()
        text = mock_clipboard.setText.call_args[0][0]
        assert "180" in text or "12" in text  # RA

    def test_copy_wcs_no_header_shows_message(self):
        """无WCS时应显示提示"""
        w = _make_mock_window()
        w._new_fits_header = None

        w._on_copy_wcs_coordinates(64, 64)

        w.statusBar().showMessage.assert_called()


# ═══════════════════════════════════════════════
#  功能 10: 模型信息显示
# ═══════════════════════════════════════════════


class TestModelInfo:
    """测试模型信息显示"""

    def test_model_info_no_model(self):
        """无模型时应提示"""
        w = _make_mock_window()
        w._inference_engine = None

        w._on_model_info()

        w.statusBar().showMessage.assert_called()
        msg = w.statusBar().showMessage.call_args[0][0]
        assert "模型" in msg or "加载" in msg

    @patch("scann.gui.main_window.QMessageBox")
    def test_model_info_with_model(self, mock_msgbox_cls):
        """有模型时应显示信息"""
        w = _make_mock_window()
        w._inference_engine = Mock()
        w._inference_engine.is_ready = True
        w._inference_engine.model = Mock()
        w._inference_engine.threshold = 0.5
        w._inference_engine.device = "cpu"
        # 模拟 parameters()
        w._inference_engine.model.parameters.return_value = [
            Mock(numel=Mock(return_value=100)),
            Mock(numel=Mock(return_value=200)),
        ]

        # 不应崩溃
        w._on_model_info()


# ═══════════════════════════════════════════════
#  功能 11: 最近打开菜单
# ═══════════════════════════════════════════════


class TestRecentMenu:
    """测试最近打开菜单"""

    def test_update_recent_menu_empty(self):
        """无最近文件时应显示占位文本"""
        w = _make_mock_window()
        w._config = AppConfig()

        w._on_update_recent_menu()

        w.menu_recent.clear.assert_called_once()

    def test_update_recent_menu_with_items(self):
        """有最近文件时应填充菜单"""
        w = _make_mock_window()
        w._config = AppConfig()
        w._config.recent_folders = ["/path/a", "/path/b"]

        w._on_update_recent_menu()

        w.menu_recent.clear.assert_called_once()
        assert w.menu_recent.addAction.call_count >= 2


# ═══════════════════════════════════════════════
#  功能 12: 配对列表选择加载图像
# ═══════════════════════════════════════════════


class TestPairListSelection:
    """测试配对列表选择触发图像加载"""

    @patch("scann.gui.main_window.read_fits")
    def test_file_list_selection_loads_pair(self, mock_read):
        """选择配对列表项应加载对应图像"""
        from scann.data.file_manager import FitsImagePair
        w = _make_mock_window()
        pair = FitsImagePair(
            name="img_001",
            new_path=Path("/new/img_001.fits"),
            old_path=Path("/old/img_001.fits"),
        )
        w._image_pairs = [pair]

        test_header = FitsHeader(raw={})
        mock_read.side_effect = [
            FitsImage(data=np.zeros((64, 64)), header=test_header, path=pair.new_path),
            FitsImage(data=np.ones((64, 64)), header=test_header, path=pair.old_path),
        ]

        w._on_pair_selected(0)

        assert w._new_image_data is not None
        assert w._old_image_data is not None
        assert mock_read.call_count == 2

    def test_pair_selected_out_of_range(self):
        """越界索引不应崩溃"""
        w = _make_mock_window()
        w._image_pairs = []

        w._on_pair_selected(5)  # 不应崩溃


# ═══════════════════════════════════════════════
#  功能 13: 批量处理 (降噪/伪平场)
# ═══════════════════════════════════════════════


class TestBatchProcess:
    """测试批量处理对话框集成"""

    def test_batch_process_opens_dialog(self):
        """应打开批量处理对话框"""
        w = _make_mock_window()
        with patch("scann.gui.dialogs.batch_process_dialog.BatchProcessDialog") as mock_dlg:
            mock_instance = Mock()
            mock_dlg.return_value = mock_instance
            w._on_batch_process()
            mock_dlg.assert_called_once()

    @patch("scann.gui.main_window.write_fits")
    @patch("scann.gui.main_window.read_fits")
    def test_batch_process_denoise(self, mock_read, mock_write):
        """process_started 信号应触发降噪处理"""
        w = _make_mock_window()
        w._new_folder = "/fake/new"
        w._batch_dialog = None

        test_data = np.random.rand(64, 64).astype(np.float32)
        test_header = FitsHeader(raw={})
        mock_read.return_value = FitsImage(data=test_data, header=test_header, path="/fake/f.fits")

        with patch("scann.gui.main_window.scan_fits_folder", return_value=[Path("/fake/f.fits")]):
            with patch("scann.gui.main_window.denoise") as mock_denoise:
                mock_denoise.return_value = test_data
                w._run_batch_process({
                    "input_dir": "/fake/new",
                    "output_dir": "/fake/out",
                    "denoise": True,
                    "denoise_method": "中值滤波",
                    "kernel_size": 3,
                    "flat_field": False,
                    "flat_sigma": 100.0,
                    "bit_depth": "16-bit (保持原样)",
                    "overwrite": False,
                })
                mock_denoise.assert_called_once()

    @patch("scann.gui.main_window.write_fits")
    @patch("scann.gui.main_window.read_fits")
    def test_batch_process_flat_field(self, mock_read, mock_write):
        """process_started 信号应触发伪平场校正"""
        w = _make_mock_window()
        w._new_folder = "/fake/new"

        test_data = np.random.rand(64, 64).astype(np.float32)
        test_header = FitsHeader(raw={})
        mock_read.return_value = FitsImage(data=test_data, header=test_header, path="/fake/f.fits")

        with patch("scann.gui.main_window.scan_fits_folder", return_value=[Path("/fake/f.fits")]):
            with patch("scann.gui.main_window.pseudo_flat_field") as mock_flat:
                mock_flat.return_value = test_data
                w._run_batch_process({
                    "input_dir": "/fake/new",
                    "output_dir": "/fake/out",
                    "denoise": False,
                    "denoise_method": "中值滤波",
                    "kernel_size": 3,
                    "flat_field": True,
                    "flat_sigma": 100.0,
                    "bit_depth": "16-bit (保持原样)",
                    "overwrite": False,
                })
                mock_flat.assert_called_once()


# ═══════════════════════════════════════════════
#  功能 14: 训练对话框集成
# ═══════════════════════════════════════════════


class TestTrainingIntegration:
    """测试训练对话框与后台 Trainer 集成"""

    def test_training_dialog_opens(self):
        """应打开训练对话框"""
        w = _make_mock_window()
        with patch("scann.gui.dialogs.training_dialog.TrainingDialog") as mock_dlg:
            mock_instance = Mock()
            mock_dlg.return_value = mock_instance
            w._on_open_training()
            mock_dlg.assert_called_once()

    @patch("scann.ai.training_worker.TrainingWorker")
    def test_training_started_calls_trainer(self, mock_worker_cls):
        """training_started 信号应启动后台训练"""
        w = _make_mock_window()
        mock_worker = Mock()
        mock_worker_cls.return_value = mock_worker
        params = {
            "pos_dir": "/fake/pos",
            "neg_dir": "/fake/neg",
            "val_split": 0.2,
            "epochs": 10,
            "batch_size": 32,
            "lr": 0.001,
            "optimizer": "Adam",
            "backbone": "ResNet18",
            "augment": True,
            "early_stop": True,
            "patience": 10,
        }
        # _on_training_started should be a method that receives the params dict
        w._on_training_started(params)
        # 应该创建了 TrainConfig 或发起了训练流程
        w.statusBar().showMessage.assert_called()

    def test_training_stopped(self):
        """training_stopped 信号应停止训练"""
        w = _make_mock_window()
        w._training_thread = None
        w._on_training_stopped()
        w.statusBar().showMessage.assert_called()


# ═══════════════════════════════════════════════
#  功能 15: 远程查询集成
# ═══════════════════════════════════════════════


class TestQueryIntegration:
    """测试查询服务集成"""

    def test_do_query_vsx_with_wcs(self):
        """有 WCS 时 _do_query 应调用 QueryService 并显示结果"""
        w = _make_mock_window()
        w._new_fits_header = FitsHeader(raw={"CTYPE1": "RA---TAN"})

        mock_sky = Mock()
        mock_sky.ra = 180.0
        mock_sky.dec = 45.0

        with patch("scann.gui.main_window.pixel_to_wcs", return_value=mock_sky):
            with patch("scann.gui.main_window.QueryService") as mock_svc_cls:
                mock_svc = Mock()
                mock_svc.query_vsx.return_value = [
                    QueryResult(
                        source="VSX", name="V1234 Sgr",
                        object_type="EA", distance_arcsec=2.5,
                    )
                ]
                mock_svc_cls.return_value = mock_svc

                with patch("scann.gui.main_window.QueryResultPopup") as mock_popup_cls:
                    mock_popup = Mock()
                    mock_popup_cls.return_value = mock_popup

                    w._do_query("vsx", 50, 50)

                    mock_svc.query_vsx.assert_called_once()
                    mock_popup.set_content.assert_called_once()
                    mock_popup.show.assert_called_once()

    def test_do_query_mpc(self):
        """MPC 查询应调用 query_mpc"""
        w = _make_mock_window()
        w._new_fits_header = FitsHeader(raw={"CTYPE1": "RA---TAN"})

        mock_sky = Mock()
        mock_sky.ra = 200.0
        mock_sky.dec = -10.0

        with patch("scann.gui.main_window.pixel_to_wcs", return_value=mock_sky):
            with patch("scann.gui.main_window.QueryService") as mock_svc_cls:
                mock_svc = Mock()
                mock_svc.query_mpc.return_value = []
                mock_svc_cls.return_value = mock_svc

                with patch("scann.gui.main_window.QueryResultPopup") as mock_popup_cls:
                    mock_popup = Mock()
                    mock_popup_cls.return_value = mock_popup

                    w._do_query("mpc", 50, 50)

                    mock_svc.query_mpc.assert_called_once()

    def test_do_query_simbad(self):
        """SIMBAD 查询应调用 query_simbad"""
        w = _make_mock_window()
        w._new_fits_header = FitsHeader(raw={"CTYPE1": "RA---TAN"})

        mock_sky = Mock()
        mock_sky.ra = 100.0
        mock_sky.dec = 20.0

        with patch("scann.gui.main_window.pixel_to_wcs", return_value=mock_sky):
            with patch("scann.gui.main_window.QueryService") as mock_svc_cls:
                mock_svc = Mock()
                mock_svc.query_simbad.return_value = []
                mock_svc_cls.return_value = mock_svc

                with patch("scann.gui.main_window.QueryResultPopup") as mock_popup_cls:
                    mock_popup = Mock()
                    mock_popup_cls.return_value = mock_popup

                    w._do_query("simbad", 50, 50)

                    mock_svc.query_simbad.assert_called_once()

    def test_do_query_no_wcs_fallback(self):
        """无 WCS 时应提示并使用像素坐标"""
        w = _make_mock_window()
        w._new_fits_header = None

        w._do_query("vsx", 50, 50)

        w.statusBar().showMessage.assert_called()
        msg = w.statusBar().showMessage.call_args[0][0]
        assert "像素坐标" in msg

    def test_do_query_tns(self):
        """TNS 查询应调用 query_tns"""
        w = _make_mock_window()
        w._new_fits_header = FitsHeader(raw={"CTYPE1": "RA---TAN"})

        mock_sky = Mock()
        mock_sky.ra = 150.0
        mock_sky.dec = 30.0

        with patch("scann.gui.main_window.pixel_to_wcs", return_value=mock_sky):
            with patch("scann.gui.main_window.QueryService") as mock_svc_cls:
                mock_svc = Mock()
                mock_svc.query_tns.return_value = []
                mock_svc_cls.return_value = mock_svc

                with patch("scann.gui.main_window.QueryResultPopup") as mock_popup_cls:
                    mock_popup = Mock()
                    mock_popup_cls.return_value = mock_popup

                    w._do_query("tns", 50, 50)

                    mock_svc.query_tns.assert_called_once()


# ═══════════════════════════════════════════════
#  功能 16: MPC 报告集成
# ═══════════════════════════════════════════════


class TestMpcReportIntegration:
    """测试 MPC 报告生成集成"""

    def test_mpc_report_with_candidates(self):
        """有候选体时应生成报告并传入对话框"""
        w = _make_mock_window()
        w._candidates = [
            Candidate(x=100, y=200, verdict=TargetVerdict.REAL),
            Candidate(x=300, y=400, verdict=TargetVerdict.REAL),
        ]
        w._new_fits_header = FitsHeader(raw={"CTYPE1": "RA---TAN"})

        with patch("scann.gui.dialogs.mpc_report_dialog.MpcReportDialog") as mock_dlg_cls:
            mock_dlg = Mock()
            mock_dlg_cls.return_value = mock_dlg

            with patch("scann.gui.main_window.pixel_to_wcs") as mock_wcs:
                mock_sky = Mock()
                mock_sky.ra = 180.0
                mock_sky.dec = 45.0
                mock_wcs.return_value = mock_sky

                with patch("scann.gui.main_window.generate_mpc_report") as mock_gen:
                    mock_gen.return_value = "     K24A01A  C2024 01 15.12345 12 00 00.00 +45 00 00.0          20.0R      XXX"
                    w._on_mpc_report()

                    mock_gen.assert_called_once()
                    mock_dlg.set_report.assert_called_once()

    def test_mpc_report_no_candidates(self):
        """无候选体时应显示提示"""
        w = _make_mock_window()
        w._candidates = []
        w._new_fits_header = None

        with patch("scann.gui.dialogs.mpc_report_dialog.MpcReportDialog") as mock_dlg_cls:
            mock_dlg = Mock()
            mock_dlg_cls.return_value = mock_dlg

            w._on_mpc_report()

            # 无候选体时不应调用 set_report
            mock_dlg.set_report.assert_not_called()

    def test_mpc_report_no_wcs(self):
        """无 WCS 时报告应使用像素坐标（或提示）"""
        w = _make_mock_window()
        w._candidates = [
            Candidate(x=100, y=200, verdict=TargetVerdict.REAL),
        ]
        w._new_fits_header = None

        with patch("scann.gui.dialogs.mpc_report_dialog.MpcReportDialog") as mock_dlg_cls:
            mock_dlg = Mock()
            mock_dlg_cls.return_value = mock_dlg

            w._on_mpc_report()

            # 无 WCS 时应显示提示
            w.statusBar().showMessage.assert_called()


# ═══════════════════════════════════════════════
#  功能: 标注工具入口
# ═══════════════════════════════════════════════


class TestAnnotationToolEntry:
    """测试标注工具系统在主窗口中的入口"""

    def test_annotation_menu_exists(self):
        """AI 菜单中应有标注工具菜单项"""
        from scann.gui.main_window import MainWindow
        import inspect
        src = inspect.getsource(MainWindow._init_menu_bar)
        assert "标注工具" in src
        assert "act_annotation" in src

    def test_annotation_shortcut_ctrl_l(self):
        """标注工具应绑定 Ctrl+L 快捷键"""
        from scann.gui.main_window import MainWindow
        import inspect
        src = inspect.getsource(MainWindow._init_menu_bar)
        assert "Ctrl+L" in src

    def test_on_open_annotation_creates_dialog(self):
        """调用 _on_open_annotation 应创建 AnnotationDialog 实例"""
        w = _make_mock_window()

        with patch("scann.gui.dialogs.annotation_dialog.AnnotationDialog") as mock_cls:
            mock_dlg = Mock()
            mock_cls.return_value = mock_dlg

            w._on_open_annotation()

            # 验证调用，包括 config 参数
            mock_cls.assert_called_once()
            call_args = mock_cls.call_args
            assert call_args[0][0] == w
            assert 'config' in call_args[1]
            mock_dlg.show.assert_called_once()

    def test_on_open_annotation_stores_reference(self):
        """打开标注对话框后应保存引用到 _annotation_dialog"""
        w = _make_mock_window()

        with patch("scann.gui.dialogs.annotation_dialog.AnnotationDialog") as mock_cls:
            mock_dlg = Mock()
            mock_cls.return_value = mock_dlg

            w._on_open_annotation()

            assert w._annotation_dialog is mock_dlg

    def test_annotation_dialog_is_non_modal(self):
        """标注对话框应以非模态方式打开 (show 而非 exec_)"""
        w = _make_mock_window()

        with patch("scann.gui.dialogs.annotation_dialog.AnnotationDialog") as mock_cls:
            mock_dlg = Mock()
            mock_cls.return_value = mock_dlg

            w._on_open_annotation()

            # show() 被调用, exec_() 不应被调用
            mock_dlg.show.assert_called_once()
            mock_dlg.exec_.assert_not_called()

    def test_annotation_signal_connected(self):
        """act_annotation.triggered 应连接到 _on_open_annotation"""
        from scann.gui.main_window import MainWindow
        import inspect
        src = inspect.getsource(MainWindow._connect_signals)
        assert "act_annotation" in src
        assert "_on_open_annotation" in src
