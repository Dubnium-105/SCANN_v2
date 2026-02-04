# -*- coding: utf-8 -*-
"""
SCANN 主窗口模块
- SCANN: 主应用窗口类
"""

import sys
import os
import re
import time
import datetime
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QProgressBar, QMessageBox, QFileDialog,
    QCheckBox, QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout, QComboBox,
    QStackedWidget, QMenu, QProgressDialog, QButtonGroup, QApplication
)
from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtGui import QFont, QColor, QPixmap, QImage, QKeySequence

from .config import ConfigManager
from .database import DatabaseManager
from .downloader import LinkedDownloader, DBDownloadWindow
from .widgets import ImageViewer, SuspectListWidget, SuspectGlobalKeyFilter
from .workers import BatchWorker


class SCANN(QMainWindow):
    """SCANN 主应用窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SCANN - Supernova Candidate Analysis via Neural Network")
        self.resize(1400, 900)
        
        self.groups = {}
        self.batch_results = {}
        self.current_group = ""
        self.candidates = []
        self.current_preview_img = None
        self.crop_rect = None
        self.worker = None
        self.io_pool = ThreadPoolExecutor(max_workers=2)
        self.suspects_data = []
        self._suspect_shortcut_backup = {}
        self._t_recall_cached = None
        self._suspect_global_filter = SuspectGlobalKeyFilter(self)

        # 闪烁相关
        self.blink_timer = QTimer(self)
        self.blink_timer.setInterval(400)
        self.blink_timer.timeout.connect(self.blink_tick)
        self.blink_state = 0

        os.makedirs("dataset/positive", exist_ok=True)
        os.makedirs("dataset/negative", exist_ok=True)

        self.cfg = ConfigManager.load()

        # Fix: Robust path finding
        if getattr(sys, 'frozen', False):
            self.base_path = os.path.dirname(sys.executable)
        else:
            self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.model_path = self.cfg.get('model_path', '')
        if not self.model_path or not os.path.exists(self.model_path):
            self.model_path = os.path.join(self.base_path, "best_model.pth")
        
        if os.path.exists(self.model_path):
            print(f"Model found: {self.model_path}")
        else:
            print(f"Model NOT found at: {self.model_path}")

        self.init_ui()
        
        # 初始化下载引擎
        self.downloader = LinkedDownloader()
        self.downloader.status_msg.connect(lambda m: self.statusBar().showMessage(m))
        self.downloader.all_finished.connect(self._on_all_downloads_done)

        if self.cfg['last_folder'] and os.path.exists(self.cfg['last_folder']):
            self.load_folder(self.cfg['last_folder'])

    def init_ui(self):
        """初始化界面"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # === 左侧面板 ===
        left_panel = QVBoxLayout()
        
        btn_load = QPushButton("📂 加载文件夹")
        btn_load.clicked.connect(lambda: self.load_folder())
        left_panel.addWidget(btn_load)

        self.cb_auto_clear = QCheckBox("每次计算前强制清空缓存")
        self.cb_auto_clear.setChecked(self.cfg.get('auto_clear_cache', False))
        self.cb_auto_clear.setStyleSheet("color: blue;")
        left_panel.addWidget(self.cb_auto_clear)

        # 模型选择区域
        model_layout = QHBoxLayout()
        self.lbl_model = QLabel(os.path.basename(self.model_path) if self.model_path else "未选择")
        self.lbl_model.setStyleSheet("color: #666;")
        btn_model = QPushButton("选模型")
        btn_model.setFixedWidth(60)
        btn_model.clicked.connect(self.select_model_file)
        
        model_layout.addWidget(QLabel("模型:"))
        model_layout.addWidget(self.lbl_model)
        model_layout.addWidget(btn_model)
        left_panel.addLayout(model_layout)

        self.btn_batch = QPushButton("⚡ 批量计算")
        self.btn_batch.setFixedHeight(35)
        self.btn_batch.setStyleSheet("background-color: #ffeb3b; font-weight: bold;")
        self.btn_batch.clicked.connect(self.start_batch_run)
        left_panel.addWidget(self.btn_batch)

        # 数据库下载按钮
        self.btn_db_download = QPushButton("🌐 数据库下载")
        self.btn_db_download.setFixedHeight(35)
        self.btn_db_download.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.btn_db_download.clicked.connect(self.open_db_download)
        left_panel.addWidget(self.btn_db_download)

        # 下载路径设置
        path_group = QGroupBox("下载路径")
        path_vbox = QVBoxLayout()
        
        self.lbl_jpg_path = QLabel(f"JPG: {self.cfg['jpg_download_dir'] if self.cfg['jpg_download_dir'] else '未设置'}")
        self.lbl_jpg_path.setToolTip(self.cfg['jpg_download_dir'])
        btn_set_jpg = QPushButton("设置 JPG")
        btn_set_jpg.clicked.connect(lambda: self.set_download_path('jpg_download_dir'))
        
        self.lbl_fits_path = QLabel(f"FITS: {self.cfg['fits_download_dir'] if self.cfg['fits_download_dir'] else '未设置'}")
        self.lbl_fits_path.setToolTip(self.cfg['fits_download_dir'])
        btn_set_fits = QPushButton("设置 FITS")
        btn_set_fits.clicked.connect(lambda: self.set_download_path('fits_download_dir'))
        
        h1 = QHBoxLayout()
        h1.addWidget(self.lbl_jpg_path, 1)
        h1.addWidget(btn_set_jpg)
        h2 = QHBoxLayout()
        h2.addWidget(self.lbl_fits_path, 1)
        h2.addWidget(btn_set_fits)
        path_vbox.addLayout(h1)
        path_vbox.addLayout(h2)
        path_group.setLayout(path_vbox)
        left_panel.addWidget(path_group)

        # 显示可疑目标按钮
        self.btn_show_suspects = QPushButton("🧐 显示可疑目标")
        self.btn_show_suspects.setFixedHeight(35)
        self.btn_show_suspects.setStyleSheet("background-color: #ff9800; color: white; font-weight: bold;")
        self.btn_show_suspects.clicked.connect(self.toggle_suspects_mode)
        left_panel.addWidget(self.btn_show_suspects)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_panel.addWidget(self.progress_bar)

        # === 核心内容区域 (Stack 切换) ===
        self.left_stack = QStackedWidget()
        
        # --- Page 0: 标准视图 ---
        page0 = QWidget()
        p0_layout = QVBoxLayout(page0)
        p0_layout.setContentsMargins(0, 0, 0, 0)
        
        self.file_list = QListWidget()
        self.file_list.currentRowChanged.connect(self.on_file_selected)
        self.file_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.file_list.customContextMenuRequested.connect(self.show_file_list_context_menu)
        
        p0_layout.addWidget(QLabel("文件列表 (绿=有目标, 蓝=已归档):"))
        p0_layout.addWidget(self.file_list, 2)

        # 参数设置
        gb = QGroupBox("检测参数")
        form = QFormLayout()
        
        self.spin_thresh = QSpinBox()
        self.spin_thresh.setRange(5, 255)
        self.spin_thresh.setValue(self.cfg['thresh'])
        self.spin_min_area = QSpinBox()
        self.spin_min_area.setRange(1, 100)
        self.spin_min_area.setValue(self.cfg['min_area'])
        self.spin_sharpness = QDoubleSpinBox()
        self.spin_sharpness.setRange(1.0, 5.0)
        self.spin_sharpness.setSingleStep(0.1)
        self.spin_sharpness.setValue(self.cfg['sharpness'])
        self.spin_max_sharpness = QDoubleSpinBox()
        self.spin_max_sharpness.setRange(1.0, 20.0)
        self.spin_max_sharpness.setSingleStep(0.1)
        self.spin_max_sharpness.setValue(self.cfg.get('max_sharpness', 5.0))
        self.spin_contrast = QSpinBox()
        self.spin_contrast.setRange(0, 100)
        self.spin_contrast.setValue(self.cfg['contrast'])
        self.spin_edge = QSpinBox()
        self.spin_edge.setRange(0, 100)
        self.spin_edge.setValue(self.cfg.get('edge_margin', 10))
        
        self.cb_dynamic_thresh = QCheckBox("动态阈值 (Median+Offset)")
        self.cb_dynamic_thresh.setChecked(self.cfg.get('dynamic_thresh', False))
        self.cb_dynamic_thresh.setToolTip("开启后，阈值 = 背景中位数 + 设定值")
        self.cb_kill_flat = QCheckBox("去除平坦光斑")
        self.cb_kill_flat.setChecked(self.cfg['kill_flat'])
        self.cb_kill_history = QCheckBox("去除历史 (区域)")
        self.cb_kill_history.setChecked(self.cfg['kill_hist'])
        self.cb_kill_history.setStyleSheet("color: red;")
        self.cb_kill_dipole = QCheckBox("去除偶极子")
        self.cb_kill_dipole.setChecked(self.cfg['kill_dipole'])
        self.cb_auto_crop = QCheckBox("自动切除白边")
        self.cb_auto_crop.setChecked(self.cfg['auto_crop'])
        
        self.spin_crowd_high_score = QDoubleSpinBox()
        self.spin_crowd_high_score.setRange(0.0, 1.0)
        self.spin_crowd_high_score.setSingleStep(0.01)
        self.spin_crowd_high_score.setValue(self.cfg.get('crowd_high_score', 0.85))
        self.spin_crowd_high_count = QSpinBox()
        self.spin_crowd_high_count.setRange(1, 500)
        self.spin_crowd_high_count.setValue(self.cfg.get('crowd_high_count', 10))
        self.spin_crowd_high_penalty = QDoubleSpinBox()
        self.spin_crowd_high_penalty.setRange(0.0, 1.0)
        self.spin_crowd_high_penalty.setSingleStep(0.01)
        self.spin_crowd_high_penalty.setValue(self.cfg.get('crowd_high_penalty', 0.50))
        
        form.addRow("亮度阈值/Offset:", self.spin_thresh)
        form.addRow("最小面积:", self.spin_min_area)
        form.addRow("Min 锐度:", self.spin_sharpness)
        form.addRow("Max 锐度:", self.spin_max_sharpness)
        form.addRow("对比度:", self.spin_contrast)
        form.addRow("边缘忽略(px):", self.spin_edge)
        form.addRow(self.cb_dynamic_thresh)
        form.addRow(self.cb_kill_flat)
        form.addRow(self.cb_kill_history)
        form.addRow(self.cb_kill_dipole)
        form.addRow(self.cb_auto_crop)
        form.addRow("拥挤惩罚 高分阈值:", self.spin_crowd_high_score)
        form.addRow("拥挤惩罚 数量阈值:", self.spin_crowd_high_count)
        form.addRow("拥挤惩罚 扣分幅度:", self.spin_crowd_high_penalty)
        
        gb.setLayout(form)
        p0_layout.addWidget(gb)

        self.cand_list = QListWidget()
        self.cand_list.currentRowChanged.connect(self.on_candidate_selected)
        self.cand_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.cand_list.customContextMenuRequested.connect(lambda pos: self.show_list_context_menu(self.cand_list, pos))
        p0_layout.addWidget(QLabel("候选体 (点击大图可手动添加):"))
        p0_layout.addWidget(self.cand_list, 1)
        
        # --- Page 1: 可疑目标列表 ---
        page1 = QWidget()
        p1_layout = QVBoxLayout(page1)
        p1_layout.setContentsMargins(0, 0, 0, 0)
        self.suspect_list = SuspectListWidget(self)
        self.suspect_list.currentItemChanged.connect(self.on_suspect_current_changed)
        self.suspect_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.suspect_list.customContextMenuRequested.connect(lambda pos: self.show_list_context_menu(self.suspect_list, pos))
        p1_layout.addWidget(QLabel("🔥 高价值可疑目标 (按 AI 排序):"))
        p1_layout.addWidget(self.suspect_list)
        
        self.left_stack.addWidget(page0)
        self.left_stack.addWidget(page1)
        
        left_panel.addWidget(self.left_stack, 1)
        layout.addLayout(left_panel, 1)

        # === 右侧面板 ===
        right_panel = QVBoxLayout()
        self.lbl_title = QLabel("准备就绪")
        self.lbl_title.setFont(QFont("Arial", 16, QFont.Bold))
        self.lbl_title.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(self.lbl_title)

        self.lbl_triplet = QLabel()
        self.lbl_triplet.setAlignment(Qt.AlignCenter)
        self.lbl_triplet.setStyleSheet("background: black; border: 1px solid #666;")
        self.lbl_triplet.setMinimumHeight(280)
        right_panel.addWidget(self.lbl_triplet, 0)

        # 标记按钮区
        btn_layout = QHBoxLayout()
        self.btn_save = QPushButton("✅ 真目标 (S)")
        self.btn_save.setFixedHeight(50)
        self.btn_save.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; font-size: 14px;")
        self.btn_save.clicked.connect(lambda: self.save_dataset_sample(True))
        self.btn_save.setShortcut("S")

        self.btn_skip = QPushButton("🤷‍♂️ 跳过 (Space)")
        self.btn_skip.setFixedHeight(50)
        self.btn_skip.setStyleSheet("background-color: #9E9E9E; color: white; font-weight: bold; font-size: 14px;")
        self.btn_skip.clicked.connect(self.skip_sample)
        self.btn_skip.setShortcut(Qt.Key_Space)
        btn_layout.addWidget(self.btn_skip)

        self.btn_next = QPushButton("❌ 假目标 (D)")
        self.btn_next.setFixedHeight(50)
        self.btn_next.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; font-size: 14px;")
        self.btn_next.clicked.connect(lambda: self.save_dataset_sample(False))
        self.btn_next.setShortcut("D")

        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_next)
        right_panel.addLayout(btn_layout)

        # === 视图切换按钮 ===
        view_layout = QHBoxLayout()
        self.btn_view_a = QPushButton("[1] Diff (A)")
        self.btn_view_b = QPushButton("[2] New (B)")
        self.btn_view_c = QPushButton("[3] Ref (C)")
        
        for b, slot, sc in [(self.btn_view_a, lambda: self.switch_main_view(0), "1"),
                            (self.btn_view_b, lambda: self.switch_main_view(1), "2"),
                            (self.btn_view_c, lambda: self.switch_main_view(2), "3")]:
            b.clicked.connect(slot)
            b.setShortcut(sc)
            b.setCheckable(True)
            view_layout.addWidget(b)
            
        # 闪烁按钮
        self.btn_blink = QPushButton("✨ 闪烁 (R)")
        self.btn_blink.setCheckable(True)
        self.btn_blink.clicked.connect(self.toggle_blink)
        self.btn_blink.setShortcut("R")
        self.btn_blink.setStyleSheet("""
            QPushButton:checked { background-color: #ff9800; color: white; }
        """)
        view_layout.addWidget(self.btn_blink)
        
        # 闪烁速度选择
        self.combo_blink_speed = QComboBox()
        self.combo_blink_speed.addItems(["0.1s", "0.25s", "0.5s", "0.75s", "1.0s", "2.0s"])
        self.combo_blink_speed.setCurrentText("0.5s")
        self.combo_blink_speed.setFixedWidth(60)
        self.combo_blink_speed.currentIndexChanged.connect(self.update_blink_speed)
        view_layout.addWidget(self.combo_blink_speed)

        # 隐藏标记按钮
        self.btn_hide_overlay = QCheckBox("隐藏圈 (H)")
        self.btn_hide_overlay.clicked.connect(self.toggle_overlay)
        self.btn_hide_overlay.setShortcut("H")
        self.btn_hide_overlay.setStyleSheet("color: #e91e63; font-weight: bold;")
        view_layout.addWidget(self.btn_hide_overlay)

        # 默认选中 A
        self.btn_view_a.setChecked(True)
        self.view_group = QButtonGroup(self)
        self.view_group.addButton(self.btn_view_a)
        self.view_group.addButton(self.btn_view_b)
        self.view_group.addButton(self.btn_view_c)
        
        right_panel.addLayout(view_layout)

        self.view_context = ImageViewer()
        self.view_context.point_selected.connect(self.on_context_click)
        
        right_panel.addWidget(self.view_context, 1)
        right_panel.addWidget(QLabel("提示：左键点击=设点 | 滚轮=缩放 | 右键拖拽=平移"))

        layout.addLayout(right_panel, 3)

    # ===== 配置与文件管理 =====
    
    def select_model_file(self):
        """选择模型文件"""
        path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", self.base_path, "PyTorch Model (*.pth)")
        if path:
            self.model_path = path
            self.cfg['model_path'] = path
            self.lbl_model.setText(os.path.basename(path))
            self.lbl_model.setToolTip(path)
            ConfigManager.save(self.cfg)
            print(f"Model switched to: {path}")

    def save_current_config(self):
        """保存当前配置"""
        self.cfg['thresh'] = self.spin_thresh.value()
        self.cfg['min_area'] = self.spin_min_area.value()
        self.cfg['sharpness'] = self.spin_sharpness.value()
        self.cfg['contrast'] = self.spin_contrast.value()
        self.cfg['edge_margin'] = self.spin_edge.value()
        self.cfg['kill_flat'] = self.cb_kill_flat.isChecked()
        self.cfg['kill_hist'] = self.cb_kill_history.isChecked()
        self.cfg['kill_dipole'] = self.cb_kill_dipole.isChecked()
        self.cfg['auto_crop'] = self.cb_auto_crop.isChecked()
        self.cfg['auto_clear_cache'] = self.cb_auto_clear.isChecked()
        self.cfg['max_sharpness'] = self.spin_max_sharpness.value()
        self.cfg['dynamic_thresh'] = self.cb_dynamic_thresh.isChecked()
        self.cfg['crowd_high_score'] = self.spin_crowd_high_score.value()
        self.cfg['crowd_high_count'] = self.spin_crowd_high_count.value()
        self.cfg['crowd_high_penalty'] = self.spin_crowd_high_penalty.value()
        ConfigManager.save(self.cfg)

    def set_download_path(self, key):
        """设置下载路径"""
        path = QFileDialog.getExistingDirectory(self, "选择保存目录", self.cfg.get(key, ""))
        if path:
            self.cfg[key] = path
            ConfigManager.save(self.cfg)
            if key == 'jpg_download_dir':
                self.lbl_jpg_path.setText(f"JPG: {path}")
                self.lbl_jpg_path.setToolTip(path)
            else:
                self.lbl_fits_path.setText(f"FITS: {path}")
                self.lbl_fits_path.setToolTip(path)

    def closeEvent(self, event):
        """关闭事件处理"""
        self.save_current_config()
        
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()

        pending = DatabaseManager.get_pending_count()
        if pending > 0:
            initial_pending = pending
            print(f"Waiting for {pending} database writes to finish...")
            progress = QProgressDialog(f"正在保存数据，剩余 {pending} 条...", "强制退出", 0, initial_pending, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            while pending > 0:
                QApplication.processEvents()
                time.sleep(0.1)
                new_pending = DatabaseManager.get_pending_count()
                progress.setValue(initial_pending - new_pending)
                progress.setLabelText(f"正在保存数据，剩余 {new_pending} 条...")
                if progress.wasCanceled():
                    break
                pending = new_pending
                
        DatabaseManager.stop_async()
        super().closeEvent(event)

    # ===== 文件加载 =====
    
    def load_folder(self, path=None):
        """加载图片文件夹"""
        if not path:
            path = QFileDialog.getExistingDirectory(self, "选择图片文件夹", self.cfg['last_folder'])
        if not path:
            return
        self.cfg['last_folder'] = path
        
        old_selection = None
        curr_item = self.file_list.currentItem()
        if curr_item:
            old_selection = curr_item.data(Qt.UserRole)
            
        self.groups = {}
        self.batch_results = {}
        self.file_list.clear()
        files = os.listdir(path)
        for f in files:
            if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            name, _ = os.path.splitext(f)
            if len(name) < 2:
                continue
            suffix = name[-1].lower()
            stem = name[:-1]
            if suffix in ['a', 'b', 'c']:
                if stem not in self.groups:
                    self.groups[stem] = {}
                self.groups[stem][suffix] = os.path.join(path, f)
        count = 0
        sorted_keys = sorted(self.groups.keys())
        db = DatabaseManager.load_summaries_map()
        
        target_row = -1
        for k in sorted_keys:
            if len(self.groups[k]) == 3:
                item = QListWidgetItem(k)
                item.setData(Qt.UserRole, k)
                if k in db:
                    rec = db[k]
                    status = rec.get("status", "unseen")
                    self.batch_results[k] = rec
                    if status == "processed":
                        item.setText(f"{k} [已归档]")
                        item.setForeground(QColor(0, 100, 255))
                        item.setFont(QFont("Arial", 10, QFont.Bold))
                    elif rec.get("candidates_count", 0) > 0:
                        item.setText(f"{k} [{rec.get('candidates_count', 0)}个目标]")
                        item.setForeground(QColor(0, 200, 0))
                        item.setFont(QFont("Arial", 10, QFont.Bold))
                    else:
                        item.setText(k)
                
                self.file_list.addItem(item)
                if old_selection and k == old_selection:
                    target_row = count
                count += 1
        
        if count > 0:
            self.lbl_title.setText(f"加载了 {count} 组图片")
            if target_row != -1:
                self.file_list.setCurrentRow(target_row)
            elif self.file_list.count() > 0:
                self.file_list.setCurrentRow(0)
        else:
            QMessageBox.warning(self, "警告", "文件夹内未找到完整三联图")

    def delete_source_files(self, name, list_item):
        """删除源文件"""
        if name not in self.groups:
            return
        paths = self.groups[name]
        errs = []
        for k in ['a', 'b', 'c']:
            path = paths.get(k)
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    errs.append(str(e))
        if errs:
            QMessageBox.warning(self, "删除出错", "\n".join(errs))
            return
        del self.groups[name]
        DatabaseManager.delete_record(name)
        if name in self.batch_results:
            del self.batch_results[name]
        row = self.file_list.row(list_item)
        self.file_list.takeItem(row)
        if self.current_group == name:
            self.current_preview_img = None
            self.view_context.scene.clear()
            self.lbl_triplet.clear()
            self.cand_list.clear()
            self.candidates = []

    # ===== 批量处理 =====
    
    def start_batch_run(self):
        """开始批量计算"""
        if not self.groups:
            return
        self.save_current_config()
        if self.cb_auto_clear.isChecked():
            DatabaseManager.clear_all()
            self.batch_results = {}
            for i in range(self.file_list.count()):
                item = self.file_list.item(i)
                name = item.data(Qt.UserRole)
                if not name:
                    name = item.text().split(" ")[0]
                item.setText(name)
                item.setForeground(QColor(0, 0, 0))
                item.setFont(QFont("Arial", 10))
            print("缓存已自动清除")

        # 过滤完整组
        complete_groups = {}
        incomplete_details = []
        for name, paths in self.groups.items():
            missing = []
            for k in ['a', 'b', 'c']:
                if k not in paths:
                    missing.append(k)
                elif not os.path.exists(paths[k]):
                    missing.append(f"{k}(path missing)")
            
            if not missing:
                complete_groups[name] = paths
            else:
                incomplete_details.append(f"{name}: missing {missing}")

        if incomplete_details:
            print("=== Incomplete Groups (Skipped) ===")
            for msg in incomplete_details[:20]:
                print(msg)
            if len(incomplete_details) > 20:
                print(f"... and {len(incomplete_details)-20} more.")
            print("===================================")
            QMessageBox.warning(self, "数据完整性检查",
                                f"发现 {len(incomplete_details)} 组数据不完整（缺图），已自动跳过。\n"
                                f"本次将处理 {len(complete_groups)} 组有效数据。")

        if not complete_groups:
            QMessageBox.warning(self, "错误", "没有找到任何完整的三联图组！无法开始。")
            return

        self.btn_batch.setEnabled(False)
        self.btn_batch.setText("计算中...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()

        try:
            self.worker = BatchWorker(complete_groups, self.cfg)
            self.worker.progress.connect(self.update_progress)
            self.worker.finished.connect(self.on_batch_finished)
            self.worker.start()
        except Exception as e:
            print(f"Failed to initialize BatchWorker: {e}")
            QMessageBox.critical(self, "启动失败", f"无法初始化计算引擎：\n{str(e)}\n\n请检查模型文件路径是否正确。")
            self.btn_batch.setEnabled(True)
            self.btn_batch.setText("批量计算")
            self.progress_bar.setVisible(False)

    def update_progress(self, curr, total, msg):
        """更新进度"""
        self.progress_bar.setValue(int(curr / total * 100))
        self.lbl_title.setText(msg)

    def on_batch_finished(self, results):
        """批量处理完成回调"""
        self.btn_batch.setEnabled(True)
        self.btn_batch.setText("⚡ 批量计算")
        self.progress_bar.setVisible(False)
        self.batch_results.update(results)
        first_hit_row = -1
        total_hits = 0
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            name = item.data(Qt.UserRole)
            if not name:
                name = item.text().split(" ")[0]
            rec = results.get(name)
            if not rec:
                rec = self.batch_results.get(name)
            if rec:
                cands = rec.get("candidates", [])
                status = rec.get("status", "unseen")
                if status == "processed":
                    item.setText(f"{name} [已归档]")
                    item.setForeground(QColor(0, 100, 255))
                    item.setFont(QFont("Arial", 10, QFont.Bold))
                elif cands:
                    item.setText(f"{name} [{len(cands)}个目标]")
                    item.setForeground(QColor(0, 200, 0))
                    item.setFont(QFont("Arial", 10, QFont.Bold))
                    total_hits += len(cands)
                    if first_hit_row == -1:
                        first_hit_row = i
                else:
                    item.setText(name)
                    item.setForeground(QColor(0, 0, 0))
                    item.setFont(QFont("Arial", 10))
        QMessageBox.information(self, "完成", f"处理结束，本次发现 {total_hits} 个新目标")
        if first_hit_row != -1:
            self.file_list.setCurrentRow(first_hit_row)
            self.load_candidates_from_batch(self.file_list.item(first_hit_row).data(Qt.UserRole))

    # ===== 下载功能 =====
    
    def open_db_download(self):
        """打开数据库下载窗口"""
        win = DBDownloadWindow(self.downloader, self)
        win.exec_()

    def _on_all_downloads_done(self, success_count, fail_count):
        """所有下载完成回调"""
        msg = f"🎉 批量下载任务已完成！成功: {success_count}"
        if fail_count > 0:
            msg += f" | 失败: {fail_count}"
        self.statusBar().showMessage(msg, 10000)
        
        current_folder = self.cfg.get('last_folder')
        if current_folder and os.path.exists(current_folder):
            self.load_folder(current_folder)

    def download_fits_for_item(self, list_widget, item):
        """下载 FITS 文件"""
        if list_widget == self.suspect_list:
            text = item.text()
            match = re.search(r'\[(.*?)\]', text)
            if not match:
                return
            stem = match.group(1)
        elif list_widget == self.file_list:
            stem = item.data(Qt.UserRole)
            if not stem:
                stem = item.text().split(" ")[0]
        else:
            stem = self.current_group

        linkage = self.downloader.get_linkage(stem)
        if not linkage:
            self.statusBar().showMessage(f"❌ 数据库中未找到 {stem} 的下载链接，请先在数据库浏览器中扫描。")
            return

        if linkage['status'] == 'downloaded' and linkage['local_fits_path'] and os.path.exists(linkage['local_fits_path']):
            self.statusBar().showMessage(f"✅ FITS 原图已存在: {linkage['local_fits_path']}")
            return

        save_dir = self.cfg.get('fits_download_dir')
        if not save_dir or not os.path.exists(save_dir):
            save_dir = QFileDialog.getExistingDirectory(self, "选择 FITS 保存目录")
            if not save_dir:
                return
            self.cfg['fits_download_dir'] = save_dir
            ConfigManager.save(self.cfg)
            self.lbl_fits_path.setText(f"FITS: {save_dir}")
            self.lbl_fits_path.setToolTip(save_dir)

        filename = self.downloader.clean_filename(os.path.basename(linkage['remote_fits_url']))
        if not filename.lower().endswith(".fts"):
            filename += ".fts"
        save_path = os.path.join(save_dir, filename)
        
        if os.path.exists(save_path):
            self.downloader.update_linkage(stem, status='downloaded', local_fits_path=save_path)
            self.statusBar().showMessage(f"✅ FITS 原图已存在于所选目录: {save_path}")
            return

        self.downloader.submit_download(stem, linkage['remote_fits_url'], save_dir)

    # ===== 视图控制 =====
    
    def switch_main_view(self, mode):
        """切换主视图"""
        if not hasattr(self, 'img_a'):
            return
        
        if self.blink_timer.isActive():
            self.btn_blink.setChecked(False)
            self.blink_timer.stop()
        
        target_img = None
        if mode == 0:
            target_img = self.img_a
        elif mode == 1:
            target_img = self.img_b
        elif mode == 2:
            target_img = self.img_c
        
        if target_img is not None:
            self.current_preview_img = target_img
            self.view_context.set_image(target_img)
            curr_row = self.cand_list.currentRow()
            self.view_context.draw_overlays(self.candidates, curr_row, self.btn_hide_overlay.isChecked())
            
            if mode == 0:
                self.btn_view_a.setChecked(True)
            elif mode == 1:
                self.btn_view_b.setChecked(True)
            elif mode == 2:
                self.btn_view_c.setChecked(True)

    def toggle_overlay(self):
        """切换圆圈显示"""
        self.view_context.draw_overlays(self.candidates, self.cand_list.currentRow(), self.btn_hide_overlay.isChecked())

    def toggle_blink(self):
        """切换闪烁"""
        if self.btn_blink.isChecked():
            self.update_blink_speed()
            self.blink_timer.start()
        else:
            self.blink_timer.stop()
            self.switch_main_view(0)
    
    def update_blink_speed(self):
        """更新闪烁速度"""
        text = self.combo_blink_speed.currentText()
        try:
            sec = float(text.replace('s', ''))
            self.blink_timer.setInterval(int(sec * 1000))
        except Exception:
            self.blink_timer.setInterval(400)

    def blink_tick(self):
        """闪烁定时器回调"""
        if not hasattr(self, 'img_b') or not hasattr(self, 'img_c'):
            return
        
        self.blink_state = 1 - self.blink_state
        if self.blink_state == 0:
            self.view_context.set_image(self.img_b)
            self.btn_view_b.setChecked(True)
        else:
            self.view_context.set_image(self.img_c)
            self.btn_view_c.setChecked(True)
            
        curr_row = self.cand_list.currentRow()
        self.view_context.draw_overlays(self.candidates, curr_row, self.btn_hide_overlay.isChecked())

    # ===== 文件选择处理 =====
    
    def on_file_selected(self, row):
        """文件选择事件"""
        item = self.file_list.item(row)
        if item is None:
            return
        
        self.view_context.scene.setSceneRect(QRectF())

        stem = item.data(Qt.UserRole)
        if not stem:
            stem = item.text().split(" ")[0]
        self.current_group = stem
        self.lbl_title.setText(stem)
        paths = self.groups[stem]
        self.raw_a = cv2.imread(paths['a'])
        self.raw_b = cv2.imread(paths['b'])
        self.raw_c = cv2.imread(paths['c'])

        if self.raw_a is None or self.raw_b is None or self.raw_c is None:
            missing = [k for k, v in zip(['a', 'b', 'c'], [self.raw_a, self.raw_b, self.raw_c]) if v is None]
            self.statusBar().showMessage(f"❌ 无法读取图片: {stem} (缺失/损坏: {missing})")
            self.lbl_triplet.setText(f"读取失败: {missing}")
            return

        record = self.batch_results.get(stem)
        use_cached_crop = False
        if record and "crop_rect" in record and record["crop_rect"] is not None:
            self.crop_rect = record["crop_rect"]
            use_cached_crop = True
        else:
            if self.cb_auto_crop.isChecked():
                self.crop_rect = self.get_auto_crop_rect(self.raw_a)
            else:
                self.crop_rect = (0, 0, self.raw_a.shape[1], self.raw_a.shape[0])
        
        x, y, w, h = self.crop_rect
        self.img_a = self.raw_a[y:y+h, x:x+w]
        self.img_b = self.raw_b[y:y+h, x:x+w]
        self.img_c = self.raw_c[y:y+h, x:x+w]
        
        if self.btn_blink.isChecked():
            if not self.blink_timer.isActive():
                self.blink_timer.start()
            if self.blink_state == 0:
                self.view_context.set_image(self.img_b)
                self.btn_view_b.setChecked(True)
            else:
                self.view_context.set_image(self.img_c)
                self.btn_view_c.setChecked(True)
            self.view_context.draw_overlays(self.candidates, self.cand_list.currentRow(), self.btn_hide_overlay.isChecked())
        else:
            self.blink_timer.stop()
            self.switch_main_view(0)
        
        if stem in self.batch_results:
            self.load_candidates_from_batch(stem)
        else:
            self.candidates = []
            self.cand_list.clear()
            self.lbl_triplet.clear()
            self.lbl_triplet.setText("无数据 (请点击批量计算)")
            self.view_context.draw_overlays([], -1)

    def get_auto_crop_rect(self, img):
        """获取自动裁剪区域"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0, 0, img.shape[1], img.shape[0]
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        pad = 2
        return max(0, x + pad), max(0, y + pad), max(1, w - 2 * pad), max(1, h - 2 * pad)

    def load_candidates_from_batch(self, name):
        """从批量结果加载候选体"""
        rec = self.batch_results.get(name, {})
        if "candidates" not in rec:
            full = DatabaseManager.get_record(name)
            if full:
                if "crop_rect" in rec and rec.get("crop_rect") is not None and full.get("crop_rect") is None:
                    full["crop_rect"] = rec.get("crop_rect")
                self.batch_results[name] = full
                rec = full
        self.candidates = rec.get("candidates", [])
        if self._is_suspect_mode_active():
            self.candidates.sort(key=lambda c: (1 if c.get('manual', False) else 0, -c.get('ai_score', 0)))
            for i, c in enumerate(self.candidates):
                c['id'] = i + 1
            return

        self.refresh_cand_list()
        if self.candidates:
            self.cand_list.setCurrentRow(0)
        else:
            self.view_context.draw_overlays([], -1)

    def _is_suspect_mode_active(self):
        """检查是否处于可疑目标模式"""
        try:
            return hasattr(self, 'left_stack') and self.left_stack.currentIndex() == 1
        except Exception:
            return False

    def refresh_cand_list(self):
        """刷新候选体列表"""
        self.cand_list.clear()
        
        if self._t_recall_cached is None:
            tR = 0.5
            try:
                if os.path.exists(self.model_path):
                    meta = torch.load(self.model_path, map_location='cpu')
                    if isinstance(meta, dict):
                        tR = meta.get('t_recall', 0.5)
            except Exception:
                tR = 0.5
            self._t_recall_cached = tR
        else:
            tR = self._t_recall_cached
            
        self.candidates.sort(key=lambda c: (1 if c.get('manual', False) else 0, -c.get('ai_score', 0)))
        
        for i, c in enumerate(self.candidates):
            c['id'] = i + 1
            
            ai_score = c.get('ai_score', 0)
            score_str = f"{ai_score * 100:.2f}%"
            
            sharp = c.get('sharp', 0)
            area = c.get('area', 0)
            peak = c.get('peak', 0)
            rise = c.get('rise', 0)
            
            verdict = c.get('verdict', None)
            verdict_suffix = ""
            verdict_color = None
            if verdict == 'real':
                verdict_suffix = " [已存真]"
                verdict_color = QColor(0, 100, 0)
            elif verdict == 'bogus':
                verdict_suffix = " [已存假]"
                verdict_color = QColor(100, 0, 0)
            
            if c.get('manual', False):
                txt = f"#{i+1} [手动添加]{verdict_suffix}"
                item = QListWidgetItem(txt)
                if verdict_color:
                    item.setForeground(verdict_color)
                else:
                    item.setForeground(QColor(255, 0, 255))
            else:
                txt = f"#{i+1} AI:{score_str} S:{sharp:.1f} A:{int(area)} D:{int(peak)} R:{int(rise)}{verdict_suffix}"
                item = QListWidgetItem(txt)
                
                if verdict_color:
                    item.setForeground(verdict_color)
                elif ai_score >= tR:
                    item.setForeground(QColor(255, 0, 0))
                    item.setFont(QFont("Arial", 11, QFont.Bold))
                else:
                    item.setForeground(QColor(128, 128, 128))
            
            self.cand_list.addItem(item)

    def on_candidate_selected(self, row):
        """候选体选择事件"""
        if row < 0 or row >= len(self.candidates):
            return
        cand = self.candidates[row]
        cx, cy = cand['x'], cand['y']
        
        self.view_context.draw_overlays(self.candidates, row, self.btn_hide_overlay.isChecked())

        label_text = f"Manual #{cand['id']}" if cand.get('manual', False) else f"Diff #{cand['id']}"
        
        p_a = self._crop_patch_common(self.img_a, cx, cy)
        p_b = self._crop_patch_common(self.img_b, cx, cy)
        p_c = self._crop_patch_common(self.img_c, cx, cy)
        
        disp_sz = 200
        disp_a = cv2.resize(p_a, (disp_sz, disp_sz), interpolation=cv2.INTER_NEAREST)
        disp_b = cv2.resize(p_b, (disp_sz, disp_sz), interpolation=cv2.INTER_NEAREST)
        disp_c = cv2.resize(p_c, (disp_sz, disp_sz), interpolation=cv2.INTER_NEAREST)
        
        center = disp_sz // 2
        radius = int(15 * (disp_sz / 80))
        
        cv2.circle(disp_a, (center, center), radius, (0, 255, 0), 2)
        cv2.putText(disp_a, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        cv2.circle(disp_b, (center, center), radius, (0, 255, 0), 2)
        cv2.putText(disp_b, "New", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        cv2.circle(disp_c, (center, center), radius, (0, 255, 0), 2)
        cv2.putText(disp_c, "Ref", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        combined = np.hstack([disp_a, disp_b, disp_c])
        if not combined.flags['C_CONTIGUOUS']:
            combined = np.ascontiguousarray(combined)
        h_c, w_c, ch = combined.shape
        qimg = QImage(combined.data.tobytes(), w_c, h_c, ch * w_c, QImage.Format_RGB888)
        self.lbl_triplet.setPixmap(QPixmap.fromImage(qimg))

    def _crop_patch_common(self, src_img, cx, cy, crop_sz=80):
        """通用裁剪函数"""
        half = crop_sz // 2
        curr_h, curr_w = src_img.shape[:2]
        
        canvas = np.zeros((crop_sz, crop_sz, 3), dtype=np.uint8)
        x1 = cx - half
        y1 = cy - half
        x2 = x1 + crop_sz
        y2 = y1 + crop_sz
        
        src_x1 = max(0, x1)
        src_y1 = max(0, y1)
        src_x2 = min(curr_w, x2)
        src_y2 = min(curr_h, y2)
        
        dst_x1 = src_x1 - x1
        dst_y1 = src_y1 - y1
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        
        p_h = src_y2 - src_y1
        p_w = src_x2 - src_x1
        c_h = dst_y2 - dst_y1
        c_w = dst_x2 - dst_x1
        
        if p_h > 0 and p_w > 0 and p_h == c_h and p_w == c_w:
            patch_data = src_img[src_y1:src_y2, src_x1:src_x2]
            if len(patch_data.shape) == 2:
                patch_data = cv2.cvtColor(patch_data, cv2.COLOR_GRAY2BGR)
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = patch_data
            
        return canvas

    # ===== 可疑目标模式 =====
    
    def toggle_suspects_mode(self):
        """切换可疑目标模式"""
        if self.left_stack.currentIndex() == 1:
            try:
                QApplication.instance().removeEventFilter(self._suspect_global_filter)
            except Exception:
                pass
            self.left_stack.setCurrentIndex(0)
            self.btn_show_suspects.setText("🧐 显示可疑目标")
            self.btn_show_suspects.setStyleSheet("background-color: #ff9800; color: white; font-weight: bold;")
            if self._suspect_shortcut_backup:
                for w, sc in self._suspect_shortcut_backup.items():
                    try:
                        w.setShortcut(sc)
                    except Exception:
                        pass
                self._suspect_shortcut_backup = {}
            return

        try:
            QApplication.instance().installEventFilter(self._suspect_global_filter)
        except Exception:
            pass

        self._suspect_shortcut_backup = {
            self.btn_save: self.btn_save.shortcut(),
            self.btn_next: self.btn_next.shortcut(),
            self.btn_skip: self.btn_skip.shortcut(),
            self.btn_blink: self.btn_blink.shortcut()
        }
        for w in self._suspect_shortcut_backup.keys():
            try:
                w.setShortcut(QKeySequence())
            except Exception:
                pass

        all_suspects = []
        try:
            summaries = DatabaseManager.load_summaries_map()
        except Exception:
            summaries = {}
        for name, s in summaries.items():
            if int(s.get("has_ai", 0)) != 1:
                continue
            rec = DatabaseManager.get_record(name) or {}
            cands = rec.get("candidates", [])
            sorted_cands = sorted(cands, key=lambda x: x.get("ai_score", 0), reverse=True)
            for i, c in enumerate(sorted_cands):
                if ("ai_score" in c) and (not c.get("manual", False)):
                    w = c.copy()
                    w["stem"] = name
                    w["id"] = i + 1
                    all_suspects.append(w)
        
        if not all_suspects:
            QMessageBox.warning(self, "没有候选目标", "当前没有含 AI 评分的目标，请先运行批量计算。")
            return

        all_suspects.sort(key=lambda x: x.get('ai_score', 0), reverse=True)
        
        limit = 500
        total_count = len(all_suspects)
        if len(all_suspects) > limit:
            print(f"Warning: Too many suspects ({len(all_suspects)}), showing top {limit}")
            all_suspects = all_suspects[:limit]
            
        self.suspects_data = all_suspects
        self.suspect_list.clear()
        
        for cand in all_suspects:
            score = cand.get('ai_score', 0)
            stem = cand.get('stem', 'Unknown')
            cid = cand.get('id', '?')
            
            verdict = cand.get('verdict', None)
            verdict_mark = ""
            fg_color = None
            
            if verdict == 'real':
                verdict_mark = " [已存真]"
                fg_color = QColor(0, 150, 0)
            elif verdict == 'bogus':
                verdict_mark = " [已存假]"
                fg_color = QColor(150, 0, 0)
            
            item_text = f"[{stem}] ID:{cid} | AI: {score*100:.2f}%{verdict_mark}"
            item = QListWidgetItem(item_text)
            
            if fg_color:
                item.setForeground(fg_color)
            elif score >= 0.5:
                item.setForeground(QColor(255, 0, 0))
                item.setFont(QFont("Arial", 11, QFont.Bold))
            else:
                item.setForeground(QColor(128, 128, 128))
                
            self.suspect_list.addItem(item)
            
        self.left_stack.setCurrentIndex(1)
        self.btn_show_suspects.setText(f"🔙 退出可疑列表 (Top {limit}/{total_count})")
        self.btn_show_suspects.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        
        if self.suspect_list.count() > 0:
            self.suspect_list.setCurrentRow(0)
            self.suspect_list.setFocus()

    def on_suspect_current_changed(self, current, previous):
        """可疑目标选择改变"""
        if not current:
            return
        row = self.suspect_list.row(current)
        if row < 0 or row >= len(self.suspects_data):
            return
        
        cand = self.suspects_data[row]
        self.jump_to_candidate(cand, activate_main=False)
        self.suspect_list.setFocus()

    def handle_suspect_action(self, is_positive):
        """处理可疑目标操作"""
        row = self.suspect_list.currentRow()
        if row < 0 and self.suspect_list.count() > 0:
            self.suspect_list.setCurrentRow(0)
            row = 0
        if row < 0 or row >= len(self.suspects_data):
            return
        
        cand_wrapper = self.suspects_data[row]
        
        self.save_dataset_sample(is_positive, auto_jump=False, explicit_candidate=cand_wrapper)
        cand_wrapper['verdict'] = 'real' if is_positive else 'bogus'
        
        item = self.suspect_list.item(row)
        text = item.text()
        if "[已" not in text:
            suffix = " [已存真]" if is_positive else " [已存假]"
            item.setText(text + suffix)
        
        color = QColor(0, 150, 0) if is_positive else QColor(150, 0, 0)
        item.setForeground(color)
        
        self.handle_suspect_skip()

    def handle_suspect_skip(self):
        """跳过可疑目标"""
        row = self.suspect_list.currentRow()
        if row < 0 and self.suspect_list.count() > 0:
            self.suspect_list.setCurrentRow(0)
            row = 0

        if row < self.suspect_list.count() - 1:
            self.suspect_list.setCurrentRow(row + 1)
            try:
                it = self.suspect_list.currentItem()
                if it:
                    self.suspect_list.scrollToItem(it)
            except Exception:
                pass
            self.suspect_list.setFocus()
            QApplication.processEvents()
        else:
            QMessageBox.information(self, "提示", "已到达可疑目标列表底部")

    def jump_to_candidate(self, cand_wrapper, activate_main=True):
        """跳转到候选体"""
        target_stem = cand_wrapper.get('stem')
        target_x = cand_wrapper.get('x')
        target_y = cand_wrapper.get('y')
        
        if not target_stem:
            return
        
        if target_stem != self.current_group:
            found_row = -1
            for i in range(self.file_list.count()):
                item = self.file_list.item(i)
                name = item.data(Qt.UserRole)
                if not name:
                    name = item.text().split(" ")[0]
                if name == target_stem:
                    found_row = i
                    break
            
            if found_row != -1:
                self.file_list.setCurrentRow(found_row)
            else:
                QMessageBox.warning(self, "错误", f"在列表中找不到文件: {target_stem}")
                return
                
        target_row = -1
        for i, c in enumerate(self.candidates):
            if c.get('x') == target_x and c.get('y') == target_y:
                target_row = i
                break
        
        if target_row != -1:
            self.cand_list.setCurrentRow(target_row)
            self.on_candidate_selected(target_row)
            if activate_main:
                self.activateWindow()
            try:
                if self._is_suspect_mode_active():
                    self.suspect_list.setFocus()
            except Exception:
                pass
        else:
            print(f"Warning: Candidate at ({target_x}, {target_y}) not found in {target_stem}")

    # ===== 数据集样本保存 =====
    
    def save_dataset_sample(self, is_positive, auto_jump=True, explicit_candidate=None):
        """保存数据集样本"""
        if not self.candidates and not explicit_candidate:
            return
        
        if not hasattr(self, 'img_a') or self.img_a is None:
            self.statusBar().showMessage("❌ 无法保存：图片数据未加载（可能文件已被删除）", 5000)
            return

        cand = None
        row = -1
        
        if explicit_candidate:
            target_x, target_y = explicit_candidate.get('x'), explicit_candidate.get('y')
            target_stem = explicit_candidate.get('stem')
            if target_stem and target_stem != self.current_group:
                print(f"Warning: saving candidate from {target_stem} but current group is {self.current_group}")
            
            for i, c in enumerate(self.candidates):
                if c.get('x') == target_x and c.get('y') == target_y:
                    cand = c
                    row = i
                    break
            if not cand:
                print(f"Error: Explicit candidate ({target_x},{target_y}) not found in current group {self.current_group}.")
                return
        else:
            row = self.cand_list.currentRow()
            if row < 0 or row >= len(self.candidates):
                return
            cand = self.candidates[row]

        cand['saved'] = True
        cand['verdict'] = 'real' if is_positive else 'bogus'
        
        today_str = datetime.datetime.now().strftime("%Y%m%d")
        base_dir = os.path.join("dataset", today_str)
        subdir = "positive" if is_positive else "negative"
        save_dir = os.path.join(base_dir, subdir)
        os.makedirs(save_dir, exist_ok=True)
        
        cid = cand.get('id', row + 1)
        sharp = cand.get('sharp', 0)
        is_manual = cand.get('manual', False)
        
        prefix = "REAL" if is_positive else "BOGUS"
        m_tag = "MANUAL_" if is_manual else ""
        fname = f"{today_str}_{prefix}_{m_tag}{self.current_group}_cand{cid}_S{sharp:.1f}.png"
        
        cx, cy = cand['x'], cand['y']
        try:
            p_a = self._crop_patch_common(self.img_a, cx, cy)
            p_b = self._crop_patch_common(self.img_b, cx, cy)
            p_c = self._crop_patch_common(self.img_c, cx, cy)
            combined = np.hstack([p_a, p_b, p_c])
            
            self.io_pool.submit(self._threaded_save_image, save_dir, fname, combined)
        except Exception as e:
            print(f"Prepare save failed: {e}")

        if row >= 0:
            item = self.cand_list.item(row)
            if item:
                text = item.text()
                if "[已" not in text:
                    suffix = " [已存真]" if is_positive else " [已存假]"
                    item.setText(text + suffix)
                item.setForeground(QColor(0, 100, 0) if is_positive else QColor(100, 0, 0))

        DatabaseManager.update_record(self.current_group, self.candidates, crop_rect=self.crop_rect)
        if self.current_group in self.batch_results:
            self.batch_results[self.current_group]['candidates'] = self.candidates
            
        if explicit_candidate:
            return

        if not auto_jump:
            return
        if is_manual:
            return

        if row < self.cand_list.count() - 1:
            self.cand_list.setCurrentRow(row + 1)
        else:
            QMessageBox.information(self, "完成", "本张图片所有候选体已处理完毕")

    def _threaded_save_image(self, save_dir, fname, img_data):
        """后台保存图片"""
        try:
            counter = 1
            final_path = os.path.join(save_dir, fname)
            base_name, ext = os.path.splitext(fname)
            while os.path.exists(final_path):
                final_path = os.path.join(save_dir, f"{base_name}_{counter}{ext}")
                counter += 1
            
            cv2.imwrite(final_path, img_data)
            print(f"Saved (Async): {final_path}")
        except Exception as e:
            print(f"Async Save Error: {e}")

    def skip_sample(self, auto_jump=True):
        """跳过当前样本"""
        if not self.candidates:
            return
        row = self.cand_list.currentRow()
        
        if not auto_jump:
            return

        cand = self.candidates[row]
        if cand.get('manual', False):
            pass
        elif row == len(self.candidates) - 1:
            DatabaseManager.mark_status(self.current_group, "processed")
            curr_item = self.file_list.currentItem()
            if curr_item:
                curr_item.setText(f"{self.current_group} [已归档]")
                curr_item.setForeground(QColor(0, 100, 255))
                curr_item.setFont(QFont("Arial", 10, QFont.Bold))
            self.jump_to_next_image()
        else:
            self.cand_list.setCurrentRow(row + 1)

    def jump_to_next_image(self):
        """跳转到下一张有目标的图片"""
        curr_idx = self.file_list.currentRow()
        for i in range(curr_idx + 1, self.file_list.count()):
            item = self.file_list.item(i)
            name = item.data(Qt.UserRole)
            if not name:
                name = item.text().split(" ")[0]
            rec = self.batch_results.get(name)
            if rec and rec.get("candidates") and rec.get("status") != "processed":
                self.file_list.setCurrentRow(i)
                return
        QMessageBox.information(self, "提示", "后续没有待处理的有目标图片了！")

    # ===== 点击与诊断 =====
    
    def on_context_click(self, x, y):
        """图片点击事件"""
        if self.current_preview_img is None:
            return
        
        self.perform_diagnosis(x, y)
        
        unsaved_manual_idx = -1
        for i, c in enumerate(self.candidates):
            if c.get('manual', False) and not c.get('saved', False):
                unsaved_manual_idx = i
                break
        
        if unsaved_manual_idx != -1:
            self.candidates[unsaved_manual_idx]['x'] = x
            self.candidates[unsaved_manual_idx]['y'] = y
            self.candidates[unsaved_manual_idx]['rise'] = 999
            self.candidates[unsaved_manual_idx]['ai_score'] = 0.0
            print(f"Updated unsaved manual target #{self.candidates[unsaved_manual_idx]['id']} to ({x}, {y})")
            manual_idx = unsaved_manual_idx
        else:
            next_id = len(self.candidates) + 1
            new_cand = {
                'id': next_id,
                'x': x, 'y': y,
                'area': 999, 'sharp': 9.9, 'peak': 255, 'contrast': 100,
                'rise': 999,
                'crop_off': (0, 0),
                'manual': True,
                'ai_score': 0.0,
                'saved': False
            }
            self.candidates.append(new_cand)
            manual_idx = len(self.candidates) - 1
            print(f"Created new manual target #{next_id} at ({x}, {y})")

        DatabaseManager.update_record(self.current_group, self.candidates, crop_rect=self.crop_rect)
        if self.current_group in self.batch_results:
            self.batch_results[self.current_group]['candidates'] = self.candidates
            
        self.refresh_cand_list()
        self.cand_list.setCurrentRow(manual_idx)

    def perform_diagnosis(self, x, y):
        """执行漏检诊断"""
        try:
            if not hasattr(self, 'img_a'):
                return
            
            def to_gray(img):
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            
            gray_a = to_gray(self.img_a)
            gray_b = to_gray(self.img_b)
            gray_c = to_gray(self.img_c)
            h, w = gray_a.shape
            
            r = 3
            x0, x1 = max(0, x - r), min(w, x + r + 1)
            y0, y1 = max(0, y - r), min(h, y + r + 1)
            
            roi_a = gray_a[y0:y1, x0:x1]
            roi_b = gray_b[y0:y1, x0:x1]
            roi_c = gray_c[y0:y1, x0:x1]
            
            if roi_a.size == 0:
                return
            
            peak = float(np.max(roi_a))
            mean = float(np.mean(roi_a))
            median = float(np.median(roi_a))
            sharpness = peak / (mean + 1e-6)
            contrast = peak - median
            
            val_b = float(np.max(roi_b)) if roi_b.size > 0 else 0
            val_c = float(np.max(roi_c)) if roi_c.size > 0 else 0
            rise = val_b - val_c
            
            reasons = []
            
            thresh = self.cfg['thresh']
            if self.cfg.get('dynamic_thresh', False):
                bg_a = np.median(gray_a)
                thresh += bg_a
            
            if peak < thresh:
                reasons.append(f"❌ 亮度不足 (Peak={peak:.1f} < {thresh:.1f})")
            
            min_sharp = self.cfg['sharpness']
            if self.cfg['kill_flat'] and sharpness < min_sharp:
                reasons.append(f"❌ 过于平坦 (Sharp={sharpness:.2f} < {min_sharp})")
                
            min_contrast = self.cfg['contrast']
            if self.cfg['kill_flat'] and contrast < min_contrast:
                reasons.append(f"❌ 对比度低 (Cont={contrast:.1f} < {min_contrast})")
            
            edge = self.cfg.get('edge_margin', 10)
            if x < edge or y < edge or x > w - edge or y > h - edge:
                reasons.append(f"❌ 位于边缘 (Edge < {edge})")
                
            if rise < 0:
                reasons.append(f"⚠️ 负增亮 (Rise={rise:.1f})")
                
            msg = []
            msg.append(f"📍 坐标: ({x}, {y})")
            msg.append("-" * 30)
            msg.append(f"📊 基础特征:")
            msg.append(f"   • Peak (亮度): {peak:.1f}")
            msg.append(f"   • Sharp (锐度): {sharpness:.2f}")
            msg.append(f"   • Contrast (对比): {contrast:.1f}")
            msg.append(f"   • Rise (增亮): {rise:.1f} (B={val_b:.1f}, C={val_c:.1f})")
            msg.append("-" * 30)
            
            if reasons:
                msg.append("🛑 潜在漏检原因:")
                for r in reasons:
                    msg.append(f"   {r}")
            else:
                msg.append("✅ 各项指标正常 (可能是面积/长宽比等其他原因被滤)")
                
            print("\n".join(msg))
            QMessageBox.information(self, "目标诊断报告", "\n".join(msg))
            
        except Exception as e:
            print(f"Diagnosis failed: {e}")

    # ===== 右键菜单 =====
    
    def show_list_context_menu(self, list_widget, pos):
        """显示列表右键菜单"""
        item = list_widget.itemAt(pos)
        if not item:
            return
        
        menu = QMenu()
        download_action = menu.addAction("📥 下载对应 FITS 原图")
        action = menu.exec_(list_widget.mapToGlobal(pos))
        
        if action == download_action:
            self.download_fits_for_item(list_widget, item)

    def show_file_list_context_menu(self, pos):
        """显示文件列表右键菜单"""
        item = self.file_list.itemAt(pos)
        if not item:
            return
        
        menu = QMenu()
        act_download = menu.addAction("📥 下载此组对应的 FITS 原图")
        menu.addSeparator()
        act_clear = menu.addAction("🔄 清除缓存并重算")
        act_delete = menu.addAction("🔥 彻底删除源文件 (abc)")
        
        action = menu.exec_(self.file_list.mapToGlobal(pos))
        
        name = item.data(Qt.UserRole)
        if not name:
            name = item.text().split(" ")[0]

        if action == act_download:
            self.download_fits_for_item(self.file_list, item)
        elif action == act_clear:
            DatabaseManager.delete_record(name)
            if name in self.batch_results:
                del self.batch_results[name]
            QMessageBox.information(self, "提示", f"已清除 {name} 的缓存，请点击批量计算重新提取。")
            item.setText(name)
            item.setForeground(QColor(0, 0, 0))
            item.setFont(QFont("Arial", 10))
        elif action == act_delete:
            reply = QMessageBox.question(self, "高能预警", f"确定要永久删除 {name} 相关的三张图片吗？\n此操作不可恢复！",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.delete_source_files(name, item)


def main():
    """应用程序入口"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # 设置默认字体
    font = QFont("Microsoft YaHei UI", 9)
    app.setFont(font)
    
    window = SCANN()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
