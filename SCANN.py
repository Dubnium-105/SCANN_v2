import sys
import os
import json
import sqlite3
import time
import cv2
import numpy as np
import torch
import traceback
import queue
import threading
import requests
import urllib.parse
import hashlib
import re
import collections
from concurrent.futures import ThreadPoolExecutor
from torchvision import models, transforms
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRectF, QPointF, QEvent, QTimer, QObject
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QPainter, QPen, QBrush, QWheelEvent, QMouseEvent, QKeySequence
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ================= 三联图联动下载引擎 =================
class LinkedDownloader(QObject):
    download_progress = pyqtSignal(str, int, int) # name, current, total
    download_finished = pyqtSignal(str, bool, str) # name, success, path_or_error
    all_finished = pyqtSignal(int, int) # success_count, fail_count
    status_msg = pyqtSignal(str)

    BASE_JPG_URL = "https://nadc.china-vo.org/psp/hmt/PSP-HMT-DATA/output/"
    BASE_FITS_URL = "https://nadc.china-vo.org/psp/hmt/PSP-HMT-DATA/data/"
    
    def __init__(self):
        super().__init__()
        # --- 动态获取路径：确保在任何运行环境下都能准确定位到脚本目录 ---
        if getattr(sys, 'frozen', False):
            self._SCRIPT_DIR = os.path.dirname(sys.executable)
        else:
            self._SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
            
        self.APPDATA_DIR = os.path.join(self._SCRIPT_DIR, "SCANN_Data")
        self.DB_FILE = os.path.join(self.APPDATA_DIR, "psp_linkage.db")
        print(f"📦 数据库位置: {self.DB_FILE}") 
        # ---------------------------------------------------------------
        
        if not os.path.exists(self.APPDATA_DIR):
            os.makedirs(self.APPDATA_DIR)
        self.session = self._create_session()
        self._init_db()
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.active_tasks = {} # jpg_stem -> future
        self.cancel_requested = set() # jpg_stem set for active cancellation
        self.stats_lock = threading.Lock()
        self.session_success = 0
        self.session_fail = 0

    def _create_session(self):
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def stop_all(self):
        """中止所有下载"""
        count = 0
        for stem, future in list(self.active_tasks.items()):
            if future.cancel():
                count += 1
                self.active_tasks.pop(stem, None)
            else:
                self.cancel_requested.add(stem)
        self.status_msg.emit(f"已请求中止所有任务 (取消了 {count} 个等待中的任务)")

    def _init_db(self):
        with sqlite3.connect(self.DB_FILE) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS linkage (
                    jpg_stem TEXT PRIMARY KEY COLLATE NOCASE,
                    local_jpg_path TEXT,
                    remote_fits_url TEXT,
                    status TEXT DEFAULT 'pending',
                    local_fits_path TEXT,
                    timestamp REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON linkage(status);")

    @staticmethod
    def clean_filename(name):
        return re.sub(r'[\\/:*?"<>|]', '_', name)

    @staticmethod
    def map_jpg_to_fits(jpg_url):
        if "/output/" not in jpg_url: return None
        # 1. 替换目录
        url = jpg_url.replace("/output/", "/data/")
        # 2. 移除 JPG 后缀
        if url.lower().endswith(".jpg"): url = url[:-4]
        elif url.lower().endswith(".jpeg"): url = url[:-5]
        
        # 3. 终极修复：将分身后缀 (.fts1-9a/b/c) 还原为标准 FITS 后缀 (.fts)
        # 比如：.../NGC4866.fts5b -> .../NGC4866.fts
        url = re.sub(r'\.fts[1-9][abc]?$', '.fts', url, flags=re.I)
        return url

    def _normalize_stem(self, stem):
        if not stem: return ""
        # 内部处理：去空格、转小写、切除分身后缀 (fts1-9, a/b/c)
        s = stem.replace(" ", "").lower().strip()
        return re.sub(r'\.fts[1-9][abc]?$', '', s)

    def get_linkage(self, jpg_stem):
        if not jpg_stem: return None
        search_id = self._normalize_stem(jpg_stem)
        
        print(f"🔍 [Query] 目标: '{jpg_stem}' -> ID: '{search_id}'")
        with sqlite3.connect(self.DB_FILE) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("SELECT * FROM linkage WHERE jpg_stem = ?", (search_id,))
            res = cur.fetchone()
            if res:
                print(f"✅ [Query] 找到 FITS 联动记录")
                return dict(res)
            else:
                print(f"❌ [Query] 数据库中没有记录: '{search_id}'")
            return None

    def update_linkage(self, jpg_stem, **kwargs):
        if not jpg_stem: return
        search_stem = self._normalize_stem(jpg_stem)
        cols = ", ".join([f"{k} = ?" for k in kwargs.keys()])
        vals = list(kwargs.values()) + [search_stem]
        with sqlite3.connect(self.DB_FILE) as conn:
            conn.execute(f"UPDATE linkage SET {cols} WHERE jpg_stem = ? COLLATE NOCASE", vals)

    def add_linkage(self, jpg_stem, local_jpg_path, remote_fits_url):
        if not jpg_stem: return
        save_stem = self._normalize_stem(jpg_stem)
        with sqlite3.connect(self.DB_FILE) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO linkage (jpg_stem, local_jpg_path, remote_fits_url, timestamp)
                VALUES (?, ?, ?, ?)
            """, (save_stem, local_jpg_path, remote_fits_url, time.time()))

    def batch_add_linkage(self, items):
        """批量添加联动信息，极速且精准覆盖"""
        if not items: return
        now = time.time()
        
        # 1. 内部清洗：去空格、转小写、切除分身后缀，精准识别 FITS 归属
        unique_items = {}
        for stem, path, url in items:
            clean_id = self._normalize_stem(stem)
            if not clean_id: continue
            unique_items[clean_id] = (path, url)
        
        print(f"📥 [Store] 正在同步 {len(unique_items)} 组 FITS 联动数据...")
        
        # 2. 强力写入数据库
        try:
            with sqlite3.connect(self.DB_FILE) as conn:
                conn.executemany("""
                    INSERT OR REPLACE INTO linkage (jpg_stem, local_jpg_path, remote_fits_url, timestamp)
                    VALUES (?, ?, ?, ?)
                """, [(stem, data[0], data[1], now) for stem, data in unique_items.items()])
            
            self.status_msg.emit(f"💾 数据库同步完成: 记录了 {len(unique_items)} 组联动信息")
            print(f"✅ [Store] 联动记录同步完成")
        except Exception as e:
            print(f"❌ [Store] 写入失败: {str(e)}")
            self.status_msg.emit(f"❌ 数据库写入失败: {str(e)}")

    def clear_all_linkage(self):
        """清空所有联动记录 (用于数据库大更新时)"""
        try:
            with sqlite3.connect(self.DB_FILE) as conn:
                conn.execute("DELETE FROM linkage;")
            self.status_msg.emit("🧹 联动数据库已清空")
            return True
        except Exception as e:
            self.status_msg.emit(f"❌ 清空失败: {e}")
            return False

    def download_task(self, url, save_path, jpg_stem):
        filename = os.path.basename(save_path)
        try:
            if jpg_stem in self.cancel_requested:
                self.cancel_requested.remove(jpg_stem)
                return

            head = self.session.head(url, timeout=15)
            if head.status_code == 404:
                self.status_msg.emit(f"❌ 404 错误: {filename}")
                self.download_finished.emit(jpg_stem, False, "404 Not Found")
                return

            total_size = int(head.headers.get('content-length', 0))
            initial_pos = 0
            mode = 'wb'
            if os.path.exists(save_path):
                initial_pos = os.path.getsize(save_path)
                if initial_pos < total_size: mode = 'ab'
                elif initial_pos == total_size:
                    # FIX: 即使文件已存在且大小一致，也必须同步更新 linkage 数据库状态
                    self.download_finished.emit(jpg_stem, True, save_path)
                    is_fits = False
                    for fext in ['.fts', '.fts2', '.fits']:
                        if url.lower().endswith(fext) or f"{fext}?" in url.lower():
                            is_fits = True; break
                    if is_fits:
                        self.update_linkage(jpg_stem, status='downloaded', local_fits_path=save_path)
                    else:
                        self.update_linkage(jpg_stem, local_jpg_path=save_path)
                    return
                else: initial_pos = 0

            headers = {'Range': f'bytes={initial_pos}-'} if initial_pos > 0 else {}
            resp = self.session.get(url, headers=headers, stream=True, timeout=30)
            resp.raise_for_status()

            with open(save_path, mode) as f:
                current_pos = initial_pos
                for chunk in resp.iter_content(chunk_size=128*1024):
                    if jpg_stem in self.cancel_requested:
                        self.cancel_requested.remove(jpg_stem)
                        self.status_msg.emit(f"🛑 已中止下载: {filename}")
                        return
                    if chunk:
                        f.write(chunk)
                        current_pos += len(chunk)
                        self.download_progress.emit(jpg_stem, current_pos, total_size)

            # 下载完成后更新数据库
            is_fits = False
            for fext in ['.fts', '.fts2', '.fits']:
                if url.lower().endswith(fext) or f"{fext}?" in url.lower():
                    is_fits = True; break
            if is_fits:
                self.update_linkage(jpg_stem, status='downloaded', local_fits_path=save_path)
            else:
                self.update_linkage(jpg_stem, local_jpg_path=save_path)
                
            self.download_finished.emit(jpg_stem, True, save_path)
            self.status_msg.emit(f"✅ 下载完成: {filename}")
            with self.stats_lock: self.session_success += 1
        except Exception as e:
            self.download_finished.emit(jpg_stem, False, str(e))
            self.status_msg.emit(f"❌ 下载失败: {filename} ({str(e)})")
            with self.stats_lock: self.session_fail += 1
        finally:
            self.active_tasks.pop(jpg_stem, None)
            # 如果所有任务都结束了，发送总信号
            if not self.active_tasks:
                self.all_finished.emit(self.session_success, self.session_fail)
                # 重置计数器以便下次使用
                with self.stats_lock:
                    self.session_success = 0
                    self.session_fail = 0

    def submit_download(self, jpg_stem, remote_url, local_save_dir, override_filename=None):
        if jpg_stem in self.active_tasks: return
        
        # FIX: 更严格的扩展名判断
        url_lower = remote_url.lower()
        is_fits = False
        for fext in ['.fts', '.fts2', '.fits']:
            if url_lower.endswith(fext) or f"{fext}?" in url_lower:
                is_fits = True
                break
        
        if override_filename:
            filename = self.clean_filename(override_filename)
        else:
            raw_filename = os.path.basename(remote_url)
            unquoted_filename = urllib.parse.unquote(raw_filename)
            filename = self.clean_filename(unquoted_filename)
        
        # 只有确实是 FITS 且文件名还没以 .fts 结尾时才追加
        if is_fits and not filename.lower().endswith(('.fts', '.fts2', '.fits')):
            filename += ".fts"
            
        save_path = os.path.join(local_save_dir, filename)
        future = self.executor.submit(self.download_task, remote_url, save_path, jpg_stem)
        self.active_tasks[jpg_stem] = future

class DBDownloadWindow(QDialog):
    sig_load_done = pyqtSignal(object)
    sig_scan_done = pyqtSignal(object)
    sig_scan_status = pyqtSignal(int, int) # found, skipped

    def __init__(self, downloader, parent=None):
        super().__init__(parent)
        self.downloader = downloader
        self.setWindowTitle("🌐 PSP 数据库级联下载")
        self.resize(1000, 700)
        self.current_url = downloader.BASE_JPG_URL
        self.history_stack = []
        self.stop_scan_flag = False
        
        self.sig_load_done.connect(self._on_load_done)
        self.sig_scan_done.connect(self._on_scan_done)
        self.sig_scan_status.connect(self._on_scan_status)
        
        self.init_ui()
        self.load_directory(self.current_url)

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # 顶部导航
        nav_layout = QHBoxLayout()
        self.btn_back = QPushButton("⬅️ 返回上级")
        self.btn_back.clicked.connect(self.go_back)
        self.lbl_path = QLabel(self.current_url)
        self.lbl_path.setStyleSheet("color: #666; font-family: Consolas;")
        nav_layout.addWidget(self.btn_back)
        nav_layout.addWidget(self.lbl_path, 1)
        layout.addLayout(nav_layout)

        # 列表显示
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_widget.itemDoubleClicked.connect(self.on_item_double_clicked)
        self.list_widget.setFont(QFont("Consolas", 10))
        layout.addWidget(self.list_widget)

        # 底部操作
        bottom_layout = QHBoxLayout()
        self.btn_select_all = QPushButton("✅ 全选")
        self.btn_select_all.clicked.connect(lambda: self.list_widget.selectAll())
        
        self.btn_clear_db = QPushButton("🧹 清空联动库")
        self.btn_clear_db.setStyleSheet("color: #757575;")
        self.btn_clear_db.setToolTip("当服务器更新导致下载链接失效时使用")
        self.btn_clear_db.clicked.connect(self.confirm_clear_linkage)

        self.btn_download = QPushButton("📥 批量下载所选 JPG")
        self.btn_download.setFixedHeight(40)
        self.btn_download.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_download.clicked.connect(self.start_batch_download)

        self.btn_stop = QPushButton("🛑 中止")
        self.btn_stop.setFixedHeight(40)
        self.btn_stop.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        self.btn_stop.clicked.connect(self.stop_all_actions)
        
        bottom_layout.addWidget(self.btn_select_all)
        bottom_layout.addWidget(self.btn_clear_db)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.btn_stop)
        bottom_layout.addWidget(self.btn_download)
        layout.addLayout(bottom_layout)

        # 进度显示
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

    def confirm_clear_linkage(self):
        reply = QMessageBox.question(self, "确认清空", "确定要清空本地 FITS 联动库吗？\n\n这不会删除已下载的文件，但会导致右键无法直接下载 FITS，直到您再次扫描服务器。",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            if self.downloader.clear_all_linkage():
                QMessageBox.information(self, "成功", "联动库已清空。")

    def load_directory(self, url):
        self.lbl_path.setText(url)
        self.list_widget.clear()
        self.list_widget.addItem("⏳ 正在加载目录...")
        
        def fetch():
            try:
                resp = self.downloader.session.get(url, timeout=15)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, 'html.parser')
                items = []
                for a in soup.find_all('a'):
                    href = a.get('href')
                    text = a.text.strip()
                    if href in ("../", "/"): continue
                    full_url = urllib.parse.urljoin(url, href)
                    is_dir = href.endswith('/')
                    items.append((text, full_url, is_dir))
                self.sig_load_done.emit(items)
            except Exception as e:
                self.sig_load_done.emit(str(e))

        threading.Thread(target=fetch, daemon=True).start()

    def _on_load_done(self, result):
        self.list_widget.clear()
        if isinstance(result, str):
            QMessageBox.warning(self, "加载失败", result)
            return
        for text, f_url, is_dir in result:
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, {"url": f_url, "is_dir": is_dir})
            if is_dir:
                item.setIcon(self.style().standardIcon(QStyle.SP_DirIcon))
                item.setForeground(QColor("#FF9800"))
            else:
                item.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
            self.list_widget.addItem(item)

    def on_item_double_clicked(self, item):
        data = item.data(Qt.UserRole)
        if data and data['is_dir']:
            self.history_stack.append(self.current_url)
            self.current_url = data['url']
            self.load_directory(self.current_url)

    def stop_all_actions(self):
        """中止下载和正在进行的扫描"""
        self.stop_scan_flag = True
        self.downloader.stop_all()
        self.btn_download.setEnabled(True)
        self.progress_bar.setVisible(False)
        if self.parent() and hasattr(self.parent(), 'statusBar'):
            self.parent().statusBar().showMessage("⏹ 已尝试中止所有操作")

    def go_back(self):
        if self.history_stack:
            self.current_url = self.history_stack.pop()
            self.load_directory(self.current_url)

    def _on_scan_status(self, found, skipped):
        if self.parent() and hasattr(self.parent(), 'statusBar'):
            self.parent().statusBar().showMessage(f"🔍 正在扫描... 已发现: {found} | 匹配本地: {skipped}")

    def start_batch_download(self):
        selected_items = self.list_widget.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "提示", "请先选择要下载的项目")
            return

        # 1. 确定 JPG 保存目录
        save_dir = self.parent().cfg.get('jpg_download_dir')
        if not save_dir or not os.path.exists(save_dir):
            save_dir = QFileDialog.getExistingDirectory(self, "选择 JPG 保存目录")
            if not save_dir: return
            self.parent().cfg['jpg_download_dir'] = save_dir
            ConfigManager.save(self.parent().cfg)
            self.parent().lbl_jpg_path.setText(f"JPG: {save_dir}")
            self.parent().lbl_jpg_path.setToolTip(save_dir)

        # 2. 并行扫描并收集任务
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0) 
        self.btn_download.setEnabled(False)
        self.stop_scan_flag = False
        
        def scan_and_submit():
            try:
                # 在子线程中一次性构建 existing_files 集合，避免扫描时频繁 I/O
                existing_files = set()
                try:
                    for fn in os.listdir(save_dir):
                        existing_files.add(fn.lower())
                except Exception: pass

                all_tasks = []
                scan_jobs = []
                for item in selected_items:
                    data = item.data(Qt.UserRole)
                    name = item.text().rstrip('/')
                    if data['is_dir']:
                        scan_jobs.append((data['url'], name, 1))
                    else:
                        jpg_url = data['url']
                        fits_url = self.downloader.map_jpg_to_fits(jpg_url)
                        filename = urllib.parse.unquote(os.path.basename(jpg_url))
                        
                        # 提取 stem (保持原始大小写)
                        jpg_stem = filename
                        for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
                            if jpg_stem.endswith(ext):
                                jpg_stem = jpg_stem[:-len(ext)]
                                break
                                
                        all_tasks.append({"stem": jpg_stem, "url": jpg_url, "fits": fits_url, "filename": filename})
                
                if scan_jobs:
                    # 并行递归扫描，支持前缀和深度限制
                    results, skipped_links = self._parallel_scan_v63_style(scan_jobs, existing_files, max_depth=4)
                    all_tasks.extend(results)
                else:
                    skipped_links = []
                
                if self.stop_scan_flag:
                    self.sig_scan_done.emit("扫描已中止")
                else:
                    self.sig_scan_done.emit((all_tasks, skipped_links, save_dir))
            except Exception as e:
                traceback.print_exc()
                self.sig_scan_done.emit(str(e))

        threading.Thread(target=scan_and_submit, daemon=True).start()

    def _parallel_scan_v63_style(self, scan_jobs, existing_files, max_depth=4):
        """参考 v63 扁平化命名逻辑的并行扫描引擎 (Regex 版)"""
        all_results = []
        skipped_linkages = [] 
        visited_urls = set() # 防环路大脑
        lock = threading.Lock()
        queue = collections.deque(scan_jobs)
        active_count = 0
        found_count = 0
        skipped_count = 0
        last_update_time = time.time()
        cv = threading.Condition(lock)

        def worker():
            nonlocal active_count, found_count, skipped_count, last_update_time
            while True:
                with lock:
                    while not queue and active_count > 0 and not self.stop_scan_flag:
                        cv.wait(timeout=1)
                    if self.stop_scan_flag or (not queue and active_count == 0):
                        return
                    
                    url, prefix, depth = queue.popleft()
                    if url in visited_urls: 
                        continue
                    visited_urls.add(url)
                    active_count += 1

                try:
                    resp = self.downloader.session.get(url, timeout=10)
                    resp.raise_for_status()
                    
                    # 极其严谨的正则：排除所有以 . 或 / 开头的链接，防止爬回上级或根目录
                    matches = re.findall(r'href="([^.?/][^"?/]*(?:/|\.jpg))"', resp.text, re.I)
                    
                    new_jobs = []
                    local_files = []
                    local_skipped = []
                    
                    for href in matches:
                        # 二次保险：绝对不爬上级目录或特殊链接
                        if href.startswith('.') or href.startswith('/') or '?' in href:
                            continue
                            
                        full_url = urllib.parse.urljoin(url, href)
                        
                        if href.endswith('/'):
                            if depth < max_depth:
                                clean_name = urllib.parse.unquote(href.rstrip('/'))
                                new_prefix = f"{prefix}_{clean_name}"
                                new_jobs.append((full_url, new_prefix, depth + 1))
                        elif href.lower().endswith('.jpg'):
                            # 彻底移除前缀，恢复原始文件名
                            save_name = urllib.parse.unquote(href)
                            clean_save_name = self.downloader.clean_filename(save_name)
                            
                            # 提取 stem (保持原始大小写，确保与标注软件匹配)
                            stem = clean_save_name
                            for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
                                if stem.endswith(ext):
                                    stem = stem[:-len(ext)]
                                    break
                            
                            fits_url = self.downloader.map_jpg_to_fits(full_url)

                            with lock:
                                found_count += 1
                                if clean_save_name.lower() in existing_files:
                                    skipped_count += 1
                                    local_skipped.append((stem, clean_save_name, fits_url))
                                    continue
                            
                            local_files.append({
                                "stem": stem, 
                                "url": full_url, 
                                "fits": fits_url, 
                                "filename": save_name
                            })

                    with lock:
                        all_results.extend(local_files)
                        skipped_linkages.extend(local_skipped)
                        queue.extend(new_jobs)
                        now = time.time()
                        if now - last_update_time >= 1.0:
                            self.sig_scan_status.emit(found_count, skipped_count)
                            last_update_time = now
                        cv.notify_all()
                except Exception:
                    pass
                finally:
                    with lock:
                        active_count -= 1
                        cv.notify_all()

        threads = []
        for _ in range(10):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)
        
        for t in threads:
            t.join()
            
        return all_results, skipped_linkages

    def _on_scan_done(self, result):
        self.progress_bar.setVisible(False)
        self.btn_download.setEnabled(True)
        
        if isinstance(result, str):
            if result != "扫描已中止":
                QMessageBox.warning(self, "扫描失败", result)
            else:
                if self.parent() and hasattr(self.parent(), 'statusBar'):
                    self.parent().statusBar().showMessage("⏹ 扫描已由用户中止")
            return
            
        tasks, skipped_links, save_dir = result
        if not tasks and not skipped_links:
            QMessageBox.warning(self, "提示", "未找到可下载的预览图")
            return

        # --- 极速去重结果处理 ---
        if self.parent() and hasattr(self.parent(), 'statusBar'):
            self.parent().statusBar().showMessage("🔍 正在整理扫描结果...")
        QApplication.processEvents()

        # 1. 补全暂存的联动信息 (已存在的文件) - 使用批量提交，速度提升万倍
        linkage_items = []
        for stem, filename, fits_url in skipped_links:
            local_path = os.path.join(save_dir, filename)
            linkage_items.append((stem, local_path, fits_url))
        self.downloader.batch_add_linkage(linkage_items)

        if not tasks:
            QMessageBox.information(self, "提示", f"扫描到 {len(skipped_links)} 个文件，全部已存在。")
            return

        # 2. 确认提示
        msg = f"扫描完成！共发现 {len(tasks) + len(skipped_links)} 个文件。\n\n准备下载: {len(tasks)} 个\n已存在跳过: {len(skipped_links)} 个"
        if len(tasks) > 100:
            msg = "⚠️ ⚠️ ⚠️ 任务量较大！\n\n" + msg
        
        ok = QMessageBox.question(self, "确认下载任务", msg, QMessageBox.Yes | QMessageBox.No)
        if ok == QMessageBox.No: return

        # 3. 提交下载任务
        for task in tasks:
            self.downloader.add_linkage(task['stem'], "", task['fits'])
            self.downloader.submit_download(task['stem'], task['url'], save_dir, override_filename=task['filename'])
        
        if self.parent() and hasattr(self.parent(), 'statusBar'):
            self.parent().statusBar().showMessage(f"🚀 已添加 {len(tasks)} 个任务到下载队列")

    def _recursive_scan_jpg(self, url):
        """(过时) 请使用 _parallel_scan_jpgs"""
        return self._parallel_scan_jpgs([url])

# ================= 核心配置区 =================
class ProcessingConfig:
    # === 新增/调整配置 ===
    TOPK_CHEAP = 20        # 按 cheap_score
    TOPK_RISE  = 20        # 按 rise
    TOPK_CONTRAST = 20     # 按 contrast
    TOPK_UNION = True      # 是否启用并集保底
    
    INFER_CHUNK = 512      # 推理分块
    CROP_SZ = 80
    RESIZE_HW = (224, 224) # 训练输入
    
    # --- Cheap Score 配置 ---
    # 模式: 'robust_z' (推荐) 或 'rise_only' (仅调试用)
    CHEAP_MODE = 'robust_z' 
    
    # robust_z 模式下的权重
    W_RISE = 2.0
    W_CONTRAST = 1.0
    W_SHARP = 0.5
    W_AREA_PENALTY = 0.3   # * abs(z_area)

    # 并行配置
    NUM_WORKERS = 4        # 预处理线程数

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRectF, QPointF, QEvent, QTimer, QObject
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QPainter, QPen, QBrush, QWheelEvent, QMouseEvent, QKeySequence

# ================= 数据库管理 (持久化) =================
DB_JSON_FILE = os.path.join(os.getcwd(), "SCANN_candidates.json")
DB_SQLITE_FILE = os.path.join(os.getcwd(), "SCANN_candidates.sqlite")

class AsyncDatabaseWriter(QThread):
    """专用数据库写入线程，避免 I/O 阻塞主流程"""
    def __init__(self):
        super().__init__()
        self.queue = queue.Queue()
        self._is_running = True
        self.start()

    def run(self):
        while self._is_running:
            try:
                # 阻塞等待任务
                task = self.queue.get(timeout=1)
                if task is None: break # 退出信号
                
                func, args = task
                try:
                    func(*args)
                except Exception as e:
                    print(f"❌ DB Write Error: {e}")
                    traceback.print_exc()
                finally:
                    self.queue.task_done()
            except queue.Empty:
                continue

    def stop(self):
        self._is_running = False
        self.queue.put(None)
        self.wait() # 必须等待线程彻底结束，确保最后的数据已写入

    def pending_count(self):
        return self.queue.qsize()

    def submit(self, func, *args):
        self.queue.put((func, args))

# 全局 DB Writer 实例
_db_writer = None

class DatabaseManager:
    _cache = {}
    _local = threading.local()
    _db_ready = False
    _writer_commit_every = 50
    _writer_commit_count = 0
    _writer_last_commit = 0.0

    @staticmethod
    def init_async():
        global _db_writer
        DatabaseManager._ensure_db_ready()
        if _db_writer is None:
            _db_writer = AsyncDatabaseWriter()

    @staticmethod
    def get_pending_count():
        global _db_writer
        if _db_writer:
            return _db_writer.pending_count()
        return 0

    @staticmethod
    def stop_async():
        global _db_writer
        if _db_writer:
            print("Stopping DB Writer...")
            _db_writer.stop()
            _db_writer = None
        try:
            conn = getattr(DatabaseManager._local, "conn", None)
            if conn is not None:
                conn.commit()
                conn.close()
                DatabaseManager._local.conn = None
        except Exception:
            pass

    @staticmethod
    def _get_conn():
        conn = getattr(DatabaseManager._local, "conn", None)
        if conn is not None:
            return conn

        conn = sqlite3.connect(DB_SQLITE_FILE, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA foreign_keys=ON;")
        DatabaseManager._ensure_schema(conn)
        DatabaseManager._local.conn = conn
        return conn

    @staticmethod
    def _ensure_schema(conn):
        conn.execute(
            "CREATE TABLE IF NOT EXISTS images ("
            "stem TEXT PRIMARY KEY,"
            "status TEXT,"
            "candidates_json TEXT,"
            "candidates_count INTEGER,"
            "has_ai INTEGER,"
            "max_ai REAL,"
            "crop_rect TEXT,"
            "params_hash TEXT,"
            "timestamp REAL"
            ");"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_images_ts ON images(timestamp);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_images_max_ai ON images(max_ai);")

    @staticmethod
    def _ensure_db_ready():
        if DatabaseManager._db_ready:
            return

        conn = DatabaseManager._get_conn()
        conn.execute("SELECT 1;")
        conn.commit()

        if os.path.exists(DB_JSON_FILE):
            try:
                cur = conn.execute("SELECT COUNT(1) AS n FROM images;")
                n = int(cur.fetchone()["n"])
            except Exception:
                n = 0

            if n == 0:
                DatabaseManager._migrate_from_json(conn)

        DatabaseManager._db_ready = True

    @staticmethod
    def _migrate_from_json(conn):
        try:
            print(f"SQLite: importing legacy JSON: {DB_JSON_FILE}")
            with open(DB_JSON_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            items = []
            now = time.time()
            for stem, rec in data.items():
                status = rec.get("status", "unseen")
                cands = rec.get("candidates", []) or []
                crop_rect = rec.get("crop_rect", None)
                params_hash = rec.get("params_hash", "")
                timestamp = float(rec.get("timestamp", now) or now)
                cand_json = json.dumps(cands, ensure_ascii=False)
                count = len(cands)
                max_ai = 0.0
                has_ai = 0
                for c in cands:
                    if "ai_score" in c:
                        has_ai = 1
                        try:
                            s = float(c.get("ai_score", 0.0))
                            if s > max_ai:
                                max_ai = s
                        except Exception:
                            pass

                items.append(
                    (stem, status, cand_json, count, has_ai, max_ai, json.dumps(crop_rect, ensure_ascii=False) if crop_rect is not None else None, params_hash, timestamp)
                )

            conn.execute("BEGIN;")
            conn.executemany(
                "INSERT OR REPLACE INTO images(stem,status,candidates_json,candidates_count,has_ai,max_ai,crop_rect,params_hash,timestamp) "
                "VALUES(?,?,?,?,?,?,?,?,?);",
                items
            )
            conn.commit()
            print(f"SQLite: import done, stems={len(items)}")
        except Exception as e:
            print(f"SQLite: import failed: {e}")
            traceback.print_exc()
            try:
                conn.rollback()
            except Exception:
                pass

    @staticmethod
    def _update_record_impl(name, candidates, crop_rect, status, params_hash, timestamp):
        conn = DatabaseManager._get_conn()
        cands = candidates or []
        cand_json = json.dumps(cands, ensure_ascii=False)
        count = len(cands)

        max_ai = 0.0
        has_ai = 0
        for c in cands:
            if "ai_score" in c:
                has_ai = 1
                try:
                    s = float(c.get("ai_score", 0.0))
                    if s > max_ai:
                        max_ai = s
                except Exception:
                    pass

        crop_rect_json = json.dumps(crop_rect, ensure_ascii=False) if crop_rect is not None else None

        conn.execute(
            "INSERT INTO images(stem,status,candidates_json,candidates_count,has_ai,max_ai,crop_rect,params_hash,timestamp) "
            "VALUES(?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(stem) DO UPDATE SET "
            "status=excluded.status,"
            "candidates_json=excluded.candidates_json,"
            "candidates_count=excluded.candidates_count,"
            "has_ai=excluded.has_ai,"
            "max_ai=excluded.max_ai,"
            "crop_rect=COALESCE(excluded.crop_rect, images.crop_rect),"
            "params_hash=excluded.params_hash,"
            "timestamp=excluded.timestamp;",
            (name, status, cand_json, count, has_ai, max_ai, crop_rect_json, params_hash, timestamp)
        )

        DatabaseManager._cache[name] = {
            "status": status,
            "candidates": cands,
            "timestamp": timestamp,
            "params_hash": params_hash,
            "crop_rect": crop_rect if crop_rect is not None else DatabaseManager._cache.get(name, {}).get("crop_rect", None),
            "candidates_count": count,
            "has_ai": has_ai,
            "max_ai": max_ai
        }

        DatabaseManager._writer_commit_count += 1
        now = time.time()
        if DatabaseManager._writer_commit_count >= DatabaseManager._writer_commit_every or (now - DatabaseManager._writer_last_commit) > 1.0:
            conn.commit()
            DatabaseManager._writer_commit_count = 0
            DatabaseManager._writer_last_commit = now

    @staticmethod
    def update_record(name, candidates, crop_rect=None, status="unseen", params_hash=""):
        global _db_writer
        DatabaseManager._ensure_db_ready()
        timestamp = time.time()
        if _db_writer and _db_writer.isRunning():
            _db_writer.submit(DatabaseManager._update_record_impl, name, candidates, crop_rect, status, params_hash, timestamp)
        else:
            DatabaseManager._update_record_impl(name, candidates, crop_rect, status, params_hash, timestamp)
            try:
                DatabaseManager._get_conn().commit()
            except Exception:
                pass

    @staticmethod
    def _mark_status_impl(name, status, timestamp):
        conn = DatabaseManager._get_conn()
        conn.execute("UPDATE images SET status=?, timestamp=? WHERE stem=?;", (status, timestamp, name))
        if name in DatabaseManager._cache:
            DatabaseManager._cache[name]["status"] = status
            DatabaseManager._cache[name]["timestamp"] = timestamp

        DatabaseManager._writer_commit_count += 1
        now = time.time()
        if DatabaseManager._writer_commit_count >= DatabaseManager._writer_commit_every or (now - DatabaseManager._writer_last_commit) > 1.0:
            conn.commit()
            DatabaseManager._writer_commit_count = 0
            DatabaseManager._writer_last_commit = now

    @staticmethod
    def mark_status(name, status):
        global _db_writer
        DatabaseManager._ensure_db_ready()
        timestamp = time.time()
        if _db_writer and _db_writer.isRunning():
            _db_writer.submit(DatabaseManager._mark_status_impl, name, status, timestamp)
        else:
            DatabaseManager._mark_status_impl(name, status, timestamp)
            try:
                DatabaseManager._get_conn().commit()
            except Exception:
                pass

    @staticmethod
    def load_summaries_map():
        DatabaseManager._ensure_db_ready()
        conn = DatabaseManager._get_conn()
        out = {}
        cur = conn.execute("SELECT stem,status,candidates_count,has_ai,max_ai,crop_rect,params_hash,timestamp FROM images;")
        for r in cur.fetchall():
            crop_rect = None
            try:
                if r["crop_rect"] is not None:
                    crop_rect = json.loads(r["crop_rect"])
            except Exception:
                crop_rect = None
            out[r["stem"]] = {
                "status": r["status"],
                "candidates_count": int(r["candidates_count"] or 0),
                "has_ai": int(r["has_ai"] or 0),
                "max_ai": float(r["max_ai"] or 0.0),
                "crop_rect": crop_rect,
                "params_hash": r["params_hash"] or "",
                "timestamp": float(r["timestamp"] or 0.0)
            }
        return out

    @staticmethod
    def get_record(name):
        DatabaseManager._ensure_db_ready()
        if name in DatabaseManager._cache and "candidates" in DatabaseManager._cache[name]:
            return DatabaseManager._cache[name]

        conn = DatabaseManager._get_conn()
        cur = conn.execute("SELECT stem,status,candidates_json,candidates_count,has_ai,max_ai,crop_rect,params_hash,timestamp FROM images WHERE stem=?;", (name,))
        r = cur.fetchone()
        if not r:
            return None

        try:
            cands = json.loads(r["candidates_json"]) if r["candidates_json"] else []
        except Exception:
            cands = []

        crop_rect = None
        try:
            if r["crop_rect"] is not None:
                crop_rect = json.loads(r["crop_rect"])
        except Exception:
            crop_rect = None

        rec = {
            "status": r["status"],
            "candidates": cands,
            "timestamp": float(r["timestamp"] or 0.0),
            "params_hash": r["params_hash"] or "",
            "crop_rect": crop_rect,
            "candidates_count": int(r["candidates_count"] or 0),
            "has_ai": int(r["has_ai"] or 0),
            "max_ai": float(r["max_ai"] or 0.0)
        }
        DatabaseManager._cache[name] = rec
        return rec

    @staticmethod
    def _delete_record_impl(name):
        conn = DatabaseManager._get_conn()
        conn.execute("DELETE FROM images WHERE stem=?;", (name,))
        if name in DatabaseManager._cache:
            del DatabaseManager._cache[name]
        DatabaseManager._writer_commit_count += 1
        now = time.time()
        if DatabaseManager._writer_commit_count >= DatabaseManager._writer_commit_every or (now - DatabaseManager._writer_last_commit) > 1.0:
            conn.commit()
            DatabaseManager._writer_commit_count = 0
            DatabaseManager._writer_last_commit = now

    @staticmethod
    def delete_record(name):
        global _db_writer
        DatabaseManager._ensure_db_ready()
        if _db_writer and _db_writer.isRunning():
            _db_writer.submit(DatabaseManager._delete_record_impl, name)
        else:
            DatabaseManager._delete_record_impl(name)
            try:
                DatabaseManager._get_conn().commit()
            except Exception:
                pass

    @staticmethod
    def _clear_all_impl():
        conn = DatabaseManager._get_conn()
        conn.execute("DELETE FROM images;")
        conn.commit()
        DatabaseManager._cache = {}

    @staticmethod
    def clear_all():
        global _db_writer
        DatabaseManager._ensure_db_ready()
        if _db_writer and _db_writer.isRunning():
            _db_writer.submit(DatabaseManager._clear_all_impl)
        else:
            DatabaseManager._clear_all_impl()

# ================= 辅助函数：Patch 裁剪 (独立函数以支持并行) =================
def _prepare_patch_tensor_80_static(gray_a, gray_b, gray_c, cx, cy, crop_sz=80):
    """
    CPU Side: Crop -> Stack(A,B,C) -> Tensor
    Returns: [3, 80, 80] Float Tensor (0~1) on CPU
    """
    half = crop_sz // 2
    h, w = gray_a.shape[:2]
    
    x1, y1 = cx - half, cy - half
    x2, y2 = x1 + crop_sz, y1 + crop_sz
    
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(w, x2), min(h, y2)
    
    def get_crop(img):
        if sx1 >= sx2 or sy1 >= sy2: 
            return np.zeros((crop_sz, crop_sz), dtype=np.uint8)
        crop = img[sy1:sy2, sx1:sx2]
        if (sx2 - sx1) != crop_sz or (sy2 - sy1) != crop_sz:
            padded = np.zeros((crop_sz, crop_sz), dtype=np.uint8)
            dx1 = sx1 - x1; dy1 = sy1 - y1
            dx2 = dx1 + (sx2 - sx1); dy2 = dy1 + (sy2 - sy1)
            padded[dy1:dy2, dx1:dx2] = crop
            return padded
        return crop

    pa = get_crop(gray_a)
    pb = get_crop(gray_b)
    pc = get_crop(gray_c)
    
    # Merge 3 channels
    merged = np.stack([pa, pb, pc], axis=2) # (80, 80, 3)
    
    # HWC -> CHW, Float, Scale
    tensor = torch.from_numpy(merged.transpose(2, 0, 1)).float()
    tensor /= 255.0
    
    return tensor

# ================= 辅助函数：Stage A 处理 (独立函数以支持并行) =================
def process_stage_a(name, paths, params, config_dict):
    """
    Stage A Worker Function:
    1. Read Images
    2. Auto Crop
    3. Generate Candidates (Heuristics)
    4. Compute Cheap Score
    5. Top-K Filter
    6. Prepare Patch Tensors (CPU)
    """
    try:
        t0 = time.time()
        
        # 1. Read Images
        if not all(k in paths for k in ['a','b','c']): return None
        img_a = cv2.imread(paths['a']) 
        img_b = cv2.imread(paths['b'])
        img_c = cv2.imread(paths['c'])
        if img_a is None or img_b is None or img_c is None: return None

        # 2. Auto Crop
        x_off, y_off, w, h = 0, 0, img_a.shape[1], img_a.shape[0]
        if params['auto_crop']:
            gray_full = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
            _, thr_w = cv2.threshold(gray_full, 240, 255, cv2.THRESH_BINARY_INV)
            ctrs, _ = cv2.findContours(thr_w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if ctrs:
                c_max = max(ctrs, key=cv2.contourArea)
                bx, by, bw, bh = cv2.boundingRect(c_max)
                pad = 2
                x_off = max(0, bx+pad); y_off = max(0, by+pad)
                w = max(1, bw-2*pad); h = max(1, bh-2*pad)
        crop_rect = (x_off, y_off, w, h)
        
        gray_a = cv2.cvtColor(img_a[y_off:y_off+h, x_off:x_off+w], cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(img_b[y_off:y_off+h, x_off:x_off+w], cv2.COLOR_BGR2GRAY)
        gray_c = cv2.cvtColor(img_c[y_off:y_off+h, x_off:x_off+w], cv2.COLOR_BGR2GRAY)
        
        # 3. Generate Candidates
        candidates = []
        blurred = cv2.GaussianBlur(gray_a, (3, 3), 0)
        
        actual_thresh = params['thresh']
        if params.get('dynamic_thresh', False):
            actual_thresh = np.median(gray_a) + params['thresh']
            
        _, bin_img = cv2.threshold(blurred, actual_thresh, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h_img, w_img = gray_a.shape
        p_min_area = params['min_area']
        p_edge = params.get('edge_margin', 10)
        p_sharp = params['sharpness']
        p_max_sharp = params.get('max_sharpness', 5.0)
        p_contrast = params['contrast']
        do_flat = params['kill_flat']
        do_dipole = params['kill_dipole']
        
        for c in contours:
            area = cv2.contourArea(c)
            if area < p_min_area or area > 600: continue
            
            bx, by, bw, bh = cv2.boundingRect(c)
            if (bx < p_edge) or (by < p_edge) or (bx+bw > w_img-p_edge) or (by+bh > h_img-p_edge):
                continue
                
            M = cv2.moments(c)
            if M["m00"] == 0: continue
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            
            # Transient Check
            check_r = 3
            y0_r, y1_r = max(0, cy-check_r), min(h_img, cy+check_r+1)
            x0_r, x1_r = max(0, cx-check_r), min(w_img, cx+check_r+1)
            roi_b = gray_b[y0_r:y1_r, x0_r:x1_r]
            roi_c = gray_c[y0_r:y1_r, x0_r:x1_r]
            if roi_b.size == 0 or roi_c.size == 0: continue
            
            val_b = float(np.max(roi_b))
            val_c = float(np.max(roi_c))
            rise = val_b - val_c
            
            roi_spot = gray_a[by:by+bh, bx:bx+bw]
            if roi_spot.size == 0: continue
            peak = float(np.max(roi_spot))
            mean = float(np.mean(roi_spot))
            median_spot = float(np.median(roi_spot))
            sharpness = peak / (mean + 1e-6)
            contrast = peak - median_spot
            
            if do_flat:
                if sharpness < p_sharp: continue
                if sharpness > p_max_sharp: continue
                if contrast < p_contrast: continue
            
            extent = float(area) / (bw * bh)
            aspect = float(bw)/bh if bh>0 else 0
            if area > 20 and extent > 0.90: continue
            if aspect > 3.0 or aspect < 0.33: continue
            
            if do_dipole:
                pad_d = 4
                dy0, dy1 = max(0, by-pad_d), min(h_img, by+bh+pad_d)
                dx0, dx1 = max(0, bx-pad_d), min(w_img, bx+bw+pad_d)
                if cv2.minMaxLoc(gray_a[dy0:dy1, dx0:dx1])[0] < 15: continue
                
            candidates.append({
                'x': cx, 'y': cy, 'area': area,
                'sharp': sharpness, 'contrast': contrast,
                'peak': peak, 'rise': rise,
                'val_b': val_b, 'val_c': val_c,
                'crop_off': (x_off, y_off),
                'manual': False
            })

        # 4. Cheap Score & Top-K
        if candidates:
            # --- Cheap Score ---
            if config_dict['cheap_mode'] == 'robust_z' and len(candidates) > 5:
                rises = np.array([c['rise'] for c in candidates])
                conts = np.array([c['contrast'] for c in candidates])
                sharps = np.array([c['sharp'] for c in candidates])
                areas = np.array([c['area'] for c in candidates])
                
                def get_z(arr):
                    med = np.median(arr)
                    mad = np.median(np.abs(arr - med))
                    if mad < 1e-6: return arr - med
                    return (arr - med) / (1.4826 * mad)
                    
                z_rise = get_z(rises)
                z_cont = get_z(conts)
                z_sharp = get_z(sharps)
                z_area = get_z(areas)
                
                scores = (ProcessingConfig.W_RISE * np.clip(z_rise, -5, 5) + 
                          ProcessingConfig.W_CONTRAST * np.clip(z_cont, -5, 5) +
                          ProcessingConfig.W_SHARP * np.clip(z_sharp, -5, 5) - 
                          ProcessingConfig.W_AREA_PENALTY * np.abs(z_area))
                for i, c in enumerate(candidates):
                    c['cheap_score'] = float(scores[i])
            else:
                for c in candidates:
                    c['cheap_score'] = c['rise']
            
            # --- Top-K Union ---
            if config_dict['topk_union']:
                c_cheap = sorted(candidates, key=lambda x: x['cheap_score'], reverse=True)[:config_dict['topk_cheap']]
                c_rise = sorted(candidates, key=lambda x: x['rise'], reverse=True)[:config_dict['topk_rise']]
                c_cont = sorted(candidates, key=lambda x: x['contrast'], reverse=True)[:config_dict['topk_contrast']]
                
                unique_map = {}
                for c in c_cheap + c_rise + c_cont:
                    key = (c['x'], c['y'])
                    if key not in unique_map:
                        unique_map[key] = c
                top_candidates = list(unique_map.values())
            else:
                candidates.sort(key=lambda x: x['cheap_score'], reverse=True)
                top_candidates = candidates[:config_dict['topk_cheap']]
        else:
            top_candidates = []

        # 5. Prepare Patch Tensors (CPU)
        patch_tensors = []
        final_candidates = []
        
        for cand in top_candidates:
            try:
                t = _prepare_patch_tensor_80_static(gray_a, gray_b, gray_c, cand['x'], cand['y'], crop_sz=config_dict['crop_sz'])
                patch_tensors.append(t)
                final_candidates.append(cand)
            except Exception:
                pass # Skip failed patches

        t_stage_a = time.time() - t0
        return {
            'name': name,
            'candidates': final_candidates,
            'patch_tensors': patch_tensors,
            'crop_rect': crop_rect,
            'n_raw': len(candidates),
            't_stage_a': t_stage_a
        }

    except Exception as e:
        return {'error': str(e), 'name': name, 'traceback': traceback.format_exc()}


# ================= 配置文件管理 =================
CONFIG_FILE = os.path.join(os.getcwd(), "SCANN_config.json")

class ConfigManager:
    @staticmethod
    def load():
        default = {
            "last_folder": "",
            "thresh": 80,
            "min_area": 6,
            "sharpness": 1.2,
            "contrast": 15,
            "kill_flat": True,
            "kill_hist": True,
            "kill_dipole": True,
            "auto_crop": True,
            "edge_margin": 10,
            "auto_clear_cache": False,
            "dynamic_thresh": False,
            "max_sharpness": 5.0,
            "model_path": "",
            "crowd_high_score": 0.85,
            "crowd_high_count": 10,
            "crowd_high_penalty": 0.50,
            "jpg_download_dir": "",
            "fits_download_dir": ""
        }
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    default.update(data)
            except: pass
        return default

    @staticmethod
    def save(data):
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except: pass

# ================= Fix A: 修复版 ImageViewer =================
class ImageViewer(QGraphicsView):
    # 发送点击的图片坐标 (x, y)
    point_selected = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # 交互设置
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag) 
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(20, 20, 20)))
        
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

    def set_image(self, cv_img):
        """加载 OpenCV 图片"""
        if cv_img is None: return
        if not cv_img.flags['C_CONTIGUOUS']: cv_img = np.ascontiguousarray(cv_img)
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        qimg = QImage(cv_img.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qimg)
        self.pixmap_item.setPixmap(pixmap)
        
        # 仅在首次加载或场景为空时自动适配
        if self.scene.sceneRect().isEmpty():
            self.scene.setSceneRect(QRectF(pixmap.rect()))
            self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        else:
            # 保持当前视图区域不变
            self.scene.setSceneRect(QRectF(pixmap.rect()))

    def draw_overlays(self, candidates, current_idx, hide_all=False):
        """绘制圆圈和标记"""
        # 清除旧的标记（保留 pixmap_item）
        for item in self.scene.items():
            if item != self.pixmap_item:
                self.scene.removeItem(item)

        if hide_all: return

        font = QFont("Arial", 12, QFont.Bold)
        
        for i, cand in enumerate(candidates):
            cx, cy = cand['x'], cand['y']
            is_manual = cand.get('manual', False)
            is_saved = cand.get('saved', False) # 新增：检查是否已保存
            is_selected = (i == current_idx)

            if is_manual:
                color = QColor(255, 0, 255) # 紫色 (手动)
            else:
                color = QColor(0, 255, 0)   # 绿色 (自动)
            
            # 如果已保存，给一个特殊颜色（例如青色或深色），防止混淆
            if is_saved:
                color = QColor(0, 255, 255) # 青色 (已保存)

            pen_width = 3 if is_selected else 2
            if is_selected: color = QColor(255, 0, 0) # 选中变红

            radius = 12
            ellipse = self.scene.addEllipse(cx - radius, cy - radius, radius*2, radius*2, QPen(color, pen_width))
            ellipse.setZValue(10) 
            
            # 只有选中时，或者未保存时才显示文字，避免画面太乱？
            # 或者一直显示 ID
            text = self.scene.addText(str(cand.get('id', i+1)), font)
            text.setDefaultTextColor(QColor(255, 255, 0))
            text.setPos(cx + 10, cy - 10)
            text.setZValue(10)

    def wheelEvent(self, event: QWheelEvent):
        """滚轮缩放"""
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        if event.angleDelta().y() > 0:
            self.scale(zoom_in_factor, zoom_in_factor)
        else:
            self.scale(zoom_out_factor, zoom_out_factor)

    def mousePressEvent(self, event):
        # 右键触发平移 (Hack: 模拟左键点击 ScrollHandDrag)
        if event.button() == Qt.RightButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            fake_event = QMouseEvent(QEvent.MouseButtonPress, event.pos(), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
            super().mousePressEvent(fake_event)
        
        # 左键触发选点
        elif event.button() == Qt.LeftButton:
            self.setDragMode(QGraphicsView.NoDrag)
            scene_pos = self.mapToScene(event.pos())
            if self.pixmap_item.boundingRect().contains(scene_pos):
                self.point_selected.emit(int(scene_pos.x()), int(scene_pos.y()))
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self.setDragMode(QGraphicsView.NoDrag)
            self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

# ================= 批量处理线程 =================
# ================= 批量处理线程 =================
class BatchWorker(QThread):
    progress = pyqtSignal(int, int, str) 
    finished = pyqtSignal(dict)

    def __init__(self, groups, params):
        super().__init__()
        self.groups = groups
        self.params = params
        self._is_running = True
        
        # === AI 初始化 ===
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            
        self.model = None
        self.has_model = False
        
        # Normalization constants (ImageNet)
        self.norm_mean = torch.tensor([0.2601623164967817, 0.2682929013103806, 0.26861570225529907]).view(1, 3, 1, 1).to(self.device)
        self.norm_std = torch.tensor([0.09133092247248126, 0.10773878132887775, 0.10867911864809723]).view(1, 3, 1, 1).to(self.device)
        
        # Load Model
        model_path = self.params.get('model_path', '')
        if not model_path:
             script_dir = os.path.dirname(os.path.abspath(__file__))
             model_path = os.path.join(script_dir, "best_model.pth")
        
        print(f"DEBUG: 正在尝试加载模型: {model_path}")
        print(f"DEBUG: 使用设备: {self.device}")

        if os.path.exists(model_path):
            try:
                # 1. Structure
                self.model = models.resnet18(pretrained=False)
                num_ftrs = self.model.fc.in_features
                self.model.fc = torch.nn.Linear(num_ftrs, 2)
                
                # 2. Weights
                ckpt = torch.load(model_path, map_location=self.device)
                
                state_dict = None
                if isinstance(ckpt, dict):
                    if "state" in ckpt: state_dict = ckpt["state"]
                    elif "model_state" in ckpt: state_dict = ckpt["model_state"]
                    else: state_dict = ckpt
                else:
                    state_dict = ckpt
                
                # Clean prefix
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v
                
                # Strict Load
                self.model.load_state_dict(new_state_dict, strict=True)
                self.model.to(self.device)
                self.model.eval()
                self.has_model = True
                
                print(f"✅✅✅ AI 模型加载成功！")
                
            except Exception as e:
                print("\n❌❌❌ AI 模型加载失败！")
                traceback.print_exc()
                self.has_model = False
                raise e # Fail-fast
        else:
            print(f"❌ 未找到模型文件: {model_path}")
            self.has_model = False
            raise FileNotFoundError(f"AI Model not found at: {model_path}")

        # Initialize Async DB Writer
        DatabaseManager.init_async()

    def stop(self):
        self._is_running = False

    def verify_model_ready(self):
        """Fail-fast check before batch run"""
        if not self.has_model:
            raise RuntimeError(f"AI Model NOT Ready")
        
        print("DEBUG: Performing Model Dry-Run...")
        try:
            # Dummy batch [1, 3, 224, 224] to verify model structure
            dummy = torch.randn(1, 3, 224, 224).to(self.device)
            dummy = (dummy - self.norm_mean) / self.norm_std
            with torch.no_grad():
                _ = self.model(dummy)
            print("✅ Dry-Run Passed.")
        except Exception as e:
            print("❌ Dry-Run Failed!")
            traceback.print_exc()
            raise RuntimeError(f"Model Dry-Run Failed: {e}")

    def _compute_params_hash(self):
        key_params = {
            'thresh': self.params['thresh'],
            'min_area': self.params['min_area'],
            'sharpness': self.params['sharpness'],
            'max_sharpness': self.params.get('max_sharpness', 5.0),
            'contrast': self.params['contrast'],
            'edge_margin': self.params.get('edge_margin', 10),
            'kill_flat': self.params['kill_flat'],
            'kill_hist': self.params['kill_hist'],
            'kill_dipole': self.params['kill_dipole'],
            'dynamic_thresh': self.params.get('dynamic_thresh', False),
            'model_path': self.params.get('model_path', ''),
            'topk_cheap': ProcessingConfig.TOPK_CHEAP,
            'topk_union': ProcessingConfig.TOPK_UNION
        }
        import hashlib
        s = json.dumps(key_params, sort_keys=True)
        return hashlib.md5(s.encode('utf-8')).hexdigest()

    def run(self):
        # 1. Fail-fast check
        try:
            self.verify_model_ready()
        except Exception as e:
            print(f"❌ Batch Aborted: {e}")
            traceback.print_exc()
            self.finished.emit({}) 
            raise e 
            return

        print("DEBUG: Loading DB summaries...")
        db_summaries = DatabaseManager.load_summaries_map()
        
        results = {}
        total = len(self.groups)
        count = 0
        current_hash = self._compute_params_hash()
        
        sorted_keys = sorted(self.groups.keys())
        
        # --- Parallel Execution Setup ---
        executor = ThreadPoolExecutor(max_workers=ProcessingConfig.NUM_WORKERS)
        futures = set()
        
        worker_config = {
            'crop_sz': ProcessingConfig.CROP_SZ,
            'cheap_mode': ProcessingConfig.CHEAP_MODE,
            'topk_union': ProcessingConfig.TOPK_UNION,
            'topk_cheap': ProcessingConfig.TOPK_CHEAP,
            'topk_rise': ProcessingConfig.TOPK_RISE,
            'topk_contrast': ProcessingConfig.TOPK_CONTRAST
        }

        # Global Inference Batching
        pending_inference_items = [] # list of {'name': name, 'cand_idx': i, 'tensor': t}
        pending_results_map = {} # name -> {'candidates': [], 'remaining': N, 'crop_rect': ...}
        
        from concurrent.futures import wait, FIRST_COMPLETED

        def flush_inference_batch(force=False):
            nonlocal pending_inference_items, count
            BATCH_SIZE = ProcessingConfig.INFER_CHUNK
            
            while len(pending_inference_items) >= BATCH_SIZE or (force and pending_inference_items):
                # Take chunk
                chunk_size = BATCH_SIZE if len(pending_inference_items) >= BATCH_SIZE else len(pending_inference_items)
                batch_items = pending_inference_items[:chunk_size]
                pending_inference_items = pending_inference_items[chunk_size:]
                
                # Stack & Infer
                try:
                    tensors = [item['tensor'] for item in batch_items]
                    stack = torch.stack(tensors).to(self.device, non_blocking=True)
                    
                    # Resize & Norm on GPU
                    stack = torch.nn.functional.interpolate(stack, size=ProcessingConfig.RESIZE_HW, mode='bilinear', align_corners=False)
                    stack = (stack - self.norm_mean) / self.norm_std
                    
                    with torch.no_grad():
                        with torch.amp.autocast('cuda', enabled=(self.device.type=='cuda')):
                            logits = self.model(stack)
                            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    
                    # Distribute results
                    updates_by_name = {}
                    for idx, prob in enumerate(probs):
                        item = batch_items[idx]
                        name = item['name']
                        cand_idx = item['cand_idx']
                        
                        if name not in updates_by_name: updates_by_name[name] = []
                        updates_by_name[name].append((cand_idx, prob))
                        
                    # Apply updates & Check completion
                    for name, updates in updates_by_name.items():
                        cands = pending_results_map[name]['candidates']
                        for c_idx, score in updates:
                            cands[c_idx]['ai_score'] = float(score)
                            
                        pending_results_map[name]['remaining'] -= len(updates)
                        
                        if pending_results_map[name]['remaining'] <= 0:
                            final_cands = [c for c in cands if 'ai_score' in c]
                            p = self.params
                            hs = float(p.get('crowd_high_score', 0.85))
                            hc = int(p.get('crowd_high_count', 10))
                            hp = float(p.get('crowd_high_penalty', 0.50))
                            high_cnt = sum(1 for c in final_cands if c.get('ai_score', 0) >= hs)
                            if high_cnt > hc:
                                for c in final_cands:
                                    if c.get('ai_score', 0) >= hs:
                                        c['ai_score'] = max(0.0, float(c['ai_score']) - hp)
                            crop_rect = pending_results_map[name]['crop_rect']
                            
                            # --- 数据保护：合并已有的手动/判决目标 ---
                            existing_full = DatabaseManager.get_record(name)
                            if existing_full and "candidates" in existing_full:
                                for ec in existing_full["candidates"]:
                                    if ec.get("manual", False) or ec.get("verdict") is not None:
                                        # 检查是否重复 (基于坐标)
                                        is_dup = False
                                        for nc in final_cands:
                                            if abs(nc['x'] - ec['x']) < 5 and abs(nc['y'] - ec['y']) < 5:
                                                is_dup = True
                                                # 如果重复，保留已有的判决
                                                if ec.get("verdict"):
                                                    nc["verdict"] = ec["verdict"]
                                                    nc["saved"] = ec.get("saved", True)
                                                break
                                        if not is_dup:
                                            final_cands.append(ec)
                            
                            DatabaseManager.update_record(name, final_cands, crop_rect=crop_rect, params_hash=current_hash)
                            results[name] = {"candidates": final_cands, "status": "unseen", "crop_rect": crop_rect}
                            
                            del pending_results_map[name]
                            
                            count += 1
                            if count % 5 == 0:
                                self.progress.emit(count, total, f"AI处理中: {name}")

                except Exception as e:
                    print(f"❌ Global Batch Inference Error")
                    traceback.print_exc()
                    raise e

        # --- Main Loop ---
        for name in sorted_keys:
            if not self._is_running: break
            
            summary = db_summaries.get(name)
            if summary:
                cached_hash = summary.get('params_hash', '')
                if summary.get('has_ai', 0) and summary.get('candidates_count', 0) > 0 and cached_hash == current_hash:
                    record = DatabaseManager.get_record(name)
                    if record:
                        results[name] = record
                        count += 1
                        self.progress.emit(count, total, f"已从库加载: {name}")
                        continue
            
            # Submit Task (with bounded buffer)
            while len(futures) >= ProcessingConfig.NUM_WORKERS * 2:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for f in done:
                    res = f.result()
                    if not res: continue 
                    if 'error' in res: raise RuntimeError(res['error'])
                    
                    r_name = res['name']
                    r_cands = res['candidates']
                    r_tensors = res['patch_tensors']
                    
                    if not r_cands:
                        # --- 数据保护：哪怕没发现新目标，也要保留旧的手动目标 ---
                        final_cands = []
                        existing_full = DatabaseManager.get_record(r_name)
                        if existing_full and "candidates" in existing_full:
                            for ec in existing_full["candidates"]:
                                if ec.get("manual", False) or ec.get("verdict") is not None:
                                    final_cands.append(ec)
                        
                        DatabaseManager.update_record(r_name, final_cands, crop_rect=res['crop_rect'], params_hash=current_hash)
                        count += 1
                        continue

                    pending_results_map[r_name] = {
                        'candidates': r_cands,
                        'remaining': len(r_cands),
                        'crop_rect': res['crop_rect']
                    }
                    
                    for i, t in enumerate(r_tensors):
                        pending_inference_items.append({'name': r_name, 'cand_idx': i, 'tensor': t})
                    
                    flush_inference_batch()

            if not self._is_running: break
            
            # Submit new task
            future = executor.submit(process_stage_a, name, self.groups[name], self.params, worker_config)
            futures.add(future)

        # Drain remaining
        while futures:
            if not self._is_running: break
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for f in done:
                res = f.result()
                if not res: continue
                if 'error' in res: raise RuntimeError(res['error'])
                
                r_name = res['name']
                r_cands = res['candidates']
                r_tensors = res['patch_tensors']
                
                if not r_cands:
                    # --- 数据保护：哪怕没发现新目标，也要保留旧的手动目标 ---
                    final_cands = []
                    existing_full = DatabaseManager.get_record(r_name)
                    if existing_full and "candidates" in existing_full:
                        for ec in existing_full["candidates"]:
                            if ec.get("manual", False) or ec.get("verdict") is not None:
                                final_cands.append(ec)
                                
                    DatabaseManager.update_record(r_name, final_cands, crop_rect=res['crop_rect'], params_hash=current_hash)
                    count += 1
                    continue

                pending_results_map[r_name] = {
                    'candidates': r_cands,
                    'remaining': len(r_cands),
                    'crop_rect': res['crop_rect']
                }
                
                for i, t in enumerate(r_tensors):
                    pending_inference_items.append({'name': r_name, 'cand_idx': i, 'tensor': t})
                
                flush_inference_batch()

        # Final flush
        flush_inference_batch(force=True)
        
        executor.shutdown()
        self.finished.emit(results)


class SuspectListWidget(QListWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main = main_window
        self.setFont(QFont("Arial", 11))

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_S:
            self.main.handle_suspect_action(True)
        elif key == Qt.Key_D:
            self.main.handle_suspect_action(False)
        elif key == Qt.Key_Space:
            self.main.handle_suspect_skip()
        elif key == Qt.Key_R:
            try:
                self.main.btn_blink.click()
            except Exception:
                self.main.toggle_blink()
        else:
            super().keyPressEvent(event)

class SuspectGlobalKeyFilter(QObject):
    def __init__(self, main_window):
        super().__init__()
        self.main = main_window

    def eventFilter(self, obj, event):
        if not self.main._is_suspect_mode_active():
            return False

        et = event.type()
        if et not in (QEvent.ShortcutOverride, QEvent.KeyPress):
            return False

        key = event.key()
        if key not in (Qt.Key_S, Qt.Key_D, Qt.Key_Space, Qt.Key_R):
            return False

        if et == QEvent.ShortcutOverride:
            event.accept()
            return True

        if key == Qt.Key_S:
            self.main.handle_suspect_action(True)
            return True
        if key == Qt.Key_D:
            self.main.handle_suspect_action(False)
            return True
        if key == Qt.Key_Space:
            self.main.handle_suspect_skip()
            return True
        if key == Qt.Key_R:
            try:
                self.main.btn_blink.click()
            except Exception:
                self.main.toggle_blink()
            return True

        return False

from concurrent.futures import ThreadPoolExecutor

# ================= 主窗口 =================
class SCANN(QMainWindow):
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
        self.io_pool = ThreadPoolExecutor(max_workers=2) # 后台保存图片专用线程池
        self.suspects_data = [] # 缓存可疑目标数据，供列表使用
        self._suspect_shortcut_backup = {}
        self._t_recall_cached = None
        self._suspect_global_filter = SuspectGlobalKeyFilter(self)

        # 闪烁相关
        self.blink_timer = QTimer(self)
        self.blink_timer.setInterval(400) # 400ms 间隔
        self.blink_timer.timeout.connect(self.blink_tick)
        self.blink_state = 0 # 0=New, 1=Old

        os.makedirs("dataset/positive", exist_ok=True)
        os.makedirs("dataset/negative", exist_ok=True)

        self.cfg = ConfigManager.load()

        # Fix: Robust path finding for both .py and .exe (PyInstaller)
        if getattr(sys, 'frozen', False):
            self.base_path = os.path.dirname(sys.executable)
        else:
            self.base_path = os.path.dirname(os.path.abspath(__file__))

        # 优先使用配置中的模型路径
        self.model_path = self.cfg.get('model_path', '')
        if not self.model_path or not os.path.exists(self.model_path):
            # Fallback to default
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

        # 新增：数据库下载按钮
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
        
        h1 = QHBoxLayout(); h1.addWidget(self.lbl_jpg_path, 1); h1.addWidget(btn_set_jpg)
        h2 = QHBoxLayout(); h2.addWidget(self.lbl_fits_path, 1); h2.addWidget(btn_set_fits)
        path_vbox.addLayout(h1); path_vbox.addLayout(h2)
        path_group.setLayout(path_vbox)
        left_panel.addWidget(path_group)

        # 新增：显示可疑目标按钮
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
        
        # --- Page 0: 标准视图 (文件列表 + 参数 + 候选体) ---
        page0 = QWidget()
        p0_layout = QVBoxLayout(page0)
        p0_layout.setContentsMargins(0,0,0,0)
        
        self.file_list = QListWidget()
        self.file_list.currentRowChanged.connect(self.on_file_selected)
        self.file_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.file_list.customContextMenuRequested.connect(self.show_file_list_context_menu)
        
        p0_layout.addWidget(QLabel("文件列表 (绿=有目标, 蓝=已归档):"))
        p0_layout.addWidget(self.file_list, 2)

        # 参数设置
        gb = QGroupBox("检测参数")
        form = QFormLayout()
        
        self.spin_thresh = QSpinBox(); self.spin_thresh.setRange(5, 255); self.spin_thresh.setValue(self.cfg['thresh'])
        self.spin_min_area = QSpinBox(); self.spin_min_area.setRange(1, 100); self.spin_min_area.setValue(self.cfg['min_area'])
        self.spin_sharpness = QDoubleSpinBox(); self.spin_sharpness.setRange(1.0, 5.0); self.spin_sharpness.setSingleStep(0.1); self.spin_sharpness.setValue(self.cfg['sharpness'])
        self.spin_max_sharpness = QDoubleSpinBox(); self.spin_max_sharpness.setRange(1.0, 20.0); self.spin_max_sharpness.setSingleStep(0.1); self.spin_max_sharpness.setValue(self.cfg.get('max_sharpness', 5.0))
        
        # Fix B: 对比度参数加入 UI
        self.spin_contrast = QSpinBox(); self.spin_contrast.setRange(0, 100); self.spin_contrast.setValue(self.cfg['contrast'])
        
        self.spin_edge = QSpinBox(); self.spin_edge.setRange(0, 100); self.spin_edge.setValue(self.cfg.get('edge_margin', 10))
        
        self.cb_dynamic_thresh = QCheckBox("动态阈值 (Median+Offset)"); self.cb_dynamic_thresh.setChecked(self.cfg.get('dynamic_thresh', False))
        self.cb_dynamic_thresh.setToolTip("开启后，阈值 = 背景中位数 + 设定值")

        self.cb_kill_flat = QCheckBox("去除平坦光斑"); self.cb_kill_flat.setChecked(self.cfg['kill_flat'])
        self.cb_kill_history = QCheckBox("去除历史 (区域)"); self.cb_kill_history.setChecked(self.cfg['kill_hist'])
        self.cb_kill_history.setStyleSheet("color: red;")
        self.cb_kill_dipole = QCheckBox("去除偶极子"); self.cb_kill_dipole.setChecked(self.cfg['kill_dipole'])
        self.cb_auto_crop = QCheckBox("自动切除白边"); self.cb_auto_crop.setChecked(self.cfg['auto_crop'])
        
        self.spin_crowd_high_score = QDoubleSpinBox(); self.spin_crowd_high_score.setRange(0.0, 1.0); self.spin_crowd_high_score.setSingleStep(0.01); self.spin_crowd_high_score.setValue(self.cfg.get('crowd_high_score', 0.85))
        self.spin_crowd_high_count = QSpinBox(); self.spin_crowd_high_count.setRange(1, 500); self.spin_crowd_high_count.setValue(self.cfg.get('crowd_high_count', 10))
        self.spin_crowd_high_penalty = QDoubleSpinBox(); self.spin_crowd_high_penalty.setRange(0.0, 1.0); self.spin_crowd_high_penalty.setSingleStep(0.01); self.spin_crowd_high_penalty.setValue(self.cfg.get('crowd_high_penalty', 0.50))
        
        
        form.addRow("亮度阈值/Offset:", self.spin_thresh)
        form.addRow("最小面积:", self.spin_min_area)
        form.addRow("Min 锐度:", self.spin_sharpness) 
        form.addRow("Max 锐度:", self.spin_max_sharpness)
        # Fix B: UI 显示
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
        
        # --- Page 1: 可疑目标列表 (Suspect List) ---
        page1 = QWidget()
        p1_layout = QVBoxLayout(page1)
        p1_layout.setContentsMargins(0,0,0,0)
        self.suspect_list = SuspectListWidget(self)
        self.suspect_list.currentItemChanged.connect(self.on_suspect_current_changed)
        self.suspect_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.suspect_list.customContextMenuRequested.connect(lambda pos: self.show_list_context_menu(self.suspect_list, pos))
        p1_layout.addWidget(QLabel("🔥 高价值可疑目标 (按 AI 排序):"))
        p1_layout.addWidget(self.suspect_list)
        
        self.left_stack.addWidget(page0)
        self.left_stack.addWidget(page1)
        
        left_panel.addWidget(self.left_stack, 1) # Give stack layout priority

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
        # 不要把 Blink 按钮加到互斥组里，它是独立开关
        
        right_panel.addLayout(view_layout)

        self.view_context = ImageViewer()
        self.view_context.point_selected.connect(self.on_context_click) 
        
        right_panel.addWidget(self.view_context, 1)
        right_panel.addWidget(QLabel("提示：左键点击=设点 | 滚轮=缩放 | 右键拖拽=平移"))

        layout.addLayout(right_panel, 3)

    def delete_source_files(self, name, list_item):
        if name not in self.groups: return
        paths = self.groups[name]
        errs = []
        for k in ['a', 'b', 'c']:
            path = paths.get(k)
            if path and os.path.exists(path):
                try: os.remove(path)
                except Exception as e: errs.append(str(e))
        if errs:
            QMessageBox.warning(self, "删除出错", "\n".join(errs))
            return
        del self.groups[name]
        DatabaseManager.delete_record(name)
        if name in self.batch_results: del self.batch_results[name]
        row = self.file_list.row(list_item)
        self.file_list.takeItem(row)
        if self.current_group == name:
            self.current_preview_img = None
            self.view_context.scene.clear()
            self.lbl_triplet.clear()
            self.cand_list.clear()
            self.candidates = []

    def select_model_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", self.base_path, "PyTorch Model (*.pth)")
        if path:
            self.model_path = path
            self.cfg['model_path'] = path
            self.lbl_model.setText(os.path.basename(path))
            self.lbl_model.setToolTip(path)
            ConfigManager.save(self.cfg)
            print(f"Model switched to: {path}")

    def closeEvent(self, event):
        self.save_current_config()
        
        # 停止计算线程
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()

        # Ensure all async writes are flushed
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

    def save_current_config(self):
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

    def load_folder(self, path=None):
        if not path:
            path = QFileDialog.getExistingDirectory(self, "选择图片文件夹", self.cfg['last_folder'])
        if not path: return
        self.cfg['last_folder'] = path
        
        # 保存当前选中项的 ID
        old_selection = None
        curr_item = self.file_list.currentItem()
        if curr_item:
            old_selection = curr_item.data(Qt.UserRole)
            
        self.groups = {}
        self.batch_results = {}
        self.file_list.clear()
        files = os.listdir(path)
        for f in files:
            if not f.lower().endswith(('.jpg', '.jpeg', '.png')): continue
            name, _ = os.path.splitext(f)
            if len(name) < 2: continue
            suffix = name[-1].lower()
            stem = name[:-1]
            if suffix in ['a', 'b', 'c']:
                if stem not in self.groups: self.groups[stem] = {}
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

    def _on_all_downloads_done(self, success_count, fail_count):
        """当队列中所有下载任务完成后触发"""
        msg = f"🎉 批量下载任务已完成！成功: {success_count}"
        if fail_count > 0:
            msg += f" | 失败: {fail_count}"
        self.statusBar().showMessage(msg, 10000)
        
        # 刷新当前文件夹
        current_folder = self.cfg.get('last_folder')
        if current_folder and os.path.exists(current_folder):
            self.load_folder(current_folder)

    def start_batch_run(self):
        if not self.groups: return
        self.save_current_config()
        if self.cb_auto_clear.isChecked():
            DatabaseManager.clear_all()
            self.batch_results = {}
            for i in range(self.file_list.count()):
                item = self.file_list.item(i)
                name = item.data(Qt.UserRole)
                if not name: name = item.text().split(" ")[0]
                item.setText(name)
                item.setForeground(QColor(0, 0, 0))
                item.setFont(QFont("Arial", 10))
            print("缓存已自动清除")

        # Fix: 过滤完整组，防止 KeyError
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
        
        # 如果已有正在运行的任务，尝试中止它
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()

        # 只传完整组给 Worker
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

    def toggle_suspects_mode(self):
        # 如果当前已经在可疑模式，则退出
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

        # 2. 排序 (高分在前)
        all_suspects.sort(key=lambda x: x.get('ai_score', 0), reverse=True)
        
        # 3. 限制数量并填充列表
        limit = 500
        total_count = len(all_suspects)
        if len(all_suspects) > limit:
            print(f"Warning: Too many suspects ({len(all_suspects)}), showing top {limit}")
            all_suspects = all_suspects[:limit]
            
        self.suspects_data = all_suspects # 保存数据引用
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
            
        # 4. 切换视图
        self.left_stack.setCurrentIndex(1)
        self.btn_show_suspects.setText(f"🔙 退出可疑列表 (Top {limit}/{total_count})")
        self.btn_show_suspects.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        
        if self.suspect_list.count() > 0:
            self.suspect_list.setCurrentRow(0)
            self.suspect_list.setFocus() # 确保焦点在列表上，以便直接按键

    def on_suspect_current_changed(self, current, previous):
        if not current: return
        row = self.suspect_list.row(current)
        if row < 0 or row >= len(self.suspects_data): return
        
        cand = self.suspects_data[row]
        # 跳转但不激活主窗口，保持焦点在列表
        self.jump_to_candidate(cand, activate_main=False)
        self.suspect_list.setFocus()

    def handle_suspect_action(self, is_positive):
        """处理 S/D 快捷键"""
        row = self.suspect_list.currentRow()
        if row < 0 and self.suspect_list.count() > 0:
            self.suspect_list.setCurrentRow(0)
            row = 0
        if row < 0 or row >= len(self.suspects_data): return
        
        # 获取当前选中的可疑目标数据
        cand_wrapper = self.suspects_data[row]
        
        # 1. 保存当前样本 (传入明确的目标)
        # 关键修复：直接传入 cand_wrapper，让 save_dataset_sample 内部去匹配主界面的 candidates
        self.save_dataset_sample(is_positive, auto_jump=False, explicit_candidate=cand_wrapper)
        cand_wrapper['verdict'] = 'real' if is_positive else 'bogus'
        
        # 2. 更新列表项视觉状态 (手动在这里更新一次，确保反应最快)
        item = self.suspect_list.item(row)
        text = item.text()
        if "[已" not in text:
             suffix = " [已存真]" if is_positive else " [已存假]"
             item.setText(text + suffix)
        
        color = QColor(0, 150, 0) if is_positive else QColor(150, 0, 0)
        item.setForeground(color)
        
        # 3. 自动跳到下一行
        self.handle_suspect_skip()

    def handle_suspect_skip(self):
        """处理 Space 快捷键 (跳过)"""
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
        """从可疑列表跳转到具体图像和目标"""
        target_stem = cand_wrapper.get('stem')
        # 优先使用坐标匹配，比 ID 更稳健
        target_x = cand_wrapper.get('x')
        target_y = cand_wrapper.get('y')
        
        if not target_stem: return
        
        # 1. 切换文件
        if target_stem != self.current_group:
            found_row = -1
            for i in range(self.file_list.count()):
                item = self.file_list.item(i)
                name = item.data(Qt.UserRole)
                if not name: name = item.text().split(" ")[0]
                if name == target_stem:
                    found_row = i
                    break
            
            if found_row != -1:
                self.file_list.setCurrentRow(found_row)
            else:
                QMessageBox.warning(self, "错误", f"在列表中找不到文件: {target_stem}")
                return
                
        # 2. 切换候选体 (通过坐标匹配)
        target_row = -1
        for i, c in enumerate(self.candidates):
            # 坐标完全一致则认为是同一个
            if c.get('x') == target_x and c.get('y') == target_y:
                target_row = i
                break
        
        if target_row != -1:
            # 始终同步主列表的选中项，避免残留到旧的行号
            self.cand_list.setCurrentRow(target_row)
            self.on_candidate_selected(target_row)
            if activate_main:
                self.activateWindow()
            try:
                # 保持可疑列表拥有焦点，以便 S/D/Space 连续操作
                if self._is_suspect_mode_active():
                    self.suspect_list.setFocus()
            except Exception:
                pass
        else:
            print(f"Warning: Candidate at ({target_x}, {target_y}) not found in {target_stem}")

    def update_progress(self, curr, total, msg):
        self.progress_bar.setValue(int(curr / total * 100))
        self.lbl_title.setText(msg)

    def on_batch_finished(self, results):
        self.btn_batch.setEnabled(True)
        self.btn_batch.setText("⚡ 批量计算")
        self.progress_bar.setVisible(False)
        self.batch_results.update(results)
        first_hit_row = -1
        total_hits = 0
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            name = item.data(Qt.UserRole)
            if not name: name = item.text().split(" ")[0]
            rec = results.get(name)
            if not rec: rec = self.batch_results.get(name)
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
                    if first_hit_row == -1: first_hit_row = i
                else:
                    item.setText(name)
                    item.setForeground(QColor(0, 0, 0))
                    item.setFont(QFont("Arial", 10))
        QMessageBox.information(self, "完成", f"处理结束，本次发现 {total_hits} 个新目标")
        if first_hit_row != -1:
            self.file_list.setCurrentRow(first_hit_row)
            self.load_candidates_from_batch(self.file_list.item(first_hit_row).data(Qt.UserRole))

    def get_auto_crop_rect(self, img):
        # 仅当没有缓存时使用
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return 0, 0, img.shape[1], img.shape[0]
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        pad = 2
        return max(0, x+pad), max(0, y+pad), max(1, w-2*pad), max(1, h-2*pad)

    def switch_main_view(self, mode):
        # mode: 0=Diff(A), 1=New(B), 2=Ref(C)
        if not hasattr(self, 'img_a'): return
        
        # 如果正在闪烁，切换手动视图时先停止闪烁（可选，这里我选择不强制停止，但更新选中状态）
        if self.blink_timer.isActive():
            self.btn_blink.setChecked(False)
            self.blink_timer.stop()
        
        target_img = None
        if mode == 0: target_img = self.img_a
        elif mode == 1: target_img = self.img_b
        elif mode == 2: target_img = self.img_c
        
        if target_img is not None:
            self.current_preview_img = target_img
            self.view_context.set_image(target_img)
            # 切换图片后，必须重绘圆圈
            curr_row = self.cand_list.currentRow()
            self.view_context.draw_overlays(self.candidates, curr_row, self.btn_hide_overlay.isChecked())
            
            # 更新按钮状态
            if mode == 0: self.btn_view_a.setChecked(True)
            elif mode == 1: self.btn_view_b.setChecked(True)
            elif mode == 2: self.btn_view_c.setChecked(True)

    def toggle_overlay(self):
        self.view_context.draw_overlays(self.candidates, self.cand_list.currentRow(), self.btn_hide_overlay.isChecked())

    def toggle_blink(self):
        if self.btn_blink.isChecked():
            self.update_blink_speed() # 确保使用当前选择的速度
            self.blink_timer.start()
        else:
            self.blink_timer.stop()
            # 停止时回到 Diff 视图
            self.switch_main_view(0)
    
    def update_blink_speed(self):
        text = self.combo_blink_speed.currentText()
        try:
            sec = float(text.replace('s', ''))
            self.blink_timer.setInterval(int(sec * 1000))
        except:
            self.blink_timer.setInterval(400) # 默认

    def blink_tick(self):
        if not hasattr(self, 'img_b') or not hasattr(self, 'img_c'): return
        
        # 在 New (B) 和 Ref (C) 之间切换
        self.blink_state = 1 - self.blink_state
        if self.blink_state == 0:
            # Show New
            self.view_context.set_image(self.img_b)
            self.btn_view_b.setChecked(True)
        else:
            # Show Ref
            self.view_context.set_image(self.img_c)
            self.btn_view_c.setChecked(True)
            
        # 保持圆圈绘制
        curr_row = self.cand_list.currentRow()
        self.view_context.draw_overlays(self.candidates, curr_row, self.btn_hide_overlay.isChecked())

    def open_db_download(self):
        win = DBDownloadWindow(self.downloader, self)
        win.exec_()

    def show_list_context_menu(self, list_widget, pos):
        item = list_widget.itemAt(pos)
        if not item: return
        
        menu = QMenu()
        download_action = menu.addAction("📥 下载对应 FITS 原图")
        action = menu.exec_(list_widget.mapToGlobal(pos))
        
        if action == download_action:
            self.download_fits_for_item(list_widget, item)

    def show_file_list_context_menu(self, pos):
        item = self.file_list.itemAt(pos)
        if not item: return
        
        menu = QMenu()
        act_download = menu.addAction("📥 下载此组对应的 FITS 原图")
        menu.addSeparator()
        act_clear = menu.addAction("🔄 清除缓存并重算")
        act_delete = menu.addAction("🔥 彻底删除源文件 (abc)")
        
        action = menu.exec_(self.file_list.mapToGlobal(pos))
        
        name = item.data(Qt.UserRole)
        if not name: name = item.text().split(" ")[0]

        if action == act_download:
            self.download_fits_for_item(self.file_list, item)
        elif action == act_clear:
            DatabaseManager.delete_record(name)
            if name in self.batch_results: del self.batch_results[name]
            QMessageBox.information(self, "提示", f"已清除 {name} 的缓存，请点击批量计算重新提取。")
            item.setText(name)
            item.setForeground(QColor(0,0,0))
            item.setFont(QFont("Arial", 10))
        elif action == act_delete:
            reply = QMessageBox.question(self, "高能预警", f"确定要永久删除 {name} 相关的三张图片吗？\n此操作不可恢复！", 
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.delete_source_files(name, item)

    def set_download_path(self, key):
        path = QFileDialog.getExistingDirectory(self, "选择保存目录", self.cfg.get(key, ""))
        if path:
            self.cfg[key] = path
            ConfigManager.save(self.cfg)
            # 更新 UI 显示 (完整路径)
            if key == 'jpg_download_dir':
                self.lbl_jpg_path.setText(f"JPG: {path}")
                self.lbl_jpg_path.setToolTip(path)
            else:
                self.lbl_fits_path.setText(f"FITS: {path}")
                self.lbl_fits_path.setToolTip(path)

    def download_fits_for_item(self, list_widget, item):
        # 从 item 获取 stem
        if list_widget == self.suspect_list:
            # 文本格式示例: [IC 1934.fts2] ID:1 | AI: 9.83%
            text = item.text()
            match = re.search(r'\[(.*?)\]', text)
            if not match: return
            stem = match.group(1) # 移除 .lower()，保持原名查询
        elif list_widget == self.file_list:
            # 文件列表中的项，直接获取 stem
            stem = item.data(Qt.UserRole)
            if not stem: stem = item.text().split(" ")[0] # 移除 .lower()
        else:
            # cand_list 选中的通常是当前 self.current_group
            stem = self.current_group # 移除 .lower()

        # 从数据库查找
        linkage = self.downloader.get_linkage(stem)
        if not linkage:
            self.statusBar().showMessage(f"❌ 数据库中未找到 {stem} 的下载链接，请先在数据库浏览器中扫描。")
            return

        if linkage['status'] == 'downloaded' and linkage['local_fits_path'] and os.path.exists(linkage['local_fits_path']):
            self.statusBar().showMessage(f"✅ FITS 原图已存在: {linkage['local_fits_path']}")
            return

        # 获取配置好的保存目录
        save_dir = self.cfg.get('fits_download_dir')
        if not save_dir or not os.path.exists(save_dir):
            save_dir = QFileDialog.getExistingDirectory(self, "选择 FITS 保存目录")
            if not save_dir: return
            self.cfg['fits_download_dir'] = save_dir
            ConfigManager.save(self.cfg)
            self.lbl_fits_path.setText(f"FITS: {save_dir}")
            self.lbl_fits_path.setToolTip(save_dir)

        # 再次检查所选目录是否已存在该文件
        filename = self.downloader.clean_filename(os.path.basename(linkage['remote_fits_url']))
        if not filename.lower().endswith(".fts"):
            filename += ".fts"
        save_path = os.path.join(save_dir, filename)
        
        if os.path.exists(save_path):
            self.downloader.update_linkage(stem, status='downloaded', local_fits_path=save_path)
            self.statusBar().showMessage(f"✅ FITS 原图已存在于所选目录: {save_path}")
            return

        self.downloader.submit_download(stem, linkage['remote_fits_url'], save_dir)

    def on_file_selected(self, row):
        item = self.file_list.item(row)
        if item is None: return
        
        # 切换新文件时，重置视图区域（让它自动适配全图）
        self.view_context.scene.setSceneRect(QRectF()) 

        stem = item.data(Qt.UserRole)
        if not stem: stem = item.text().split(" ")[0]
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

        # Fix C: 优先使用 Batch 算出的 crop_rect，保证坐标一致性
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
        
        # 逻辑修改：如果开启了闪烁，切换图片时保持闪烁；否则切回 Diff
        if self.btn_blink.isChecked():
            if not self.blink_timer.isActive():
                self.blink_timer.start()
            # 立即刷新一下显示，避免等待 Timer 造成的延迟
            if self.blink_state == 0:
                self.view_context.set_image(self.img_b)
                self.btn_view_b.setChecked(True)
            else:
                self.view_context.set_image(self.img_c)
                self.btn_view_c.setChecked(True)
            # 重绘圆圈
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

    def load_candidates_from_batch(self, name):
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
        try:
            return hasattr(self, 'left_stack') and self.left_stack.currentIndex() == 1
        except Exception:
            return False

    def refresh_cand_list(self):
        self.cand_list.clear()
        
        # 1. 读取阈值 (从模型文件里读，如果没有就给个保守默认值 0.5)
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
            
        # 2. 排序：自动候选优先，其次按 AI 分数高到低；手动目标固定排在末尾
        self.candidates.sort(key=lambda c: (1 if c.get('manual', False) else 0, -c.get('ai_score', 0)))
        
        for i, c in enumerate(self.candidates):
            c['id'] = i + 1 # 重置 ID 以匹配显示顺序
            
            # === 这里是获取 AI 分数的关键 ===
            ai_score = c.get('ai_score', 0)
            score_str = f"{ai_score * 100:.2f}%"
            
            # 获取其他参数
            sharp = c.get('sharp', 0)
            area = c.get('area', 0)
            peak = c.get('peak', 0)
            rise = c.get('rise', 0)
            
            # 恢复之前的判决状态
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
                    item.setForeground(QColor(255, 0, 255)) # 紫色
            else:
                # === 重点：把 AI 分数拼接到显示的文字里 ===
                # 格式修改为： AI:xx% S:锐度 A:面积 D:峰值 R:增亮
                txt = f"#{i+1} AI:{score_str} S:{sharp:.1f} A:{int(area)} D:{int(peak)} R:{int(rise)}{verdict_suffix}"
                item = QListWidgetItem(txt)
                
                # 3. 颜色逻辑：
                # 如果已判决，优先显示判决颜色
                if verdict_color:
                    item.setForeground(verdict_color)
                # >= tR : 红色粗体 (高置信度)
                # < tR  : 灰色 (低置信度)
                elif ai_score >= tR: 
                    item.setForeground(QColor(255, 0, 0))
                    item.setFont(QFont("Arial", 11, QFont.Bold))
                else: 
                    item.setForeground(QColor(128, 128, 128))
            
            self.cand_list.addItem(item)

    def on_context_click(self, x, y):
        if self.current_preview_img is None: return
        
        # === 1. 立即执行“漏检诊断” ===
        self.perform_diagnosis(x, y)
        
        # 查找是否存在"未保存"的手动目标
        unsaved_manual_idx = -1
        for i, c in enumerate(self.candidates):
            # 只有当它是手动目标，且没有被标记为 saved 时，才会被覆盖
            if c.get('manual', False) and not c.get('saved', False):
                unsaved_manual_idx = i
                break
        
        if unsaved_manual_idx != -1:
            # 覆盖旧的未保存目标
            self.candidates[unsaved_manual_idx]['x'] = x
            self.candidates[unsaved_manual_idx]['y'] = y
            # 重置特征并保持为手动目标；不参与高分排序
            self.candidates[unsaved_manual_idx]['rise'] = 999
            self.candidates[unsaved_manual_idx]['ai_score'] = 0.0
            print(f"Updated unsaved manual target #{self.candidates[unsaved_manual_idx]['id']} to ({x}, {y})")
            manual_idx = unsaved_manual_idx
        else:
            # 创建新目标 (因为之前没有未保存的手动目标)
            next_id = len(self.candidates) + 1
            new_cand = {
                'id': next_id, 
                'x': x, 'y': y, 
                'area': 999, 'sharp': 9.9, 'peak': 255, 'contrast': 100,
                'rise': 999, 
                'crop_off': (0,0),
                'manual': True,
                'ai_score': 0.0,
                'saved': False # 初始为未保存
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
        """对点击位置进行全方位诊断，分析为何被漏检"""
        try:
            if not hasattr(self, 'img_a'): return
            
            # 准备数据 (转灰度)
            def to_gray(img):
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            
            gray_a = to_gray(self.img_a)
            gray_b = to_gray(self.img_b)
            gray_c = to_gray(self.img_c)
            h, w = gray_a.shape
            
            # 1. 提取局部区域 (7x7)
            r = 3
            x0, x1 = max(0, x-r), min(w, x+r+1)
            y0, y1 = max(0, y-r), min(h, y+r+1)
            
            roi_a = gray_a[y0:y1, x0:x1]
            roi_b = gray_b[y0:y1, x0:x1]
            roi_c = gray_c[y0:y1, x0:x1]
            
            if roi_a.size == 0: return
            
            # 2. 计算核心指标
            peak = float(np.max(roi_a))
            mean = float(np.mean(roi_a))
            median = float(np.median(roi_a))
            sharpness = peak / (mean + 1e-6)
            contrast = peak - median
            
            val_b = float(np.max(roi_b)) if roi_b.size > 0 else 0
            val_c = float(np.max(roi_c)) if roi_c.size > 0 else 0
            rise = val_b - val_c
            
            # 3. 检查各项规则
            reasons = []
            
            # A. 阈值检查
            thresh = self.cfg['thresh']
            if self.cfg.get('dynamic_thresh', False):
                # 简单估算背景：用全图中值近似，或者取大一点的局部
                bg_a = np.median(gray_a) # 这里用全图近似
                thresh += bg_a
            
            if peak < thresh:
                reasons.append(f"❌ 亮度不足 (Peak={peak:.1f} < {thresh:.1f})")
            
            # B. 形态学检查
            min_sharp = self.cfg['sharpness']
            if self.cfg['kill_flat'] and sharpness < min_sharp:
                reasons.append(f"❌ 过于平坦 (Sharp={sharpness:.2f} < {min_sharp})")
                
            min_contrast = self.cfg['contrast']
            if self.cfg['kill_flat'] and contrast < min_contrast:
                reasons.append(f"❌ 对比度低 (Cont={contrast:.1f} < {min_contrast})")
            
            # C. 边缘检查
            edge = self.cfg.get('edge_margin', 10)
            if x < edge or y < edge or x > w-edge or y > h-edge:
                reasons.append(f"❌ 位于边缘 (Edge < {edge})")
                
            # D. Rise 检查 (虽然现在不硬杀，但也提示)
            if rise < 0:
                reasons.append(f"⚠️ 负增亮 (Rise={rise:.1f})")
                
            # 4. 生成报告
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
                
            # 弹窗显示
            print("\n".join(msg)) # 控制台也打一份
            QMessageBox.information(self, "目标诊断报告", "\n".join(msg))
            
        except Exception as e:
            print(f"Diagnosis failed: {e}")

    def _crop_patch_common(self, src_img, cx, cy, crop_sz=80):
        """
        通用裁剪函数：从 src_img 的 (cx, cy) 处裁剪出 crop_sz*crop_sz 的图。
        如果越界，自动用黑色填充。
        """
        half = crop_sz // 2
        curr_h, curr_w = src_img.shape[:2]
        
        canvas = np.zeros((crop_sz, crop_sz, 3), dtype=np.uint8)
        x1 = cx - half; y1 = cy - half
        x2 = x1 + crop_sz; y2 = y1 + crop_sz
        
        src_x1 = max(0, x1); src_y1 = max(0, y1)
        src_x2 = min(curr_w, x2); src_y2 = min(curr_h, y2)
        
        dst_x1 = src_x1 - x1; dst_y1 = src_y1 - y1
        dst_x2 = dst_x1 + (src_x2 - src_x1); dst_y2 = dst_y1 + (src_y2 - src_y1)
        
        p_h = src_y2 - src_y1
        p_w = src_x2 - src_x1
        c_h = dst_y2 - dst_y1
        c_w = dst_x2 - dst_x1
        
        if p_h > 0 and p_w > 0 and p_h == c_h and p_w == c_w:
            patch_data = src_img[src_y1:src_y2, src_x1:src_x2]
            # 如果是单通道灰度图，转为3通道
            if len(patch_data.shape) == 2:
                patch_data = cv2.cvtColor(patch_data, cv2.COLOR_GRAY2BGR)
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = patch_data
            
        return canvas

    def save_dataset_sample(self, is_positive, auto_jump=True, explicit_candidate=None):
        if not self.candidates and not explicit_candidate: return
        
        # 0. 检查图片是否已加载 (防止删除文件后的残留操作)
        if not hasattr(self, 'img_a') or self.img_a is None:
            self.statusBar().showMessage("❌ 无法保存：图片数据未加载（可能文件已被删除）", 5000)
            return

        # 1. 确定要保存的目标
        cand = None
        row = -1
        
        if explicit_candidate:
            # 如果指定了候选体 (来自可疑列表)，尝试在当前 candidates 中匹配它
            target_x, target_y = explicit_candidate.get('x'), explicit_candidate.get('y')
            # Fix: 必须先确保载入了该目标所属的图片组，否则 self.candidates 还是上一张图的
            target_stem = explicit_candidate.get('stem')
            if target_stem and target_stem != self.current_group:
                 # 这不应该发生，因为 on_suspect_current_changed 应该已经切过去了
                 # 但为了健壮性，我们可以 log 一下
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
            if row < 0 or row >= len(self.candidates): return
            cand = self.candidates[row]

        # 标记当前候选体为"已保存"状态
        cand['saved'] = True
        cand['verdict'] = 'real' if is_positive else 'bogus'
        
        # 2. 准备保存路径
        import datetime
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
        
        # 3. 裁剪并保存 (后台线程)
        cx, cy = cand['x'], cand['y']
        try:
            p_a = self._crop_patch_common(self.img_a, cx, cy)
            p_b = self._crop_patch_common(self.img_b, cx, cy)
            p_c = self._crop_patch_common(self.img_c, cx, cy)
            combined = np.hstack([p_a, p_b, p_c])
            
            # 提交任务
            self.io_pool.submit(self._threaded_save_image, save_dir, fname, combined)
        except Exception as e:
            print(f"Prepare save failed: {e}")

        # 4. UI 反馈 (立即执行，不等待 IO)
        # 更新 cand_list 里的状态 (如果对应行存在)
        if row >= 0:
            item = self.cand_list.item(row)
            if item:
                text = item.text()
                if "[已" not in text:
                    suffix = " [已存真]" if is_positive else " [已存假]"
                    item.setText(text + suffix)
                item.setForeground(QColor(0, 100, 0) if is_positive else QColor(100, 0, 0))

        # 5. 更新数据库 (内存更新，异步落盘)
        DatabaseManager.update_record(self.current_group, self.candidates, crop_rect=self.crop_rect)
        if self.current_group in self.batch_results:
            self.batch_results[self.current_group]['candidates'] = self.candidates
            
        # === 核心逻辑修改：如果是在可疑列表模式，这里完全不负责跳转 ===
        # 跳转逻辑由 handle_suspect_action 里的 handle_suspect_skip 接管
        if explicit_candidate: return

        # 以下只针对普通模式 (非可疑列表模式)
        if not auto_jump: return
        if is_manual: return 

        if row < self.cand_list.count() - 1:
            self.cand_list.setCurrentRow(row + 1)
        else:
            QMessageBox.information(self, "完成", "本张图片所有候选体已处理完毕")

    def _threaded_save_image(self, save_dir, fname, img_data):
        try:
            counter = 1
            final_path = os.path.join(save_dir, fname)
            base_name, ext = os.path.splitext(fname)
            # 简单的重名检测 (注意：并发下理论上可能有竞态，但单人操作几率极低，且 counter 足够安全)
            while os.path.exists(final_path):
                final_path = os.path.join(save_dir, f"{base_name}_{counter}{ext}")
                counter += 1
            
            cv2.imwrite(final_path, img_data)
            print(f"Saved (Async): {final_path}")
        except Exception as e:
            print(f"Async Save Error: {e}")

    def skip_sample(self, auto_jump=True):
        # 跳过当前候选，直接选下一个
        # 逻辑与 save 保持一致：
        # 1. 手动目标 -> 不跳图，不跳行
        # 2. 自动目标 -> 下一个 or 跳图
        if not self.candidates: return
        row = self.cand_list.currentRow()
        
        if not auto_jump: return

        cand = self.candidates[row]
        if cand.get('manual', False):
             pass
        elif row == len(self.candidates) - 1:
            # 本图处理完毕，标记为已归档并跳转
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
        curr_idx = self.file_list.currentRow()
        for i in range(curr_idx + 1, self.file_list.count()):
            item = self.file_list.item(i)
            name = item.data(Qt.UserRole)
            if not name: name = item.text().split(" ")[0]
            rec = self.batch_results.get(name)
            if rec and rec.get("candidates") and rec.get("status") != "processed":
                self.file_list.setCurrentRow(i)
                return
        QMessageBox.information(self, "提示", "后续没有待处理的有目标图片了！")

    def on_candidate_selected(self, row):
        if row < 0 or row >= len(self.candidates): return
        cand = self.candidates[row]
        cx, cy = cand['x'], cand['y']
        
        # 保持当前视图模式，只更新圆圈
        self.view_context.draw_overlays(self.candidates, row, self.btn_hide_overlay.isChecked())

        label_text = f"Manual #{cand['id']}" if cand.get('manual', False) else f"Diff #{cand['id']}"
        
        # 使用通用裁剪函数
        p_a = self._crop_patch_common(self.img_a, cx, cy)
        p_b = self._crop_patch_common(self.img_b, cx, cy)
        p_c = self._crop_patch_common(self.img_c, cx, cy)
        
        # 放大显示用的图片 (例如放大到 200x200)
        disp_sz = 200
        disp_a = cv2.resize(p_a, (disp_sz, disp_sz), interpolation=cv2.INTER_NEAREST)
        disp_b = cv2.resize(p_b, (disp_sz, disp_sz), interpolation=cv2.INTER_NEAREST)
        disp_c = cv2.resize(p_c, (disp_sz, disp_sz), interpolation=cv2.INTER_NEAREST)
        
        # 在放大后的图上画圈和字
        center = disp_sz // 2
        radius = int(15 * (disp_sz / 80)) # 按比例放大半径
        
        cv2.circle(disp_a, (center, center), radius, (0, 255, 0), 2)
        cv2.putText(disp_a, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        
        cv2.circle(disp_b, (center, center), radius, (0, 255, 0), 2)
        cv2.putText(disp_b, "New", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        
        cv2.circle(disp_c, (center, center), radius, (0, 255, 0), 2)
        cv2.putText(disp_c, "Ref", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        combined = np.hstack([disp_a, disp_b, disp_c])
        if not combined.flags['C_CONTIGUOUS']: combined = np.ascontiguousarray(combined)
        h_c, w_c, ch = combined.shape
        qimg = QImage(combined.data.tobytes(), w_c, h_c, ch*w_c, QImage.Format_RGB888)
        self.lbl_triplet.setPixmap(QPixmap.fromImage(qimg))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = SCANN()
    win.show()
    sys.exit(app.exec_())
