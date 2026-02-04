# -*- coding: utf-8 -*-
"""
SCANN 下载引擎模块
- LinkedDownloader: 三联图联动下载引擎
- DBDownloadWindow: 数据库级联下载窗口
"""

import sys
import os
import re
import time
import sqlite3
import threading
import collections
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QAbstractItemView, QProgressBar,
    QMessageBox, QFileDialog, QStyle, QApplication
)
from PyQt5.QtCore import Qt, QObject, pyqtSignal
from PyQt5.QtGui import QFont, QColor

from .config import ConfigManager


class LinkedDownloader(QObject):
    """三联图联动下载引擎"""
    
    download_progress = pyqtSignal(str, int, int)  # name, current, total
    download_finished = pyqtSignal(str, bool, str)  # name, success, path_or_error
    all_finished = pyqtSignal(int, int)  # success_count, fail_count
    status_msg = pyqtSignal(str)

    BASE_JPG_URL = "https://nadc.china-vo.org/psp/hmt/PSP-HMT-DATA/output/"
    BASE_FITS_URL = "https://nadc.china-vo.org/psp/hmt/PSP-HMT-DATA/data/"
    
    def __init__(self):
        super().__init__()
        # --- 动态获取路径 ---
        if getattr(sys, 'frozen', False):
            self._SCRIPT_DIR = os.path.dirname(sys.executable)
        else:
            self._SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
            
        self.APPDATA_DIR = os.path.join(self._SCRIPT_DIR, "SCANN_Data")
        self.DB_FILE = os.path.join(self.APPDATA_DIR, "psp_linkage.db")
        print(f"📦 数据库位置: {self.DB_FILE}") 
        
        if not os.path.exists(self.APPDATA_DIR):
            os.makedirs(self.APPDATA_DIR)
        self.session = self._create_session()
        self._init_db()
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.active_tasks = {}  # jpg_stem -> future
        self.cancel_requested = set()  # jpg_stem set for active cancellation
        self.stats_lock = threading.Lock()
        self.session_success = 0
        self.session_fail = 0

    def _create_session(self):
        """创建带重试机制的 HTTP Session"""
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
        """初始化联动数据库"""
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
        """清理文件名中的非法字符"""
        return re.sub(r'[\\/:*?"<>|]', '_', name)

    @staticmethod
    def map_jpg_to_fits(jpg_url):
        """将 JPG URL 映射到对应的 FITS URL"""
        if "/output/" not in jpg_url:
            return None
        # 1. 替换目录
        url = jpg_url.replace("/output/", "/data/")
        # 2. 移除 JPG 后缀
        if url.lower().endswith(".jpg"):
            url = url[:-4]
        elif url.lower().endswith(".jpeg"):
            url = url[:-5]
        
        # 3. 将分身后缀还原为标准 FITS 后缀
        url = re.sub(r'\.fts[1-9][abc]?$', '.fts', url, flags=re.I)
        return url

    def _normalize_stem(self, stem):
        """规范化 stem 名称"""
        if not stem:
            return ""
        s = stem.replace(" ", "").lower().strip()
        return re.sub(r'\.fts[1-9][abc]?$', '', s)

    def get_linkage(self, jpg_stem):
        """获取联动信息"""
        if not jpg_stem:
            return None
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
        """更新联动信息"""
        if not jpg_stem:
            return
        search_stem = self._normalize_stem(jpg_stem)
        cols = ", ".join([f"{k} = ?" for k in kwargs.keys()])
        vals = list(kwargs.values()) + [search_stem]
        with sqlite3.connect(self.DB_FILE) as conn:
            conn.execute(f"UPDATE linkage SET {cols} WHERE jpg_stem = ? COLLATE NOCASE", vals)

    def add_linkage(self, jpg_stem, local_jpg_path, remote_fits_url):
        """添加联动信息"""
        if not jpg_stem:
            return
        save_stem = self._normalize_stem(jpg_stem)
        with sqlite3.connect(self.DB_FILE) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO linkage (jpg_stem, local_jpg_path, remote_fits_url, timestamp)
                VALUES (?, ?, ?, ?)
            """, (save_stem, local_jpg_path, remote_fits_url, time.time()))

    def batch_add_linkage(self, items):
        """批量添加联动信息"""
        if not items:
            return
        now = time.time()
        
        unique_items = {}
        for stem, path, url in items:
            clean_id = self._normalize_stem(stem)
            if not clean_id:
                continue
            unique_items[clean_id] = (path, url)
        
        print(f"📥 [Store] 正在同步 {len(unique_items)} 组 FITS 联动数据...")
        
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
        """清空所有联动记录"""
        try:
            with sqlite3.connect(self.DB_FILE) as conn:
                conn.execute("DELETE FROM linkage;")
            self.status_msg.emit("🧹 联动数据库已清空")
            return True
        except Exception as e:
            self.status_msg.emit(f"❌ 清空失败: {e}")
            return False

    def download_task(self, url, save_path, jpg_stem):
        """下载任务执行函数"""
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
                if initial_pos < total_size:
                    mode = 'ab'
                elif initial_pos == total_size:
                    self.download_finished.emit(jpg_stem, True, save_path)
                    is_fits = any(url.lower().endswith(ext) or f"{ext}?" in url.lower() 
                                  for ext in ['.fts', '.fts2', '.fits'])
                    if is_fits:
                        self.update_linkage(jpg_stem, status='downloaded', local_fits_path=save_path)
                    else:
                        self.update_linkage(jpg_stem, local_jpg_path=save_path)
                    return
                else:
                    initial_pos = 0

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
            is_fits = any(url.lower().endswith(ext) or f"{ext}?" in url.lower() 
                          for ext in ['.fts', '.fts2', '.fits'])
            if is_fits:
                self.update_linkage(jpg_stem, status='downloaded', local_fits_path=save_path)
            else:
                self.update_linkage(jpg_stem, local_jpg_path=save_path)
                
            self.download_finished.emit(jpg_stem, True, save_path)
            self.status_msg.emit(f"✅ 下载完成: {filename}")
            with self.stats_lock:
                self.session_success += 1
        except Exception as e:
            self.download_finished.emit(jpg_stem, False, str(e))
            self.status_msg.emit(f"❌ 下载失败: {filename} ({str(e)})")
            with self.stats_lock:
                self.session_fail += 1
        finally:
            self.active_tasks.pop(jpg_stem, None)
            if not self.active_tasks:
                self.all_finished.emit(self.session_success, self.session_fail)
                with self.stats_lock:
                    self.session_success = 0
                    self.session_fail = 0

    def submit_download(self, jpg_stem, remote_url, local_save_dir, override_filename=None):
        """提交下载任务"""
        if jpg_stem in self.active_tasks:
            return
        
        url_lower = remote_url.lower()
        is_fits = any(url_lower.endswith(ext) or f"{ext}?" in url_lower 
                      for ext in ['.fts', '.fts2', '.fits'])
        
        if override_filename:
            filename = self.clean_filename(override_filename)
        else:
            raw_filename = os.path.basename(remote_url)
            unquoted_filename = urllib.parse.unquote(raw_filename)
            filename = self.clean_filename(unquoted_filename)
        
        if is_fits and not filename.lower().endswith(('.fts', '.fts2', '.fits')):
            filename += ".fts"
            
        save_path = os.path.join(local_save_dir, filename)
        future = self.executor.submit(self.download_task, remote_url, save_path, jpg_stem)
        self.active_tasks[jpg_stem] = future


class DBDownloadWindow(QDialog):
    """数据库级联下载窗口"""
    
    sig_load_done = pyqtSignal(object)
    sig_scan_done = pyqtSignal(object)
    sig_scan_status = pyqtSignal(int, int)  # found, skipped

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
        """初始化界面"""
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
        """确认清空联动库"""
        reply = QMessageBox.question(
            self, "确认清空", 
            "确定要清空本地 FITS 联动库吗？\n\n这不会删除已下载的文件，但会导致右键无法直接下载 FITS，直到您再次扫描服务器。",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            if self.downloader.clear_all_linkage():
                QMessageBox.information(self, "成功", "联动库已清空。")

    def load_directory(self, url):
        """加载目录内容"""
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
                    if href in ("../", "/"):
                        continue
                    full_url = urllib.parse.urljoin(url, href)
                    is_dir = href.endswith('/')
                    items.append((text, full_url, is_dir))
                self.sig_load_done.emit(items)
            except Exception as e:
                self.sig_load_done.emit(str(e))

        threading.Thread(target=fetch, daemon=True).start()

    def _on_load_done(self, result):
        """目录加载完成回调"""
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
        """双击进入目录"""
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
        """返回上级目录"""
        if self.history_stack:
            self.current_url = self.history_stack.pop()
            self.load_directory(self.current_url)

    def _on_scan_status(self, found, skipped):
        """扫描状态更新"""
        if self.parent() and hasattr(self.parent(), 'statusBar'):
            self.parent().statusBar().showMessage(f"🔍 正在扫描... 已发现: {found} | 匹配本地: {skipped}")

    def start_batch_download(self):
        """开始批量下载"""
        selected_items = self.list_widget.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "提示", "请先选择要下载的项目")
            return

        # 1. 确定 JPG 保存目录
        save_dir = self.parent().cfg.get('jpg_download_dir') if self.parent() else None
        if not save_dir or not os.path.exists(save_dir):
            save_dir = QFileDialog.getExistingDirectory(self, "选择 JPG 保存目录")
            if not save_dir:
                return
            if self.parent():
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
                existing_files = set()
                try:
                    for fn in os.listdir(save_dir):
                        existing_files.add(fn.lower())
                except Exception:
                    pass

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
                        
                        jpg_stem = filename
                        for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
                            if jpg_stem.endswith(ext):
                                jpg_stem = jpg_stem[:-len(ext)]
                                break
                                
                        all_tasks.append({"stem": jpg_stem, "url": jpg_url, "fits": fits_url, "filename": filename})
                
                if scan_jobs:
                    results, skipped_links = self._parallel_scan_v63_style(scan_jobs, existing_files, max_depth=4)
                    all_tasks.extend(results)
                else:
                    skipped_links = []
                
                if self.stop_scan_flag:
                    self.sig_scan_done.emit("扫描已中止")
                else:
                    self.sig_scan_done.emit((all_tasks, skipped_links, save_dir))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.sig_scan_done.emit(str(e))

        threading.Thread(target=scan_and_submit, daemon=True).start()

    def _parallel_scan_v63_style(self, scan_jobs, existing_files, max_depth=4):
        """并行扫描引擎"""
        all_results = []
        skipped_linkages = []
        visited_urls = set()
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
                    
                    matches = re.findall(r'href="([^.?/][^"?/]*(?:/|\.jpg))"', resp.text, re.I)
                    
                    new_jobs = []
                    local_files = []
                    local_skipped = []
                    
                    for href in matches:
                        if href.startswith('.') or href.startswith('/') or '?' in href:
                            continue
                            
                        full_url = urllib.parse.urljoin(url, href)
                        
                        if href.endswith('/'):
                            if depth < max_depth:
                                clean_name = urllib.parse.unquote(href.rstrip('/'))
                                new_prefix = f"{prefix}_{clean_name}"
                                new_jobs.append((full_url, new_prefix, depth + 1))
                        elif href.lower().endswith('.jpg'):
                            save_name = urllib.parse.unquote(href)
                            clean_save_name = self.downloader.clean_filename(save_name)
                            
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
        """扫描完成回调"""
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

        if self.parent() and hasattr(self.parent(), 'statusBar'):
            self.parent().statusBar().showMessage("🔍 正在整理扫描结果...")
        QApplication.processEvents()

        # 补全暂存的联动信息
        linkage_items = []
        for stem, filename, fits_url in skipped_links:
            local_path = os.path.join(save_dir, filename)
            linkage_items.append((stem, local_path, fits_url))
        self.downloader.batch_add_linkage(linkage_items)

        if not tasks:
            QMessageBox.information(self, "提示", f"扫描到 {len(skipped_links)} 个文件，全部已存在。")
            return

        # 确认提示
        msg = f"扫描完成！共发现 {len(tasks) + len(skipped_links)} 个文件。\n\n准备下载: {len(tasks)} 个\n已存在跳过: {len(skipped_links)} 个"
        if len(tasks) > 100:
            msg = "⚠️ ⚠️ ⚠️ 任务量较大！\n\n" + msg
        
        ok = QMessageBox.question(self, "确认下载任务", msg, QMessageBox.Yes | QMessageBox.No)
        if ok == QMessageBox.No:
            return

        # 提交下载任务
        for task in tasks:
            self.downloader.add_linkage(task['stem'], "", task['fits'])
            self.downloader.submit_download(task['stem'], task['url'], save_dir, override_filename=task['filename'])
        
        if self.parent() and hasattr(self.parent(), 'statusBar'):
            self.parent().statusBar().showMessage(f"🚀 已添加 {len(tasks)} 个任务到下载队列")
