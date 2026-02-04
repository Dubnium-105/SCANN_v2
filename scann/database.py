# -*- coding: utf-8 -*-
"""
SCANN 数据库管理模块
- AsyncDatabaseWriter: 异步数据库写入线程
- DatabaseManager: SQLite 数据库管理
"""

import os
import json
import sqlite3
import time
import queue
import threading
import traceback

from PyQt5.QtCore import QThread

# ================= 数据库文件路径 =================
DB_JSON_FILE = os.path.join(os.getcwd(), "SCANN_candidates.json")
DB_SQLITE_FILE = os.path.join(os.getcwd(), "SCANN_candidates.sqlite")

# 全局 DB Writer 实例
_db_writer = None


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
                if task is None:
                    break  # 退出信号
                
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
        self.wait()  # 必须等待线程彻底结束，确保最后的数据已写入

    def pending_count(self):
        return self.queue.qsize()

    def submit(self, func, *args):
        self.queue.put((func, args))


class DatabaseManager:
    """SQLite 数据库管理器"""
    
    _cache = {}
    _local = threading.local()
    _db_ready = False
    _writer_commit_every = 50
    _writer_commit_count = 0
    _writer_last_commit = 0.0

    @staticmethod
    def init_async():
        """初始化异步写入线程"""
        global _db_writer
        DatabaseManager._ensure_db_ready()
        if _db_writer is None:
            _db_writer = AsyncDatabaseWriter()

    @staticmethod
    def get_pending_count():
        """获取待写入任务数"""
        global _db_writer
        if _db_writer:
            return _db_writer.pending_count()
        return 0

    @staticmethod
    def stop_async():
        """停止异步写入线程"""
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
        """获取线程本地数据库连接"""
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
        """确保数据库表结构存在"""
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
        """确保数据库就绪，必要时从 JSON 迁移"""
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
        """从旧版 JSON 文件迁移数据"""
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
                    (stem, status, cand_json, count, has_ai, max_ai,
                     json.dumps(crop_rect, ensure_ascii=False) if crop_rect is not None else None,
                     params_hash, timestamp)
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
        """内部实现：更新记录"""
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
        """更新或插入记录"""
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
        """内部实现：标记状态"""
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
        """标记记录状态"""
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
        """加载所有记录的摘要信息"""
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
        """获取单条记录的完整信息"""
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
        """内部实现：删除记录"""
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
        """删除记录"""
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
        """内部实现：清空所有记录"""
        conn = DatabaseManager._get_conn()
        conn.execute("DELETE FROM images;")
        conn.commit()
        DatabaseManager._cache = {}

    @staticmethod
    def clear_all():
        """清空所有记录"""
        global _db_writer
        DatabaseManager._ensure_db_ready()
        if _db_writer and _db_writer.isRunning():
            _db_writer.submit(DatabaseManager._clear_all_impl)
        else:
            DatabaseManager._clear_all_impl()
