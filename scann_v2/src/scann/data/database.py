"""数据库管理模块

职责:
- 候选体数据库 (SQLite)
- 异步写入
- 配对信息管理
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from scann.core.models import Candidate, CandidateFeatures, TargetVerdict


class CandidateDatabase:
    """候选体数据库 (线程安全)"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            return conn
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        self._local.conn = conn
        return conn

    def _ensure_schema(self) -> None:
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair_name TEXT NOT NULL,
                x INTEGER NOT NULL,
                y INTEGER NOT NULL,
                ai_score REAL DEFAULT 0.0,
                verdict TEXT DEFAULT 'unknown',
                is_manual INTEGER DEFAULT 0,
                is_known INTEGER DEFAULT 0,
                known_id TEXT DEFAULT '',
                features_json TEXT,
                timestamp REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS image_pairs (
                name TEXT PRIMARY KEY,
                status TEXT DEFAULT 'unseen',
                candidate_count INTEGER DEFAULT 0,
                max_ai_score REAL DEFAULT 0.0,
                align_dx REAL DEFAULT 0.0,
                align_dy REAL DEFAULT 0.0,
                timestamp REAL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cand_pair ON candidates(pair_name);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cand_score ON candidates(ai_score DESC);")
        conn.commit()

    def save_candidates(
        self,
        pair_name: str,
        candidates: List[Candidate],
    ) -> None:
        """保存一组候选体"""
        conn = self._get_conn()
        now = time.time()

        # 清除旧数据
        conn.execute("DELETE FROM candidates WHERE pair_name = ?", (pair_name,))

        # 写入新数据
        for c in candidates:
            features_json = json.dumps({
                "peak": c.features.peak,
                "mean": c.features.mean,
                "sharpness": c.features.sharpness,
                "contrast": c.features.contrast,
                "area": c.features.area,
                "rise": c.features.rise,
            })
            conn.execute(
                "INSERT INTO candidates (pair_name, x, y, ai_score, verdict, "
                "is_manual, is_known, known_id, features_json, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    pair_name, c.x, c.y, c.ai_score,
                    c.verdict.value, int(c.is_manual), int(c.is_known),
                    c.known_id, features_json, now,
                ),
            )

        # 更新 image_pairs 摘要
        max_score = max((c.ai_score for c in candidates), default=0.0)
        conn.execute(
            "INSERT OR REPLACE INTO image_pairs (name, candidate_count, max_ai_score, timestamp) "
            "VALUES (?, ?, ?, ?)",
            (pair_name, len(candidates), max_score, now),
        )
        conn.commit()

    def get_candidates(self, pair_name: str) -> List[Candidate]:
        """获取一组候选体"""
        conn = self._get_conn()
        cur = conn.execute(
            "SELECT * FROM candidates WHERE pair_name = ? ORDER BY ai_score DESC",
            (pair_name,),
        )
        results = []
        for row in cur.fetchall():
            features = CandidateFeatures()
            if row["features_json"]:
                try:
                    f = json.loads(row["features_json"])
                    features = CandidateFeatures(**f)
                except (json.JSONDecodeError, TypeError):
                    pass

            results.append(Candidate(
                x=row["x"],
                y=row["y"],
                features=features,
                ai_score=row["ai_score"] or 0.0,
                verdict=TargetVerdict(row["verdict"] or "unknown"),
                is_manual=bool(row["is_manual"]),
                is_known=bool(row["is_known"]),
                known_id=row["known_id"] or "",
            ))
        return results

    def get_all_suspects(
        self,
        min_score: float = 0.0,
        limit: int = 500,
    ) -> List[dict]:
        """获取所有可疑目标 (跨图像)"""
        conn = self._get_conn()
        cur = conn.execute(
            "SELECT pair_name, x, y, ai_score, verdict FROM candidates "
            "WHERE ai_score >= ? AND is_manual = 0 "
            "ORDER BY ai_score DESC LIMIT ?",
            (min_score, limit),
        )
        return [dict(row) for row in cur.fetchall()]

    def update_verdict(
        self,
        pair_name: str,
        x: int,
        y: int,
        verdict: TargetVerdict,
    ) -> None:
        """更新候选体判决"""
        conn = self._get_conn()
        conn.execute(
            "UPDATE candidates SET verdict = ? WHERE pair_name = ? AND x = ? AND y = ?",
            (verdict.value, pair_name, x, y),
        )
        conn.commit()

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn:
            conn.commit()
            conn.close()
            self._local.conn = None
