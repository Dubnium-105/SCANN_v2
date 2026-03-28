"""v2 FITS 标注存储层。

目标：
- 将标注信息从单一大 JSON 迁移到 SQLite（按样本增量写入）
- 保留对 legacy annotations.json 的读取兼容
- 对训练/导出提供统一读取文档（images 列表）
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from scann.core.annotation_models import AnnotationSample, BBox

logger = logging.getLogger(__name__)

DEFAULT_DB_FILE = "annotations.db"
DEFAULT_MANIFEST_FILE = "annotations.json"
MANIFEST_VERSION = "2.2"


@dataclass
class LoadedAnnotations:
    by_id: dict[str, dict]
    loaded_from_legacy_json: bool = False


class FitsAnnotationStorage:
    """FITS 标注 SQLite 存储（单机本地）。"""

    def __init__(
        self,
        dataset_root: Path,
        db_file: str = DEFAULT_DB_FILE,
        manifest_file: str = DEFAULT_MANIFEST_FILE,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.db_path = self.dataset_root / db_file
        self.manifest_path = self.dataset_root / manifest_file
        self._manifest_written = False

    def load_annotations(self) -> LoadedAnnotations:
        """加载标注。

        优先级：
        1) manifest 指向 sqlite
        2) 默认 annotations.db
        3) legacy annotations.json(images)
        """
        manifest = self._read_manifest_json()
        if manifest.get("storage") == "sqlite":
            db_file = str(manifest.get("db_file") or DEFAULT_DB_FILE)
            self.db_path = self.dataset_root / db_file
            if self.db_path.exists():
                return LoadedAnnotations(by_id=self._load_map_from_db())

        if self.db_path.exists():
            return LoadedAnnotations(by_id=self._load_map_from_db())

        legacy_images = manifest.get("images")
        if isinstance(legacy_images, list):
            by_id: dict[str, dict] = {}
            for image in legacy_images:
                if isinstance(image, dict) and image.get("id"):
                    by_id[str(image["id"])] = image
            return LoadedAnnotations(by_id=by_id, loaded_from_legacy_json=bool(by_id))

        return LoadedAnnotations(by_id={})

    def bulk_replace(self, samples: Iterable[AnnotationSample]) -> None:
        """全量替换（用于首次迁移）。"""
        conn = self._connect()
        self._ensure_schema(conn)
        with conn:
            conn.execute("DELETE FROM bboxes")
            conn.execute("DELETE FROM images")
            for sample in samples:
                self._upsert_sample_with_conn(conn, sample)
        conn.close()
        self._write_manifest_once()

    def upsert_sample(self, sample: AnnotationSample) -> None:
        """按样本增量写入，避免重写大文件。"""
        conn = self._connect()
        self._ensure_schema(conn)
        with conn:
            self._upsert_sample_with_conn(conn, sample)
        conn.close()
        self._write_manifest_once()

    def export_document(self) -> dict:
        """导出标准文档结构 {version, images}。"""
        if self.db_path.exists():
            images = list(self._load_map_from_db().values())
            return {"version": MANIFEST_VERSION, "images": images}

        manifest = self._read_manifest_json()
        if isinstance(manifest.get("images"), list):
            return {
                "version": str(manifest.get("version") or "2.0"),
                "images": manifest.get("images") or [],
            }
        return {"version": MANIFEST_VERSION, "images": []}

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS images (
                id TEXT PRIMARY KEY,
                file_name TEXT NOT NULL,
                label TEXT,
                detail_type TEXT,
                ai_suggestion TEXT,
                ai_confidence REAL,
                metadata_json TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        image_columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(images)").fetchall()
        }
        if "metadata_json" not in image_columns:
            conn.execute("ALTER TABLE images ADD COLUMN metadata_json TEXT")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bboxes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id TEXT NOT NULL,
                box_index INTEGER NOT NULL,
                x INTEGER NOT NULL,
                y INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                label TEXT,
                detail_type TEXT,
                confidence REAL,
                FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_bboxes_image_id ON bboxes(image_id)")

    @staticmethod
    def _to_image_dict(sample: AnnotationSample) -> dict:
        image: dict = {
            "id": sample.id,
            "file_name": sample.display_name,
        }
        if sample.label:
            image["label"] = sample.label
        if sample.detail_type:
            image["detail_type"] = sample.detail_type
        if sample.bboxes:
            image["annotations"] = [bbox.to_dict() for bbox in sample.bboxes]
        if sample.ai_suggestion is not None:
            image["ai_suggestion"] = sample.ai_suggestion
        if sample.ai_confidence is not None:
            image["ai_confidence"] = sample.ai_confidence
        if sample.metadata:
            image["metadata"] = sample.metadata
            paths = sample.metadata.get("paths")
            if isinstance(paths, dict):
                image["paths"] = paths
        return image

    @staticmethod
    def _has_payload(sample: AnnotationSample) -> bool:
        return bool(
            sample.label
            or sample.detail_type
            or sample.bboxes
            or sample.ai_suggestion is not None
            or sample.ai_confidence is not None
        )

    def _upsert_sample_with_conn(self, conn: sqlite3.Connection, sample: AnnotationSample) -> None:
        if not self._has_payload(sample):
            conn.execute("DELETE FROM images WHERE id = ?", (sample.id,))
            return

        updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        conn.execute(
            """
            INSERT INTO images (id, file_name, label, detail_type, ai_suggestion, ai_confidence, metadata_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                file_name=excluded.file_name,
                label=excluded.label,
                detail_type=excluded.detail_type,
                ai_suggestion=excluded.ai_suggestion,
                ai_confidence=excluded.ai_confidence,
                metadata_json=excluded.metadata_json,
                updated_at=excluded.updated_at
            """,
            (
                sample.id,
                sample.display_name,
                sample.label,
                sample.detail_type,
                sample.ai_suggestion,
                sample.ai_confidence,
                json.dumps(sample.metadata, ensure_ascii=False) if sample.metadata else None,
                updated_at,
            ),
        )
        conn.execute("DELETE FROM bboxes WHERE image_id = ?", (sample.id,))
        for idx, bbox in enumerate(sample.bboxes):
            conn.execute(
                """
                INSERT INTO bboxes (image_id, box_index, x, y, width, height, label, detail_type, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sample.id,
                    idx,
                    int(bbox.x),
                    int(bbox.y),
                    int(bbox.width),
                    int(bbox.height),
                    bbox.label,
                    bbox.detail_type,
                    float(bbox.confidence),
                ),
            )

    def _load_map_from_db(self) -> dict[str, dict[str, object]]:
        conn = self._connect()
        self._ensure_schema(conn)
        image_rows = conn.execute(
            "SELECT id, file_name, label, detail_type, ai_suggestion, ai_confidence, metadata_json FROM images"
        ).fetchall()
        bbox_rows = conn.execute(
            """
            SELECT image_id, box_index, x, y, width, height, label, detail_type, confidence
            FROM bboxes
            ORDER BY image_id, box_index ASC
            """
        ).fetchall()
        conn.close()

        by_id: dict[str, dict[str, object]] = {}
        for row in image_rows:
            image_id = str(row["id"])
            image: dict[str, object] = {
                "id": image_id,
                "file_name": str(row["file_name"] or row["id"]),
            }
            if row["label"]:
                image["label"] = str(row["label"])
            if row["detail_type"]:
                image["detail_type"] = str(row["detail_type"])
            if row["ai_suggestion"] is not None:
                image["ai_suggestion"] = str(row["ai_suggestion"])
            if row["ai_confidence"] is not None:
                image["ai_confidence"] = float(row["ai_confidence"])
            metadata_raw = row["metadata_json"]
            if metadata_raw:
                try:
                    metadata = json.loads(str(metadata_raw))
                except Exception:
                    metadata = {}
                if isinstance(metadata, dict) and metadata:
                    image["metadata"] = metadata
                    paths = metadata.get("paths")
                    if isinstance(paths, dict):
                        image["paths"] = paths
            image["annotations"] = []
            by_id[image_id] = image

        for row in bbox_rows:
            image_id = str(row["image_id"])
            image_record = by_id.get(image_id)
            if image_record is None:
                continue
            bbox = BBox(
                x=int(row["x"]),
                y=int(row["y"]),
                width=int(row["width"]),
                height=int(row["height"]),
                label=str(row["label"] or "real"),
                confidence=float(row["confidence"] if row["confidence"] is not None else 1.0),
                detail_type=str(row["detail_type"]) if row["detail_type"] is not None else None,
            )
            annotations = image_record.get("annotations")
            if not isinstance(annotations, list):
                annotations = []
                image_record["annotations"] = annotations
            annotations.append(bbox.to_dict())

        for image in by_id.values():
            annotations = image.get("annotations")
            if isinstance(annotations, list) and not annotations:
                image.pop("annotations", None)

        return by_id

    def _read_manifest_json(self) -> dict:
        if not self.manifest_path.exists():
            return {}
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError as exc:
            logger.warning("annotations.json 解析失败: %s", exc)
            return {}

    def _write_manifest_once(self) -> None:
        if self._manifest_written:
            return
        manifest = {
            "version": MANIFEST_VERSION,
            "storage": "sqlite",
            "db_file": self.db_path.name,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._manifest_written = True


def load_v2_annotation_document(dataset_root: Path) -> dict:
    """统一读取 v2 标注文档（兼容 SQLite + legacy JSON）。"""
    storage = FitsAnnotationStorage(dataset_root)
    return storage.export_document()
