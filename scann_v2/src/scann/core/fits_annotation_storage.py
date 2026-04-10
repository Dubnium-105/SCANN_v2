"""v2 FITS 标注存储层。

当前版本以数据集级 SQLite (`scann_dataset.db`) 为主存储：
- 当前标注状态写入 `tasks` + `task_annotation_boxes_current`
- 保留对 legacy `annotations.json` / `annotations.db` 的读取兼容
- 导出时统一返回 `{version, images}` 结构，供训练和旧链路复用
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from scann.core.annotation_models import AnnotationSample, BBox
from scann.core.dataset_storage import DEFAULT_DATASET_DB_FILE, DatasetStorage

logger = logging.getLogger(__name__)

DEFAULT_DB_FILE = DEFAULT_DATASET_DB_FILE
DEFAULT_MANIFEST_FILE = "annotations.json"
DEFAULT_LEGACY_DB_FILE = "annotations.db"
MANIFEST_VERSION = "2.3"


@dataclass
class LoadedAnnotations:
    by_id: dict[str, dict]
    loaded_from_legacy_json: bool = False


class FitsAnnotationStorage:
    """FITS 标注数据库封装。"""

    def __init__(
        self,
        dataset_root: Path,
        db_file: str = DEFAULT_DB_FILE,
        manifest_file: str = DEFAULT_MANIFEST_FILE,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.manifest_path = self.dataset_root / manifest_file
        self.legacy_db_path = self.dataset_root / DEFAULT_LEGACY_DB_FILE
        self._manifest_written = False
        self._dataset_storage = DatasetStorage(self.dataset_root, db_file=db_file)
        self._dataset_storage.ensure_schema()

    def load_annotations(self) -> LoadedAnnotations:
        current = self._dataset_storage.list_current_annotations()
        if current:
            return LoadedAnnotations(by_id=current)

        manifest = self._read_manifest_json()

        if self.legacy_db_path.exists():
            return LoadedAnnotations(by_id=self._load_map_from_legacy_db())

        legacy_images = manifest.get("images")
        if isinstance(legacy_images, list):
            by_id: dict[str, dict] = {}
            for image in legacy_images:
                if isinstance(image, dict) and image.get("id"):
                    by_id[str(image["id"])] = image
            return LoadedAnnotations(by_id=by_id, loaded_from_legacy_json=bool(by_id))

        return LoadedAnnotations(by_id={})

    def bulk_replace(self, samples: Iterable[AnnotationSample]) -> None:
        for sample in samples:
            self._upsert_current_annotation(sample)
        self._write_manifest_once()

    def upsert_sample(self, sample: AnnotationSample) -> None:
        self._upsert_current_annotation(sample)
        self._write_manifest_once()

    def export_document(self) -> dict[str, Any]:
        current = self._dataset_storage.list_current_annotations()
        if current:
            images = list(current.values())
            return {"version": MANIFEST_VERSION, "storage": "dataset_db", "images": images}

        if self.legacy_db_path.exists():
            images = list(self._load_map_from_legacy_db().values())
            return {"version": "2.2", "storage": "legacy_sqlite", "images": images}

        manifest = self._read_manifest_json()
        if isinstance(manifest.get("images"), list):
            return {
                "version": str(manifest.get("version") or "2.0"),
                "storage": str(manifest.get("storage") or "legacy_json"),
                "images": manifest.get("images") or [],
            }
        return {"version": MANIFEST_VERSION, "storage": "dataset_db", "images": []}

    def _upsert_current_annotation(self, sample: AnnotationSample) -> None:
        paths = self._extract_paths(sample)
        source_view = self._resolve_source_view(sample, paths)
        self._dataset_storage.upsert_current_annotation(
            task_id=sample.id,
            source_view=source_view,
            label=sample.label,
            detail_type=sample.detail_type,
            ai_suggestion=sample.ai_suggestion,
            ai_confidence=sample.ai_confidence,
            annotations=[bbox.to_dict() for bbox in sample.bboxes],
            annotation_origin="local",
        )

    @staticmethod
    def _extract_paths(sample: AnnotationSample) -> dict[str, str]:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        paths = metadata.get("paths")
        if not isinstance(paths, dict):
            return {}
        return {
            "new": str(paths.get("new") or "").strip(),
            "old": str(paths.get("old") or "").strip(),
            "new_marked": str(paths.get("new_marked") or "").strip(),
        }

    @staticmethod
    def _resolve_source_view(sample: AnnotationSample, paths: dict[str, str]) -> str | None:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        source_view = str(metadata.get("source_view") or "").strip()
        if source_view in {"old", "new", "new_marked"}:
            return source_view
        for candidate in ("new", "old", "new_marked"):
            if paths.get(candidate):
                return candidate
        return None

    def _load_map_from_legacy_db(self) -> dict[str, dict[str, Any]]:
        connection = sqlite3.connect(str(self.legacy_db_path), timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            image_rows = connection.execute(
                "SELECT id, file_name, label, detail_type, ai_suggestion, ai_confidence, metadata_json FROM images"
            ).fetchall()
            bbox_rows = connection.execute(
                """
                SELECT image_id, box_index, x, y, width, height, label, detail_type, confidence
                FROM bboxes
                ORDER BY image_id, box_index ASC
                """
            ).fetchall()
        finally:
            connection.close()

        by_id: dict[str, dict[str, Any]] = {}
        for row in image_rows:
            image_id = str(row["id"])
            image: dict[str, Any] = {
                "id": image_id,
                "file_name": str(row["file_name"] or row["id"]),
            }
            if row["label"] is not None:
                image["label"] = str(row["label"])
            if row["detail_type"] is not None:
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
                label=str(row["label"]) if row["label"] is not None else None,
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

    def _read_manifest_json(self) -> dict[str, Any]:
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
            "storage": "dataset_db",
            "db_file": self._dataset_storage.db_path.name,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._manifest_written = True


def load_v2_annotation_document(dataset_root: Path) -> dict[str, Any]:
    """统一读取 v2 标注文档，优先走数据集数据库。"""
    storage = FitsAnnotationStorage(dataset_root)
    return storage.export_document()
