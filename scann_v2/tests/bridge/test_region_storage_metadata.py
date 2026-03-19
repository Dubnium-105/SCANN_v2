"""测试 C06：region 入库审计元数据记录。"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path


def _build_config(bridge_module, dataset_root: Path):
    bridge_module.CONFIG = bridge_module.BridgeConfig(
        dataset_root=dataset_root,
        sqlite_path=dataset_root / "annotations.db",
        new_dir=dataset_root / "new",
        old_dir=dataset_root / "old",
        new_marked_dir=dataset_root / "new_marked",
        preview_cache_dir=dataset_root / ".preview_cache",
        manifest_path=dataset_root / "annotations.json",
        label_studio_url="http://labelstudio:8080",
        label_studio_token="token-1234567890",
        label_studio_project_id=1,
        public_data_base_url="http://127.0.0.1:3001/dataset",
        js9_base_url="http://127.0.0.1:3001",
        viewer_base_url="http://127.0.0.1:3001",
        enable_preview_render=False,
    )


def test_region_storage_metadata_logged_for_js9_regions_result(tmp_path: Path, bridge_module):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True)
    _build_config(bridge_module, dataset_root)

    js9_regions_json = json.dumps(
        [
            {
                "shape": "box",
                "x": 120,
                "y": 140,
                "width": 40,
                "height": 50,
                "label": "real",
                "detail_type": "asteroid",
            }
        ],
        ensure_ascii=False,
    )

    payload = {
        "annotations": [
            {
                "task": {
                    "data": {
                        "sample_id": "sample_js9",
                        "file_name": "sample_js9.fts",
                        "annotation_mode": "js9_region_primary",
                    }
                },
                "result": [
                    {
                        "from_name": "js9_regions_json",
                        "type": "textarea",
                        "value": {"text": [js9_regions_json]},
                    },
                    {
                        "type": "rectanglelabels",
                        "value": {
                            "x": 10.0,
                            "y": 10.0,
                            "width": 20.0,
                            "height": 20.0,
                            "rectanglelabels": ["noise"],
                        },
                        "original_width": 1000,
                        "original_height": 800,
                    },
                ],
            }
        ]
    }

    class FakeRequest:
        async def json(self):
            return payload

    resp = asyncio.run(bridge_module.labelstudio_webhook(FakeRequest()))
    assert resp.updated_samples == 1

    audit_path = dataset_root / ".audit" / bridge_module.REGION_STORAGE_AUDIT_LOG_NAME
    assert audit_path.exists()

    lines = [line for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])

    assert record["sample_id"] == "sample_js9"
    assert record["region_source"] == "annotation_result.js9_regions_json"
    assert record["region_schema_version"] == bridge_module.JS9_REGION_SCHEMA_VERSION
    assert record["audit_schema_version"] == bridge_module.REGION_STORAGE_AUDIT_SCHEMA_VERSION
    assert record["annotation_mode"] == "js9_region_primary"
    assert record["js9_region_count"] == 1
    assert record["bbox_count"] == 1


def test_region_storage_metadata_logged_for_rectangle_fallback_and_schema_unchanged(tmp_path: Path, bridge_module):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True)
    _build_config(bridge_module, dataset_root)

    payload = {
        "annotations": [
            {
                "task": {
                    "data": {
                        "sample_id": "sample_fallback",
                        "file_name": "sample_fallback.fts",
                    }
                },
                "result": [
                    {
                        "type": "rectanglelabels",
                        "value": {
                            "x": 20.0,
                            "y": 30.0,
                            "width": 10.0,
                            "height": 15.0,
                            "rectanglelabels": ["supernova"],
                        },
                        "original_width": 1000,
                        "original_height": 1000,
                    }
                ],
            }
        ]
    }

    class FakeRequest:
        async def json(self):
            return payload

    resp = asyncio.run(bridge_module.labelstudio_webhook(FakeRequest()))
    assert resp.updated_samples == 1

    audit_path = dataset_root / ".audit" / bridge_module.REGION_STORAGE_AUDIT_LOG_NAME
    lines = [line for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])

    assert record["sample_id"] == "sample_fallback"
    assert record["region_source"] == "rectanglelabels_fallback"
    assert record["region_schema_version"] == bridge_module.RECTANGLELABELS_SCHEMA_VERSION
    assert record["annotation_mode"] == "unknown"
    assert record["js9_region_count"] == 0
    assert record["bbox_count"] == 1

    conn = sqlite3.connect(bridge_module.CONFIG.sqlite_path)
    try:
        images_cols = [row[1] for row in conn.execute("PRAGMA table_info(images)").fetchall()]
        bboxes_cols = [row[1] for row in conn.execute("PRAGMA table_info(bboxes)").fetchall()]
    finally:
        conn.close()

    assert images_cols == [
        "id",
        "file_name",
        "label",
        "detail_type",
        "ai_suggestion",
        "ai_confidence",
        "updated_at",
    ]
    assert bboxes_cols == [
        "id",
        "image_id",
        "box_index",
        "x",
        "y",
        "width",
        "height",
        "label",
        "detail_type",
        "confidence",
    ]
