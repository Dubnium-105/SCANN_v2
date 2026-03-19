"""测试 C07：pull -> 标注 -> webhook 的 JS9 Region 主链路回归。"""
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


def test_pull_webhook_region_primary_e2e(tmp_path: Path, monkeypatch, bridge_module):
    dataset_root = tmp_path / "dataset"
    new_dir = dataset_root / "new"
    old_dir = dataset_root / "old"
    marked_dir = dataset_root / "new_marked"
    new_dir.mkdir(parents=True)
    old_dir.mkdir(parents=True)
    marked_dir.mkdir(parents=True)

    (new_dir / "SAMPLE E2E.fts").write_bytes(b"new")
    (old_dir / "SAMPLE E2E.fts").write_bytes(b"old")
    (marked_dir / "SAMPLE E2E.fts").write_bytes(b"marked")

    _build_config(bridge_module, dataset_root)

    posted: dict[str, object] = {}

    class _Resp:
        status_code = 200
        content = b"{}"
        text = "{}"

        @staticmethod
        def json():
            return {"task_count": 1}

    def _fake_post(url, headers, json, timeout):
        posted["url"] = url
        posted["headers"] = headers
        posted["json"] = json
        posted["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr(bridge_module.requests, "post", _fake_post)

    # Step 1: pull 任务并捕获下发到 Label Studio 的 task data
    pull_resp = bridge_module.pull_tasks(bridge_module.PullRequest(import_to_label_studio=True, limit=1))
    assert pull_resp.scanned_pairs == 1
    assert pull_resp.tasks_built == 1
    assert pull_resp.tasks_imported == 1

    imported_tasks = posted["json"]
    assert isinstance(imported_tasks, list)
    assert len(imported_tasks) == 1

    task_data = imported_tasks[0]["data"]
    assert task_data["annotation_mode"] == "js9_region_primary"
    assert task_data["js9_regions_json"] is None

    # Step 2: 模拟标注员提交 js9_regions_json 结果
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
                "confidence": 0.93,
            }
        ],
        ensure_ascii=False,
    )

    payload = {
        "annotations": [
            {
                "task": {"data": task_data},
                "result": [
                    {
                        "from_name": "js9_regions_json",
                        "type": "textarea",
                        "value": {"text": [js9_regions_json]},
                        "original_width": 1000,
                        "original_height": 800,
                    }
                ],
            }
        ]
    }

    class FakeRequest:
        async def json(self):
            return payload

    # Step 3: webhook 入库
    webhook_resp = asyncio.run(bridge_module.labelstudio_webhook(FakeRequest()))
    assert webhook_resp.updated_samples == 1

    # Step 4: 验证 sqlite 回写
    conn = sqlite3.connect(bridge_module.CONFIG.sqlite_path)
    try:
        image_row = conn.execute(
            "SELECT id, file_name, label, detail_type FROM images WHERE id = ?",
            (task_data["sample_id"],),
        ).fetchone()
        bbox_row = conn.execute(
            "SELECT image_id, box_index, x, y, width, height, label, detail_type, confidence "
            "FROM bboxes WHERE image_id = ? ORDER BY box_index",
            (task_data["sample_id"],),
        ).fetchone()
    finally:
        conn.close()

    assert image_row is not None
    assert image_row[0] == task_data["sample_id"]
    assert image_row[1] == task_data["file_name"]
    assert image_row[2] == "real"
    assert image_row[3] == "asteroid"

    assert bbox_row is not None
    assert bbox_row[0] == task_data["sample_id"]
    assert bbox_row[1] == 0
    assert bbox_row[2] == 120
    assert bbox_row[3] == 140
    assert bbox_row[4] == 40
    assert bbox_row[5] == 50
    assert bbox_row[6] == "real"
    assert bbox_row[7] == "asteroid"
    assert bbox_row[8] == 0.93

    # Step 5: 验证审计日志和 manifest
    audit_path = dataset_root / ".audit" / bridge_module.REGION_STORAGE_AUDIT_LOG_NAME
    assert audit_path.exists()
    lines = [line for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    audit_record = json.loads(lines[0])
    assert audit_record["sample_id"] == task_data["sample_id"]
    assert audit_record["region_source"] == "annotation_result.js9_regions_json"
    assert audit_record["region_schema_version"] == bridge_module.JS9_REGION_SCHEMA_VERSION
    assert audit_record["annotation_mode"] == "js9_region_primary"
    assert audit_record["js9_region_count"] == 1
    assert audit_record["bbox_count"] == 1

    assert (dataset_root / "annotations.json").exists()
