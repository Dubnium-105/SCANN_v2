"""测试 webhook 向后兼容性：当没有 js9_regions_json 时回退到 rectanglelabels"""
from __future__ import annotations

import asyncio
from pathlib import Path


def test_webhook_fallback_to_rectanglelabels(tmp_path: Path, monkeypatch, bridge_module):
    """测试当没有 js9_regions_json 时，回退到解析 rectanglelabels"""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True)
    
    # 创建配置
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

    # 模拟数据库写入
    written_samples = {}

    def _fake_upsert_sample(sample_id: str, file_name: str, bboxes: list):
        written_samples[sample_id] = {"file_name": file_name, "bboxes": bboxes}

    monkeypatch.setattr(bridge_module, "_upsert_sample", _fake_upsert_sample)

    # 构造旧格式的 payload（没有 js9_regions_json）
    payload = {
        "annotations": [
            {
                "id": 1,
                "task": {
                    "id": 100,
                    "data": {
                        "sample_id": "sample1",
                        "file_name": "SAMPLE 1.fts",
                    },
                },
                "result": [
                    {
                        "type": "rectanglelabels",
                        "value": {
                            "x": 10.0,
                            "y": 20.0,
                            "width": 30.0,
                            "height": 40.0,
                            "rectanglelabels": ["asteroid"],
                        },
                        "original_width": 1000,
                        "original_height": 1000,
                    }
                ],
            }
        ]
    }

    # 创建模拟的 request 对象
    class FakeRequest:
        async def json(self):
            return payload

    # 调用 webhook（异步函数）
    resp = asyncio.run(bridge_module.labelstudio_webhook(FakeRequest()))
    assert resp.updated_samples == 1

    # 验证写入了正确的 bbox
    assert "sample1" in written_samples
    assert written_samples["sample1"]["file_name"] == "SAMPLE 1.fts"
    assert len(written_samples["sample1"]["bboxes"]) == 1
    bbox = written_samples["sample1"]["bboxes"][0]
    assert bbox["x"] == 100  # 10.0% of 1000
    assert bbox["y"] == 200  # 20.0% of 1000
    assert bbox["width"] == 300  # 30.0% of 1000
    assert bbox["height"] == 400  # 40.0% of 1000
    assert bbox["label"] == "real"
    assert bbox["detail_type"] == "asteroid"


def test_webhook_prefer_js9_regions_over_rectanglelabels(tmp_path: Path, monkeypatch, bridge_module):
    """测试当同时存在 js9_regions_json 和 rectanglelabels 时，优先使用 js9_regions_json"""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True)
    
    # 创建配置
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

    # 模拟数据库写入
    written_samples = {}

    def _fake_upsert_sample(sample_id: str, file_name: str, bboxes: list):
        written_samples[sample_id] = {"file_name": file_name, "bboxes": bboxes}

    monkeypatch.setattr(bridge_module, "_upsert_sample", _fake_upsert_sample)

    # 构造同时包含 js9_regions_json 和 rectanglelabels 的 payload
    # js9_regions_json 有一个 box (asteroid)
    # rectanglelabels 有一个 box (supernova)
    js9_regions_json = [
        {
            "shape": "box",
            "x": 100,
            "y": 200,
            "width": 50,
            "height": 60,
            "label": "real",
            "detail_type": "asteroid",
        }
    ]

    payload = {
        "annotations": [
            {
                "id": 1,
                "task": {
                    "id": 100,
                    "data": {
                        "sample_id": "sample1",
                        "file_name": "SAMPLE 1.fts",
                        "js9_regions_json": js9_regions_json,  # 优先级高
                    },
                },
                "result": [
                    {
                        "type": "rectanglelabels",
                        "value": {
                            "x": 50.0,
                            "y": 60.0,
                            "width": 70.0,
                            "height": 80.0,
                            "rectanglelabels": ["supernova"],  # 应该被忽略
                        },
                        "original_width": 1000,
                        "original_height": 1000,
                    }
                ],
            }
        ]
    }

    # 创建模拟的 request 对象
    class FakeRequest:
        async def json(self):
            return payload

    # 调用 webhook（异步函数）
    resp = asyncio.run(bridge_module.labelstudio_webhook(FakeRequest()))
    assert resp.updated_samples == 1

    # 验证使用了 js9_regions_json 而不是 rectanglelabels
    assert "sample1" in written_samples
    assert len(written_samples["sample1"]["bboxes"]) == 1
    bbox = written_samples["sample1"]["bboxes"][0]
    assert bbox["x"] == 100  # 来自 js9_regions_json
    assert bbox["y"] == 200
    assert bbox["width"] == 50
    assert bbox["height"] == 60
    assert bbox["detail_type"] == "asteroid"  # 来自 js9_regions_json
    # 不是 supernova（rectanglelabels 应该被忽略）
    assert bbox["detail_type"] != "supernova"


def test_webhook_multiple_annotations_mixed_format(tmp_path: Path, monkeypatch, bridge_module):
    """测试多个 annotation，有的有 js9_regions_json，有的只有 rectanglelabels"""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True)
    
    # 创建配置
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

    # 模拟数据库写入
    written_samples = {}

    def _fake_upsert_sample(sample_id: str, file_name: str, bboxes: list):
        written_samples[sample_id] = {"file_name": file_name, "bboxes": bboxes}

    monkeypatch.setattr(bridge_module, "_upsert_sample", _fake_upsert_sample)

    # 构造多个 annotation
    payload = {
        "annotations": [
            # 第一个 annotation：使用 js9_regions_json
            {
                "id": 1,
                "task": {
                    "id": 100,
                    "data": {
                        "sample_id": "sample1",
                        "file_name": "SAMPLE 1.fts",
                        "js9_regions_json": [
                            {
                                "shape": "box",
                                "x": 100,
                                "y": 200,
                                "width": 50,
                                "height": 60,
                                "label": "real",
                                "detail_type": "asteroid",
                            }
                        ],
                    },
                },
                "result": [
                    {
                        "type": "rectanglelabels",
                        "value": {
                            "x": 50.0,
                            "y": 60.0,
                            "width": 70.0,
                            "height": 80.0,
                            "rectanglelabels": ["supernova"],
                        },
                        "original_width": 1000,
                        "original_height": 1000,
                    }
                ],
            },
            # 第二个 annotation：只有 rectanglelabels（旧格式）
            {
                "id": 2,
                "task": {
                    "id": 200,
                    "data": {
                        "sample_id": "sample2",
                        "file_name": "SAMPLE 2.fts",
                    },
                },
                "result": [
                    {
                        "type": "rectanglelabels",
                        "value": {
                            "x": 30.0,
                            "y": 40.0,
                            "width": 50.0,
                            "height": 60.0,
                            "rectanglelabels": ["variable_star"],
                        },
                        "original_width": 800,
                        "original_height": 800,
                    }
                ],
            },
        ]
    }

    # 创建模拟的 request 对象
    class FakeRequest:
        async def json(self):
            return payload

    # 调用 webhook（异步函数）
    resp = asyncio.run(bridge_module.labelstudio_webhook(FakeRequest()))
    assert resp.updated_samples == 2

    # 验证 sample1 使用了 js9_regions_json
    assert "sample1" in written_samples
    assert len(written_samples["sample1"]["bboxes"]) == 1
    bbox1 = written_samples["sample1"]["bboxes"][0]
    assert bbox1["x"] == 100  # 来自 js9_regions_json
    assert bbox1["detail_type"] == "asteroid"

    # 验证 sample2 回退到 rectanglelabels
    assert "sample2" in written_samples
    assert len(written_samples["sample2"]["bboxes"]) == 1
    bbox2 = written_samples["sample2"]["bboxes"][0]
    assert bbox2["x"] == 240  # 30.0% of 800
    assert bbox2["y"] == 320  # 40.0% of 800
    assert bbox2["width"] == 400  # 50.0% of 800
    assert bbox2["height"] == 480  # 60.0% of 800
    assert bbox2["detail_type"] == "variable_star"


def test_webhook_empty_js9_regions_json_fallback(tmp_path: Path, monkeypatch, bridge_module):
    """测试当 js9_regions_json 为空列表时，回退到 rectanglelabels"""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True)
    
    # 创建配置
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

    # 模拟数据库写入
    written_samples = {}

    def _fake_upsert_sample(sample_id: str, file_name: str, bboxes: list):
        written_samples[sample_id] = {"file_name": file_name, "bboxes": bboxes}

    monkeypatch.setattr(bridge_module, "_upsert_sample", _fake_upsert_sample)

    # 构造 payload：js9_regions_json 为空列表，但有 rectanglelabels
    payload = {
        "annotations": [
            {
                "id": 1,
                "task": {
                    "id": 100,
                    "data": {
                        "sample_id": "sample1",
                        "file_name": "SAMPLE 1.fts",
                        "js9_regions_json": [],  # 空列表
                    },
                },
                "result": [
                    {
                        "type": "rectanglelabels",
                        "value": {
                            "x": 10.0,
                            "y": 20.0,
                            "width": 30.0,
                            "height": 40.0,
                            "rectanglelabels": ["asteroid"],
                        },
                        "original_width": 1000,
                        "original_height": 1000,
                    }
                ],
            }
        ]
    }

    # 创建模拟的 request 对象
    class FakeRequest:
        async def json(self):
            return payload

    # 调用 webhook（异步函数）
    resp = asyncio.run(bridge_module.labelstudio_webhook(FakeRequest()))
    assert resp.updated_samples == 1

    # 验证回退到 rectanglelabels
    assert "sample1" in written_samples
    assert len(written_samples["sample1"]["bboxes"]) == 1
    bbox = written_samples["sample1"]["bboxes"][0]
    assert bbox["x"] == 100  # 来自 rectanglelabels
    assert bbox["detail_type"] == "asteroid"


def test_webhook_invalid_js9_regions_json_fallback(tmp_path: Path, monkeypatch, bridge_module):
    """测试当 js9_regions_json 无效时，回退到 rectanglelabels"""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True)
    
    # 创建配置
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

    # 模拟数据库写入
    written_samples = {}

    def _fake_upsert_sample(sample_id: str, file_name: str, bboxes: list):
        written_samples[sample_id] = {"file_name": file_name, "bboxes": bboxes}

    monkeypatch.setattr(bridge_module, "_upsert_sample", _fake_upsert_sample)

    # 构造 payload：js9_regions_json 为无效的 JSON 字符串，但有 rectanglelabels
    payload = {
        "annotations": [
            {
                "id": 1,
                "task": {
                    "id": 100,
                    "data": {
                        "sample_id": "sample1",
                        "file_name": "SAMPLE 1.fts",
                        "js9_regions_json": "invalid json{{{",
                    },
                },
                "result": [
                    {
                        "type": "rectanglelabels",
                        "value": {
                            "x": 10.0,
                            "y": 20.0,
                            "width": 30.0,
                            "height": 40.0,
                            "rectanglelabels": ["asteroid"],
                        },
                        "original_width": 1000,
                        "original_height": 1000,
                    }
                ],
            }
        ]
    }

    # 创建模拟的 request 对象
    class FakeRequest:
        async def json(self):
            return payload

    # 调用 webhook（异步函数）
    resp = asyncio.run(bridge_module.labelstudio_webhook(FakeRequest()))
    assert resp.updated_samples == 1

    # 验证回退到 rectanglelabels
    assert "sample1" in written_samples
    assert len(written_samples["sample1"]["bboxes"]) == 1
    bbox = written_samples["sample1"]["bboxes"][0]
    assert bbox["x"] == 100  # 来自 rectanglelabels
    assert bbox["detail_type"] == "asteroid"
