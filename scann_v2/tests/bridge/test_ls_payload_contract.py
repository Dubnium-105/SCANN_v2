"""C05 合约测试：Label Studio 配置/模板联动

验证目标：
1. 标注模板包含 `js9_regions_json` 可提交字段。
2. webhook 可从 annotation result 中解析 `js9_regions_json`。
3. 保留 rectanglelabels 作为降级通道。
"""
from __future__ import annotations

import asyncio
from pathlib import Path


def _make_bridge_config(bridge_module, dataset_root: Path):
    return bridge_module.BridgeConfig(
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


def test_phaseb_label_config_contains_required_fields(bridge_module):
    """模板需包含 js9_regions_json 字段与 rectanglelabels 降级工具"""
    xml = bridge_module.get_label_studio_phaseb_label_config()

    assert '<HyperText name="js9_iframe" value="$js9_iframe"/>' in xml
    assert '<Image name="preview_png" value="$preview_png"/>' in xml
    assert 'name="js9_regions_json"' in xml
    assert '<RectangleLabels name="bbox" toName="preview_png"' in xml


def test_extract_js9_regions_from_result_textarea(bridge_module):
    """支持从 TextArea 结果中提取 JSON 字符串"""
    results = [
        {
            "from_name": "js9_regions_json",
            "type": "textarea",
            "value": {
                "text": [
                    '[{"shape":"box","x":120,"y":220,"width":60,"height":40,"label":"real","detail_type":"asteroid"}]'
                ]
            },
        }
    ]

    regions = bridge_module._extract_js9_regions_from_annotation_results(results)
    assert regions is not None
    assert len(regions) == 1
    assert regions[0]["shape"] == "box"
    assert regions[0]["detail_type"] == "asteroid"


def test_extract_js9_regions_from_result_empty_is_authoritative(bridge_module):
    """当结果字段明确为 [] 时，返回空列表（代表用户清空了 region）"""
    results = [
        {
            "from_name": "js9_regions_json",
            "type": "textarea",
            "value": {"text": ["[]"]},
        }
    ]

    regions = bridge_module._extract_js9_regions_from_annotation_results(results)
    assert regions == []


def test_webhook_uses_result_js9_regions_first(tmp_path: Path, monkeypatch, bridge_module):
    """同一 annotation 中，result.js9_regions_json 优先级应高于 task.data 与 rectanglelabels"""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True)
    bridge_module.CONFIG = _make_bridge_config(bridge_module, dataset_root)

    written_samples = {}

    def _fake_upsert_sample(sample_id: str, file_name: str, bboxes: list):
        written_samples[sample_id] = {"file_name": file_name, "bboxes": bboxes}

    monkeypatch.setattr(bridge_module, "_upsert_sample", _fake_upsert_sample)

    payload = {
        "annotations": [
            {
                "task": {
                    "data": {
                        "sample_id": "sample-c05-1",
                        "file_name": "SAMPLE C05-1.fts",
                        "js9_regions_json": [
                            {
                                "shape": "box",
                                "x": 5,
                                "y": 5,
                                "width": 5,
                                "height": 5,
                                "label": "bogus",
                                "detail_type": "noise",
                            }
                        ],
                    }
                },
                "result": [
                    {
                        "from_name": "js9_regions_json",
                        "type": "textarea",
                        "value": {
                            "text": [
                                '[{"shape":"box","x":100,"y":120,"width":30,"height":40,"label":"real","detail_type":"asteroid"}]'
                            ]
                        },
                    },
                    {
                        "type": "rectanglelabels",
                        "value": {
                            "x": 60.0,
                            "y": 60.0,
                            "width": 10.0,
                            "height": 10.0,
                            "rectanglelabels": ["supernova"],
                        },
                        "original_width": 1000,
                        "original_height": 1000,
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

    assert "sample-c05-1" in written_samples
    bbox = written_samples["sample-c05-1"]["bboxes"][0]
    assert bbox["x"] == 100
    assert bbox["y"] == 120
    assert bbox["detail_type"] == "asteroid"


def test_webhook_result_empty_regions_no_fallback(tmp_path: Path, monkeypatch, bridge_module):
    """当 result.js9_regions_json 明确为空时，不应回退到 rectanglelabels"""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True)
    bridge_module.CONFIG = _make_bridge_config(bridge_module, dataset_root)

    written_samples = {}

    def _fake_upsert_sample(sample_id: str, file_name: str, bboxes: list):
        written_samples[sample_id] = {"file_name": file_name, "bboxes": bboxes}

    monkeypatch.setattr(bridge_module, "_upsert_sample", _fake_upsert_sample)

    payload = {
        "annotations": [
            {
                "task": {
                    "data": {
                        "sample_id": "sample-c05-2",
                        "file_name": "SAMPLE C05-2.fts",
                    }
                },
                "result": [
                    {
                        "from_name": "js9_regions_json",
                        "type": "textarea",
                        "value": {"text": ["[]"]},
                    },
                    {
                        "type": "rectanglelabels",
                        "value": {
                            "x": 10.0,
                            "y": 10.0,
                            "width": 50.0,
                            "height": 40.0,
                            "rectanglelabels": ["asteroid"],
                        },
                        "original_width": 1000,
                        "original_height": 1000,
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

    assert "sample-c05-2" in written_samples
    assert written_samples["sample-c05-2"]["bboxes"] == []
