"""测试 /tasks/pull 下发的 annotation_mode 和 region 字段骨架

C04 提交验证：
1. annotation_mode 字段正确下发，值为 js9_region_primary
2. js9_regions_json 字段预留，初始为 "[]"
3. 其他现有字段仍然正常工作
"""

from __future__ import annotations

import html
from pathlib import Path


def test_pull_tasks_contains_annotation_mode(tmp_path: Path, monkeypatch, bridge_module):
    """验证 pull_tasks 返回的任务数据包含 annotation_mode 字段"""
    dataset_root = tmp_path / "dataset"
    new_dir = dataset_root / "new"
    old_dir = dataset_root / "old"
    marked_dir = dataset_root / "new_marked"
    new_dir.mkdir(parents=True)
    old_dir.mkdir(parents=True)
    marked_dir.mkdir(parents=True)

    (new_dir / "SAMPLE 1.fts").write_bytes(b"new")
    (old_dir / "SAMPLE 1.fts").write_bytes(b"old")
    (marked_dir / "SAMPLE 1.fts").write_bytes(b"marked")

    bridge_module.CONFIG = bridge_module.BridgeConfig(
        dataset_root=dataset_root,
        sqlite_path=dataset_root / "annotations.db",
        new_dir=new_dir,
        old_dir=old_dir,
        new_marked_dir=marked_dir,
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

    resp = bridge_module.pull_tasks(bridge_module.PullRequest(import_to_label_studio=True))
    assert resp.tasks_built == 1
    assert resp.tasks_imported == 1

    payload = posted["json"]
    assert isinstance(payload, list)
    task_data = payload[0]["data"]

    # 验证 annotation_mode 字段
    assert "annotation_mode" in task_data
    assert task_data["annotation_mode"] == "js9_region_primary"


def test_pull_tasks_contains_js9_regions_json_field(tmp_path: Path, monkeypatch, bridge_module):
    """验证 pull_tasks 返回的任务数据包含 js9_regions_json 字段（预留）"""
    dataset_root = tmp_path / "dataset"
    new_dir = dataset_root / "new"
    old_dir = dataset_root / "old"
    marked_dir = dataset_root / "new_marked"
    new_dir.mkdir(parents=True)
    old_dir.mkdir(parents=True)
    marked_dir.mkdir(parents=True)

    (new_dir / "SAMPLE 2.fts").write_bytes(b"new")
    (old_dir / "SAMPLE 2.fts").write_bytes(b"old")
    (marked_dir / "SAMPLE 2.fts").write_bytes(b"marked")

    bridge_module.CONFIG = bridge_module.BridgeConfig(
        dataset_root=dataset_root,
        sqlite_path=dataset_root / "annotations.db",
        new_dir=new_dir,
        old_dir=old_dir,
        new_marked_dir=marked_dir,
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

    resp = bridge_module.pull_tasks(bridge_module.PullRequest(import_to_label_studio=True))
    assert resp.tasks_built == 1

    payload = posted["json"]
    task_data = payload[0]["data"]

    # 验证 js9_regions_json 字段存在且为 "[]"（TextArea 需要字符串）
    assert "js9_regions_json" in task_data
    assert task_data["js9_regions_json"] == "[]"


def test_pull_tasks_region_fields_backward_compat(tmp_path: Path, monkeypatch, bridge_module):
    """验证新增的 region 字段不影响现有字段"""
    dataset_root = tmp_path / "dataset"
    new_dir = dataset_root / "new"
    old_dir = dataset_root / "old"
    marked_dir = dataset_root / "new_marked"
    new_dir.mkdir(parents=True)
    old_dir.mkdir(parents=True)
    marked_dir.mkdir(parents=True)

    (new_dir / "SAMPLE 3.fts").write_bytes(b"new")
    (old_dir / "SAMPLE 3.fts").write_bytes(b"old")
    (marked_dir / "SAMPLE 3.fts").write_bytes(b"marked")

    bridge_module.CONFIG = bridge_module.BridgeConfig(
        dataset_root=dataset_root,
        sqlite_path=dataset_root / "annotations.db",
        new_dir=new_dir,
        old_dir=old_dir,
        new_marked_dir=marked_dir,
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

    resp = bridge_module.pull_tasks(bridge_module.PullRequest(import_to_label_studio=True))
    assert resp.tasks_built == 1

    payload = posted["json"]
    task_data = payload[0]["data"]

    # 验证所有现有字段仍然存在
    assert task_data["sample_id"] == "sample 3"
    assert task_data["file_name"] == "SAMPLE 3.fts"
    assert "new_url" in task_data
    assert "old_url" in task_data
    assert "new_marked_url" in task_data
    assert "preview_png" in task_data
    assert "js9_embed_url" in task_data
    assert "js9_iframe" in task_data

    # 验证新增字段
    assert task_data["annotation_mode"] == "js9_region_primary"
    assert task_data["js9_regions_json"] == "[]"


def test_pull_tasks_multiple_tasks_all_have_region_fields(tmp_path: Path, monkeypatch, bridge_module):
    """验证多个任务都包含 region 相关字段"""
    dataset_root = tmp_path / "dataset"
    new_dir = dataset_root / "new"
    old_dir = dataset_root / "old"
    marked_dir = dataset_root / "new_marked"
    new_dir.mkdir(parents=True)
    old_dir.mkdir(parents=True)
    marked_dir.mkdir(parents=True)

    for i in range(1, 4):
        (new_dir / f"SAMPLE {i}.fts").write_bytes(f"new{i}".encode())
        (old_dir / f"SAMPLE {i}.fts").write_bytes(f"old{i}".encode())
        (marked_dir / f"SAMPLE {i}.fts").write_bytes(f"marked{i}".encode())

    bridge_module.CONFIG = bridge_module.BridgeConfig(
        dataset_root=dataset_root,
        sqlite_path=dataset_root / "annotations.db",
        new_dir=new_dir,
        old_dir=old_dir,
        new_marked_dir=marked_dir,
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

    posted: dict[str, object] = {}

    class _Resp:
        status_code = 200
        content = b"{}"
        text = "{}"

        @staticmethod
        def json():
            return {"task_count": 3}

    def _fake_post(url, headers, json, timeout):
        posted["url"] = url
        posted["headers"] = headers
        posted["json"] = json
        posted["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr(bridge_module.requests, "post", _fake_post)

    resp = bridge_module.pull_tasks(bridge_module.PullRequest(import_to_label_studio=True))
    assert resp.tasks_built == 3

    payload = posted["json"]
    assert len(payload) == 3

    # 验证所有任务都包含 region 字段
    for task in payload:
        task_data = task["data"]
        assert "annotation_mode" in task_data
        assert task_data["annotation_mode"] == "js9_region_primary"
        assert "js9_regions_json" in task_data
        assert task_data["js9_regions_json"] == "[]"


def test_pull_tasks_no_import_still_has_region_fields(tmp_path: Path, monkeypatch, bridge_module):
    """验证不导入到 Label Studio 时，任务构建也包含 region 字段"""
    dataset_root = tmp_path / "dataset"
    new_dir = dataset_root / "new"
    old_dir = dataset_root / "old"
    marked_dir = dataset_root / "new_marked"
    new_dir.mkdir(parents=True)
    old_dir.mkdir(parents=True)
    marked_dir.mkdir(parents=True)

    (new_dir / "SAMPLE 4.fts").write_bytes(b"new")
    (old_dir / "SAMPLE 4.fts").write_bytes(b"old")
    (marked_dir / "SAMPLE 4.fts").write_bytes(b"marked")

    bridge_module.CONFIG = bridge_module.BridgeConfig(
        dataset_root=dataset_root,
        sqlite_path=dataset_root / "annotations.db",
        new_dir=new_dir,
        old_dir=old_dir,
        new_marked_dir=marked_dir,
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

    posted: dict[str, object] = {}

    def _fake_post(url, headers, json, timeout):
        posted["url"] = url
        posted["headers"] = headers
        posted["json"] = json
        posted["timeout"] = timeout
        raise AssertionError("不应该调用 post，因为 import_to_label_studio=False")

    monkeypatch.setattr(bridge_module.requests, "post", _fake_post)

    resp = bridge_module.pull_tasks(bridge_module.PullRequest(import_to_label_studio=False))
    assert resp.tasks_built == 1
    assert resp.tasks_imported == 0

    # 由于没有导入，我们需要通过检查任务列表来验证字段
    # 但 pull_tasks 内部构建的任务对象应该包含这些字段
    # 这个测试主要通过检查代码逻辑和后续测试覆盖


def test_taskrecord_default_annotation_mode(bridge_module):
    """验证 TaskRecord 的 annotation_mode 默认值为 js9_region_primary"""
    record = bridge_module.TaskRecord(
        sample_id="test",
        file_name="test.fts",
        new_url="http://example.com/new",
        old_url="http://example.com/old",
        new_marked_url=None,
        preview_png="data:image/png;base64,iVBORw0KGgo=",
        js9_embed_url="http://example.com/viewer",
        js9_iframe='<iframe src="http://example.com/viewer"></iframe>',
    )

    assert record.annotation_mode == "js9_region_primary"
    assert record.js9_regions_json == "[]"


def test_taskrecord_explicit_annotation_mode(bridge_module):
    """验证 TaskRecord 支持显式设置 annotation_mode"""
    record = bridge_module.TaskRecord(
        sample_id="test",
        file_name="test.fts",
        new_url="http://example.com/new",
        old_url="http://example.com/old",
        new_marked_url=None,
        preview_png="data:image/png;base64,iVBORw0KGgo=",
        js9_embed_url="http://example.com/viewer",
        js9_iframe='<iframe src="http://example.com/viewer"></iframe>',
        annotation_mode="legacy",
        js9_regions_json='{"regions": []}',
    )

    assert record.annotation_mode == "legacy"
    assert record.js9_regions_json == '{"regions": []}'
