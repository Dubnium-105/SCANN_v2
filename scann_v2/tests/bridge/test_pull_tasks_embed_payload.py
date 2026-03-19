from __future__ import annotations

import html
from pathlib import Path


def test_pull_tasks_contains_embed_payload_fields(tmp_path: Path, monkeypatch, bridge_module):
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

    assert "new_url" in task_data
    assert "old_url" in task_data
    assert "new_marked_url" in task_data
    assert "preview_png" in task_data
    assert "js9_embed_url" in task_data
    assert "js9_iframe" in task_data
    assert ("viewer/js9" in task_data["js9_embed_url"]) or task_data["js9_embed_url"].endswith(".html")
    assert task_data["js9_embed_url"] in html.unescape(task_data["js9_iframe"])
