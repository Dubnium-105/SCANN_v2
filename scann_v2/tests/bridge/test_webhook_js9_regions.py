"""测试 webhook 解析 js9_regions_json 字段的功能"""
from __future__ import annotations

from pathlib import Path


def test_extract_js9_regions_from_task_data_valid_json(tmp_path: Path, monkeypatch, bridge_module):
    """测试从 task data 中提取有效的 js9_regions_json（JSON 字符串格式）"""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True)
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

    js9_regions_json = """[
        {
            "shape": "box",
            "x": 100,
            "y": 200,
            "width": 50,
            "height": 60,
            "label": "real",
            "detail_type": "asteroid"
        },
        {
            "shape": "circle",
            "x": 300,
            "y": 400,
            "radius": 25,
            "label": "bogus",
            "detail_type": "noise"
        }
    ]"""

    data = {
        "sample_id": "sample1",
        "js9_regions_json": js9_regions_json,
    }

    regions = bridge_module._extract_js9_regions_from_task_data(data)
    assert regions is not None
    assert len(regions) == 2
    assert regions[0]["shape"] == "box"
    assert regions[1]["shape"] == "circle"


def test_extract_js9_regions_from_task_data_valid_list(tmp_path: Path, monkeypatch, bridge_module):
    """测试从 task data 中提取有效的 js9_regions_json（列表格式）"""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True)
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

    js9_regions_json = [
        {
            "shape": "box",
            "x": 100,
            "y": 200,
            "width": 50,
            "height": 60,
            "label": "real",
            "detail_type": "asteroid"
        }
    ]

    data = {
        "sample_id": "sample1",
        "js9_regions_json": js9_regions_json,
    }

    regions = bridge_module._extract_js9_regions_from_task_data(data)
    assert regions is not None
    assert len(regions) == 1
    assert regions[0]["shape"] == "box"


def test_extract_js9_regions_from_task_data_missing(tmp_path: Path, monkeypatch, bridge_module):
    """测试当 js9_regions_json 不存在时返回 None"""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True)
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

    data = {
        "sample_id": "sample1",
    }

    regions = bridge_module._extract_js9_regions_from_task_data(data)
    assert regions is None


def test_extract_js9_regions_from_task_data_invalid_json(tmp_path: Path, monkeypatch, bridge_module):
    """测试当 js9_regions_json 是无效的 JSON 字符串时返回 None"""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True)
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

    data = {
        "sample_id": "sample1",
        "js9_regions_json": "invalid json{{{",
    }

    regions = bridge_module._extract_js9_regions_from_task_data(data)
    assert regions is None


def test_extract_js9_regions_from_task_data_not_list(tmp_path: Path, monkeypatch, bridge_module):
    """测试当 js9_regions_json 不是列表时返回 None"""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True)
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

    data = {
        "sample_id": "sample1",
        "js9_regions_json": {"shape": "box", "x": 100, "y": 200},
    }

    regions = bridge_module._extract_js9_regions_from_task_data(data)
    assert regions is None


def test_convert_js9_regions_to_bboxes_box(tmp_path: Path, monkeypatch, bridge_module):
    """测试将 box region 转换为 bbox"""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True)
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

    regions = [
        {
            "shape": "box",
            "x": 100,
            "y": 200,
            "width": 50,
            "height": 60,
            "label": "real",
            "detail_type": "asteroid",
            "confidence": 0.95,
        }
    ]

    bboxes = bridge_module._convert_js9_regions_to_bboxes(regions, 1000, 1000)
    assert len(bboxes) == 1
    assert bboxes[0]["x"] == 100
    assert bboxes[0]["y"] == 200
    assert bboxes[0]["width"] == 50
    assert bboxes[0]["height"] == 60
    assert bboxes[0]["label"] == "real"
    assert bboxes[0]["detail_type"] == "asteroid"
    assert bboxes[0]["confidence"] == 0.95


def test_convert_js9_regions_to_bboxes_circle(tmp_path: Path, monkeypatch, bridge_module):
    """测试将 circle region 转换为 bbox"""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True)
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

    regions = [
        {
            "shape": "circle",
            "x": 300,
            "y": 400,
            "radius": 25,
            "label": "bogus",
            "detail_type": "noise",
        }
    ]

    bboxes = bridge_module._convert_js9_regions_to_bboxes(regions, 1000, 1000)
    assert len(bboxes) == 1
    assert bboxes[0]["x"] == 275  # 300 - 25
    assert bboxes[0]["y"] == 375  # 400 - 25
    assert bboxes[0]["width"] == 50  # 25 * 2
    assert bboxes[0]["height"] == 50
    assert bboxes[0]["label"] == "bogus"
    assert bboxes[0]["detail_type"] == "noise"


def test_convert_js9_regions_to_bboxes_polygon(tmp_path: Path, monkeypatch, bridge_module):
    """测试将 polygon region 转换为 bbox"""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True)
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

    regions = [
        {
            "shape": "polygon",
            "x": 0,
            "y": 0,
            "vertices": [[100, 100], [200, 150], [180, 250], [80, 200]],
            "label": "real",
            "detail_type": "asteroid",
        }
    ]

    bboxes = bridge_module._convert_js9_regions_to_bboxes(regions, 1000, 1000)
    assert len(bboxes) == 1
    assert bboxes[0]["x"] == 80
    assert bboxes[0]["y"] == 100
    assert bboxes[0]["width"] == 120  # 200 - 80
    assert bboxes[0]["height"] == 150  # 250 - 100
    assert bboxes[0]["label"] == "real"


def test_convert_js9_regions_to_bboxes_clipping(tmp_path: Path, monkeypatch, bridge_module):
    """测试坐标裁剪功能"""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True)
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

    # 越界坐标测试
    regions = [
        {
            "shape": "box",
            "x": -10,  # 负坐标
            "y": -20,
            "width": 150,  # 超过图像宽度
            "height": 200,
            "label": "real",
            "detail_type": "asteroid",
        }
    ]

    bboxes = bridge_module._convert_js9_regions_to_bboxes(regions, 100, 100)
    assert len(bboxes) == 1
    assert bboxes[0]["x"] == 0  # 被裁剪为 0
    assert bboxes[0]["y"] == 0  # 被裁剪为 0
    assert bboxes[0]["width"] == 100  # 被裁剪为 100
    assert bboxes[0]["height"] == 100  # 被裁剪为 100


def test_convert_js9_regions_to_bboxes_label_inference(tmp_path: Path, monkeypatch, bridge_module):
    """测试从 detail_type 推断 label"""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True)
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

    # 没有 label，但有 detail_type
    regions = [
        {
            "shape": "box",
            "x": 100,
            "y": 200,
            "width": 50,
            "height": 60,
            "detail_type": "asteroid",  # 应该推断为 "real"
        }
    ]

    bboxes = bridge_module._convert_js9_regions_to_bboxes(regions, 1000, 1000)
    assert len(bboxes) == 1
    assert bboxes[0]["label"] == "real"
    assert bboxes[0]["detail_type"] == "asteroid"


def test_convert_js9_regions_to_bboxes_filter_invalid(tmp_path: Path, monkeypatch, bridge_module):
    """测试过滤无效的 bbox（没有 label 或尺寸为 0）"""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True)
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

    regions = [
        {
            "shape": "box",
            "x": 100,
            "y": 200,
            "width": 50,
            "height": 60,
            "label": "real",
            "detail_type": "asteroid",
        },
        {
            "shape": "box",
            "x": 300,
            "y": 400,
            "width": 0,  # 宽度为 0
            "height": 10,
            "label": "real",
            "detail_type": "asteroid",
        },
        {
            "shape": "box",
            "x": 500,
            "y": 600,
            "width": 50,
            "height": 60,
            # 没有 label
            "detail_type": "unknown_type",  # 未知类型
        },
    ]

    bboxes = bridge_module._convert_js9_regions_to_bboxes(regions, 1000, 1000)
    # 只应该有一个有效的 bbox
    assert len(bboxes) == 1
    assert bboxes[0]["x"] == 100
    assert bboxes[0]["label"] == "real"
