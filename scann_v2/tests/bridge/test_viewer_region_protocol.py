from __future__ import annotations

from fastapi.testclient import TestClient


def test_viewer_html_contains_regions_state(bridge_module):
    """测试 viewer HTML 包含 regionsState 变量初始化"""
    client = TestClient(bridge_module.app)
    resp = client.get(
        "/viewer/js9",
        params={
            "new": "http://127.0.0.1:3001/dataset/new/a.fts",
            "old": "http://127.0.0.1:3001/dataset/old/a.fts",
            "sample_id": "s1",
        },
    )

    assert resp.status_code == 200
    body = resp.text
    # 检查 regionsState 初始化
    assert "let regionsState = [];" in body
    assert "regionsState" in body


def test_viewer_html_contains_collectregions_function(bridge_module):
    """测试 viewer HTML 包含 collectRegions 函数定义"""
    client = TestClient(bridge_module.app)
    resp = client.get(
        "/viewer/js9",
        params={
            "new": "http://127.0.0.1:3001/dataset/new/a.fts",
            "old": "http://127.0.0.1:3001/dataset/old/a.fts",
            "sample_id": "s1",
        },
    )

    assert resp.status_code == 200
    body = resp.text
    # 检查 collectRegions 函数存在
    assert "function collectRegions()" in body
    # 检查 collectRegions 调用 JS9.GetRegions
    assert 'window.JS9.GetRegions("scannJS9")' in body
    # 检查返回 regionsState
    assert "return regionsState;" in body


def test_viewer_html_contains_applyregions_function(bridge_module):
    """测试 viewer HTML 包含 applyRegions 函数定义"""
    client = TestClient(bridge_module.app)
    resp = client.get(
        "/viewer/js9",
        params={
            "new": "http://127.0.0.1:3001/dataset/new/a.fts",
            "old": "http://127.0.0.1:3001/dataset/old/a.fts",
            "sample_id": "s1",
        },
    )

    assert resp.status_code == 200
    body = resp.text
    # 检查 applyRegions 函数存在
    assert "function applyRegions(regions)" in body
    # 检查 applyRegions 清除旧 regions
    assert 'window.JS9.ClearRegions("scannJS9")' in body
    # 检查 applyRegions 调用 JS9.AddRegions
    assert "window.JS9.AddRegions" in body
    # 检查更新 regionsState
    assert 'regionsState = regions;' in body


def test_viewer_html_contains_postmessage_listener(bridge_module):
    """测试 viewer HTML 包含 postMessage 监听器"""
    client = TestClient(bridge_module.app)
    resp = client.get(
        "/viewer/js9",
        params={
            "new": "http://127.0.0.1:3001/dataset/new/a.fts",
            "old": "http://127.0.0.1:3001/dataset/old/a.fts",
            "sample_id": "s1",
        },
    )

    assert resp.status_code == 200
    body = resp.text
    # 检查 addEventListener 存在
    assert 'window.addEventListener("message"' in body
    # 检查过滤 source 为 scann-host
    assert 'data?.source !== "scann-host"' in body
    # 检查处理 collectRegions 动作
    assert 'case "collectRegions":' in body
    # 检查处理 applyRegions 动作
    assert 'case "applyRegions":' in body
    # 检查处理 getRegions 动作
    assert 'case "getRegions":' in body


def test_viewer_html_postviewermessage_function(bridge_module):
    """测试 viewer HTML 包含 postViewerMessage 函数"""
    client = TestClient(bridge_module.app)
    resp = client.get(
        "/viewer/js9",
        params={
            "new": "http://127.0.0.1:3001/dataset/new/a.fts",
            "old": "http://127.0.0.1:3001/dataset/old/a.fts",
            "sample_id": "s1",
        },
    )

    assert resp.status_code == 200
    body = resp.text
    # 检查 postViewerMessage 函数存在
    assert "function postViewerMessage(type, payload)" in body
    # 检查发送到 parent
    assert 'window.parent.postMessage(' in body
    # 检查 source 标识
    assert 'source: "scann-viewer"' in body
    # 检查包含 type 和 payload
    assert "type, payload" in body


def test_viewer_protocol_regionscollected_message(bridge_module):
    """测试 collectRegions 动作发送 regionsCollected 消息"""
    client = TestClient(bridge_module.app)
    resp = client.get(
        "/viewer/js9",
        params={
            "new": "http://127.0.0.1:3001/dataset/new/a.fts",
            "old": "http://127.0.0.1:3001/dataset/old/a.fts",
            "sample_id": "s1",
        },
    )

    assert resp.status_code == 200
    body = resp.text
    # 检查收集 regions 后发送消息
    assert 'postViewerMessage("regionsCollected", regions)' in body


def test_viewer_protocol_regionsapplied_message(bridge_module):
    """测试 applyRegions 动作发送 regionsApplied 消息"""
    client = TestClient(bridge_module.app)
    resp = client.get(
        "/viewer/js9",
        params={
            "new": "http://127.0.0.1:3001/dataset/new/a.fts",
            "old": "http://127.0.0.1:3001/dataset/old/a.fts",
            "sample_id": "s1",
        },
    )

    assert resp.status_code == 200
    body = resp.text
    # 检查应用 regions 后发送成功消息
    assert 'postViewerMessage("regionsApplied", { success, regions: regionsState })' in body


def test_viewer_protocol_regionsdata_message(bridge_module):
    """测试 getRegions 动作发送 regionsData 消息"""
    client = TestClient(bridge_module.app)
    resp = client.get(
        "/viewer/js9",
        params={
            "new": "http://127.0.0.1:3001/dataset/new/a.fts",
            "old": "http://127.0.0.1:3001/dataset/old/a.fts",
            "sample_id": "s1",
        },
    )

    assert resp.status_code == 200
    body = resp.text
    # 检查获取 regions 后发送数据
    assert 'postViewerMessage("regionsData", regionsState)' in body


def test_viewer_region_shape_box_support(bridge_module):
    """测试 viewer 支持 box 形状 regions"""
    client = TestClient(bridge_module.app)
    resp = client.get(
        '/viewer/js9',
        params={
            "new": "http://127.0.0.1:3001/dataset/new/a.fts",
            "old": "http://127.0.0.1:3001/dataset/old/a.fts",
            "sample_id": "s1",
        },
    )

    assert resp.status_code == 200
    body = resp.text
    # 检查 box 形状处理
    assert 'case "box":' in body
    assert 'window.JS9.AddRegions("box", [region.x, region.y, region.width, region.height]' in body


def test_viewer_region_shape_circle_support(bridge_module):
    """测试 viewer 支持 circle 形状 regions"""
    client = TestClient(bridge_module.app)
    resp = client.get(
        "/viewer/js9",
        params={
            "new": "http://127.0.0.1:3001/dataset/new/a.fts",
            "old": "http://127.0.0.1:3001/dataset/old/a.fts",
            "sample_id": "s1",
        },
    )

    assert resp.status_code == 200
    body = resp.text
    # 检查 circle 形状处理
    assert 'case "circle":' in body
    assert 'window.JS9.AddRegions("circle", [region.x, region.y, region.radius]' in body


def test_viewer_region_shape_polygon_support(bridge_module):
    """测试 viewer 支持 polygon 形状 regions"""
    client = TestClient(bridge_module.app)
    resp = client.get(
        "/viewer/js9",
        params={
            "new": "http://127.0.0.1:3001/dataset/new/a.fts",
            "old": "http://127.0.0.1:3001/dataset/old/a.fts",
            "sample_id": "s1",
        },
    )

    assert resp.status_code == 200
    body = resp.text
    # 检查 polygon 形状处理
    assert 'case "polygon":' in body
    assert 'region.vertices && Array.isArray(region.vertices)' in body
    assert "region.vertices.flat()" in body


def test_viewer_region_attributes_preserved(bridge_module):
    """测试 viewer 保留 region 的 label、detail_type、confidence 属性"""
    client = TestClient(bridge_module.app)
    resp = client.get(
        "/viewer/js9",
        params={
            "new": "http://127.0.0.1:3001/dataset/new/a.fts",
            "old": "http://127.0.0.1:3001/dataset/old/a.fts",
            "sample_id": "s1",
        },
    )

    assert resp.status_code == 200
    body = resp.text
    # 检查 collectRegions 保留属性
    assert 'label: r.label || null' in body
    assert 'detail_type: r.detail_type || null' in body
    assert 'confidence: r.confidence || 1.0' in body
    # 检查 applyRegions 传递属性
    assert 'if (region.label) opts.label = region.label;' in body
    assert 'if (region.detail_type) opts.detail_type = region.detail_type;' in body
    assert 'if (region.confidence !== undefined) opts.confidence = region.confidence;' in body


def test_viewer_region_collect_error_handling(bridge_module):
    """测试 viewer collectRegions 的错误处理"""
    client = TestClient(bridge_module.app)
    resp = client.get(
        "/viewer/js9",
        params={
            "new": "http://127.0.0.1:3001/dataset/new/a.fts",
            "old": "http://127.0.0.1:3001/dataset/old/a.fts",
            "sample_id": "s1",
        },
    )

    assert resp.status_code == 200
    body = resp.text
    # 检查 JS9 检查
    assert 'if (hasJS9() && !js9Wrapper.classList.contains("hidden"))' in body
    # 检查 try-catch
    assert "try {" in body
    assert '} catch (_err) {' in body
    # 检查错误日志
    assert 'console.error("Failed to collect JS9 regions:", _err)' in body


def test_viewer_region_apply_error_handling(bridge_module):
    """测试 viewer applyRegions 的错误处理"""
    client = TestClient(bridge_module.app)
    resp = client.get(
        "/viewer/js9",
        params={
            "new": "http://127.0.0.1:3001/dataset/new/a.fts",
            "old": "http://127.0.0.1:3001/dataset/old/a.fts",
            "sample_id": "s1",
        },
    )

    assert resp.status_code == 200
    body = resp.text
    # 检查数组验证
    assert 'if (!Array.isArray(regions)) return false;' in body
    # 检查 try-catch
    assert "try {" in body
    assert '} catch (_err) {' in body
    # 检查错误日志
    assert 'console.error("Failed to apply JS9 regions:", _err)' in body
    # 检查返回失败
    assert 'return false;' in body


def test_viewer_postmessage_error_handling(bridge_module):
    """测试 viewer postMessage 的错误处理"""
    client = TestClient(bridge_module.app)
    resp = client.get(
        "/viewer/js9",
        params={
            "new": "http://127.0.0.1:3001/dataset/new/a.fts",
            "old": "http://127.0.0.1:3001/dataset/old/a.fts",
            "sample_id": "s1",
        },
    )

    assert resp.status_code == 200
    body = resp.text
    # 检查 postViewerMessage 错误处理
    assert 'catch (_err) {' in body
    assert 'console.error("Failed to post message to parent:", _err)' in body
