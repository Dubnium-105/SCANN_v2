from __future__ import annotations

from fastapi.testclient import TestClient


def test_viewer_route_returns_toolbar_dom(bridge_module):
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
    assert 'id="scann-js9-toolbar"' in body
    assert 'id="scann-js9-viewer"' in body
    assert 'id="btn-blink"' in body
    assert 'id="btn-invert"' in body


def test_viewer_route_missing_required_query_gets_4xx(bridge_module):
    client = TestClient(bridge_module.app)
    resp = client.get("/viewer/js9", params={"new": "x", "sample_id": "s1"})
    assert 400 <= resp.status_code < 500
