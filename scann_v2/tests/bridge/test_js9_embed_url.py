from __future__ import annotations

from urllib.parse import parse_qs, urlsplit


def test_make_js9_embed_url_encodes_params(bridge_module):
    bridge_module.CONFIG.js9_base_url = "http://127.0.0.1:3001"

    new_url = "http://127.0.0.1:3001/dataset/new/中文 文件.fts"
    old_url = "http://127.0.0.1:3001/dataset/old/a+b?.fts"
    marked_url = "http://127.0.0.1:3001/dataset/new_marked/m 1.fts"
    sample_id = "样本 A <script>alert(1)</script>"

    url = bridge_module._make_js9_embed_url(new_url, old_url, sample_id, marked_url)

    parts = urlsplit(url)
    assert parts.path == "/viewer/js9"

    qs = parse_qs(parts.query)
    assert qs["new"][0] == new_url
    assert qs["old"][0] == old_url
    assert qs["marked"][0] == marked_url
    assert qs["sample_id"][0] == sample_id

    assert "<script>" not in url
    assert "%3Cscript%3E" in url
