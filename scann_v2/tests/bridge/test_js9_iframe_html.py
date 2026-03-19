from __future__ import annotations


def test_make_js9_iframe_contains_iframe_and_fallback_links(bridge_module):
    html = bridge_module._make_js9_iframe(
        new_url="http://127.0.0.1:3001/dataset/new/a.fts",
        old_url="http://127.0.0.1:3001/dataset/old/a.fts",
        new_marked_url="http://127.0.0.1:3001/dataset/new_marked/a.fts",
        js9_embed_url="http://127.0.0.1:3001/viewer/js9?new=n&old=o&sample_id=s1",
        sample_id="s1",
    )

    assert "<iframe" in html
    assert "viewer/js9?" in html
    assert "新图" in html
    assert "旧图" in html
    assert "新图(标注)" in html
    assert 'data-sample-id="s1"' in html


def test_make_js9_iframe_disables_marked_link_when_missing(bridge_module):
    html = bridge_module._make_js9_iframe(
        new_url="http://127.0.0.1:3001/dataset/new/a.fts",
        old_url="http://127.0.0.1:3001/dataset/old/a.fts",
        new_marked_url=None,
        js9_embed_url="http://127.0.0.1:3001/viewer/js9?new=n&old=o&sample_id=s1",
        sample_id="s1",
    )

    assert "新图(标注)" not in html
