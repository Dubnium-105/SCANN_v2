from __future__ import annotations

import os

from scann.ai.cache_paths import configure_torch_cache


def test_configure_torch_cache_honors_scann_override(tmp_path, monkeypatch) -> None:
    target = tmp_path / "persistent-cache"
    monkeypatch.setenv("SCANN_TORCH_CACHE_DIR", str(target))
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "ignored-cache"))

    resolved = configure_torch_cache(tmp_path / "default-root")

    assert resolved == target.resolve()
    assert resolved.is_dir()
    assert os.environ["TORCH_HOME"] == str(target.resolve())
    assert os.environ["TORCH_HUB_DIR"] == str(target.resolve())


def test_configure_torch_cache_uses_default_root(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("SCANN_TORCH_CACHE_DIR", raising=False)
    monkeypatch.delenv("TORCH_HOME", raising=False)

    resolved = configure_torch_cache(tmp_path)

    assert resolved == (tmp_path / "models" / "torch_cache").resolve()
