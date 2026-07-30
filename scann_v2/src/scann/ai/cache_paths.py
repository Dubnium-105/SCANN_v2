from __future__ import annotations

import os
from pathlib import Path


def configure_torch_cache(default_root: Path) -> Path:
    configured = (
        str(os.getenv("SCANN_TORCH_CACHE_DIR") or "").strip()
        or str(os.getenv("TORCH_HOME") or "").strip()
    )
    cache_dir = Path(configured).expanduser() if configured else Path(default_root) / "models" / "torch_cache"
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(cache_dir)
    os.environ["TORCH_HUB_DIR"] = str(cache_dir)
    return cache_dir
