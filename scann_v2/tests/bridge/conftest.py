from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture(scope="session")
def bridge_module() -> ModuleType:
    app_path = Path(__file__).resolve().parents[2] / "bridge" / "app.py"
    spec = importlib.util.spec_from_file_location("scann_bridge_app", app_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load bridge module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
