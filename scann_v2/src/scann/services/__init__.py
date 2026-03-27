"""Service layer exports.

Keep imports lazy so optional dependencies used by specific services
do not break unrelated imports such as API test collection.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ConfigService",
    "PairService",
    "DetectionPipeline",
    "PipelineResult",
    "ModelInfo",
    "ModelLoadResult",
    "ModelService",
]

_EXPORTS = {
    "ConfigService": ("scann.services.config_service", "ConfigService"),
    "PairService": ("scann.services.pair_service", "PairService"),
    "DetectionPipeline": ("scann.services.detection_service", "DetectionPipeline"),
    "PipelineResult": ("scann.services.detection_service", "PipelineResult"),
    "ModelInfo": ("scann.services.model_service", "ModelInfo"),
    "ModelLoadResult": ("scann.services.model_service", "ModelLoadResult"),
    "ModelService": ("scann.services.model_service", "ModelService"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
