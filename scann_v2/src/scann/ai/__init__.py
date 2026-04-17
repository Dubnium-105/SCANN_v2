"""AI layer - model definition, inference, training."""

from __future__ import annotations

from typing import Any

__all__ = ["TrainingWorker"]


def __getattr__(name: str) -> Any:
    if name == "TrainingWorker":
        from scann.ai.training_worker import TrainingWorker

        return TrainingWorker
    raise AttributeError(name)
