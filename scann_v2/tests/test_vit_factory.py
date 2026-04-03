from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scann.experiments import legacy_runner


class _FakeHead:
    def __init__(self, in_features: int = 32) -> None:
        self.in_features = in_features


class _FakeHeads:
    def __init__(self, in_features: int = 32) -> None:
        self.head = _FakeHead(in_features=in_features)


class _FakeModel:
    def __init__(self, image_size: int) -> None:
        self.image_size = image_size
        self.heads = _FakeHeads()
        self.loaded_state_dict = None
        self.strict = None

    def load_state_dict(self, state_dict, strict: bool = False):
        self.loaded_state_dict = state_dict
        self.strict = strict


class _FakeWeights:
    def __init__(self, *, min_size: int) -> None:
        self.meta = {"min_size": (min_size, min_size)}

    def get_state_dict(self, progress: bool = True):
        return {"encoder.pos_embedding": "state"}


def test_create_experiment_model_supports_vit_h_14_without_pretrained(monkeypatch):
    calls: list[tuple[str, object, int]] = []

    def _fake_vit_h_14(*, weights=None, image_size=224):
        calls.append(("vit_h_14", weights, image_size))
        return _FakeModel(image_size=image_size)

    monkeypatch.setattr(legacy_runner.models, "vit_h_14", _fake_vit_h_14)
    monkeypatch.setattr(
        legacy_runner.models,
        "ViT_H_14_Weights",
        SimpleNamespace(DEFAULT=_FakeWeights(min_size=518)),
    )

    model = legacy_runner.create_experiment_model(
        "vit_huge",
        pretrained=False,
        image_size=518,
    )

    assert model.image_size == 518
    assert calls == [("vit_h_14", None, 518)]


def test_create_experiment_model_interpolates_vit_h_14_weights_for_custom_size(monkeypatch):
    builder_calls: list[tuple[str, object, int]] = []
    interpolate_calls: list[tuple[int, int, object]] = []

    def _fake_vit_h_14(*, weights=None, image_size=224):
        builder_calls.append(("vit_h_14", weights, image_size))
        return _FakeModel(image_size=image_size)

    def _fake_interpolate(*, weights, image_size, patch_size):
        interpolate_calls.append((image_size, patch_size, weights))
        return {"encoder.pos_embedding": "interpolated"}

    fake_weights = _FakeWeights(min_size=518)
    monkeypatch.setattr(legacy_runner.models, "vit_h_14", _fake_vit_h_14)
    monkeypatch.setattr(
        legacy_runner.models,
        "ViT_H_14_Weights",
        SimpleNamespace(DEFAULT=fake_weights),
    )
    monkeypatch.setattr(legacy_runner, "_interpolate_vit_state_dict", _fake_interpolate)

    model = legacy_runner.create_experiment_model(
        "vit_h_14",
        pretrained=True,
        image_size=640,
    )

    assert model.image_size == 640
    assert builder_calls == [("vit_h_14", None, 640)]
    assert interpolate_calls == [(640, 14, fake_weights)]
    assert model.loaded_state_dict == {"encoder.pos_embedding": "interpolated"}
    assert model.strict is False
