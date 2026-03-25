from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

from scann.ai import device_utils
from scann.ai.device_utils import AcceleratorInfo


def _fake_device(spec: str):
    return SimpleNamespace(spec=spec, type=spec.split(":", 1)[0])


def test_canonicalize_device_request_maps_domestic_aliases():
    assert device_utils.canonicalize_device_request("ascend") == "npu"
    assert device_utils.canonicalize_device_request("cambricon") == "mlu"
    assert device_utils.canonicalize_device_request("mthreads") == "musa"


def test_resolve_device_auto_can_select_domestic_accelerator(monkeypatch):
    monkeypatch.setattr(device_utils.torch, "device", _fake_device)

    def fake_info(spec_or_key):
        key = getattr(spec_or_key, "key", spec_or_key)
        if key == "cuda":
            return AcceleratorInfo(key="cuda", label="CUDA", available=False)
        if key == "npu":
            return AcceleratorInfo(key="npu", label="Ascend NPU", available=True, device_count=1)
        return AcceleratorInfo(key=str(key), label=str(key), available=False)

    monkeypatch.setattr(device_utils, "get_accelerator_info", fake_info)

    resolved = device_utils.resolve_device("auto")

    assert resolved.backend == "npu"
    assert resolved.resolved.spec == "npu:0"
    assert resolved.used_fallback is False


def test_resolve_device_falls_back_to_cpu_when_requested_backend_unavailable(monkeypatch):
    monkeypatch.setattr(device_utils.torch, "device", _fake_device)
    monkeypatch.setattr(
        device_utils,
        "get_accelerator_info",
        lambda spec_or_key: AcceleratorInfo(
            key=getattr(spec_or_key, "key", str(spec_or_key)),
            label="Unavailable",
            available=False,
        ),
    )

    resolved = device_utils.resolve_device("mlu")

    assert resolved.backend == "cpu"
    assert resolved.resolved.spec == "cpu"
    assert resolved.used_fallback is True


def test_get_mixed_precision_context_prefers_backend_specific_autocast(monkeypatch):
    events: list[str] = []

    @contextmanager
    def fake_autocast():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    monkeypatch.setattr(
        device_utils.torch,
        "npu",
        SimpleNamespace(amp=SimpleNamespace(autocast=fake_autocast)),
        raising=False,
    )

    with device_utils.get_mixed_precision_context(SimpleNamespace(type="npu"), enabled=True):
        events.append("body")

    assert events == ["enter", "body", "exit"]


def test_device_choice_lists_include_domestic_backends():
    settings_values = {value for _label, value in device_utils.get_settings_device_choices()}
    training_labels = [label for label, _value in device_utils.get_training_device_choices()]

    assert {"npu", "mlu", "musa"}.issubset(settings_values)
    assert any("NPU" in label for label in training_labels)
    assert any("MLU" in label for label in training_labels)
