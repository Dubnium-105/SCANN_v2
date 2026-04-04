"""Compute device compatibility helpers.

This module centralizes accelerator discovery, device resolution, and
mixed-precision context selection so the rest of the codebase can stay
device-agnostic.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
import importlib
from typing import Iterator

import torch


@dataclass(frozen=True)
class AcceleratorSpec:
    key: str
    label: str
    aliases: tuple[str, ...]
    extension_modules: tuple[str, ...] = ()
    torch_attr: str | None = None
    default_device: str | None = None


@dataclass(frozen=True)
class AcceleratorInfo:
    key: str
    label: str
    available: bool
    device_count: int = 0
    current_name: str = ""
    version: str = ""
    reason: str = ""


@dataclass(frozen=True)
class ResolvedDevice:
    requested: str
    resolved: torch.device
    backend: str
    label: str
    used_fallback: bool = False
    message: str = ""


ACCELERATOR_SPECS: tuple[AcceleratorSpec, ...] = (
    AcceleratorSpec(
        key="cuda",
        label="CUDA / NVIDIA",
        aliases=("cuda", "cuda:0", "gpu", "nvidia"),
        torch_attr="cuda",
        default_device="cuda:0",
    ),
    AcceleratorSpec(
        key="dcu",
        label="DCU / Hygon",
        aliases=("dcu", "dcu:0", "hygon", "k100", "k100_ai", "dcu-k100"),
        extension_modules=("torch_dcu",),
        torch_attr="cuda",
        default_device="cuda:0",
    ),
    AcceleratorSpec(
        key="npu",
        label="Ascend NPU",
        aliases=("npu", "npu:0", "ascend", "huawei", "huawei-ascend"),
        extension_modules=("torch_npu",),
        torch_attr="npu",
        default_device="npu:0",
    ),
    AcceleratorSpec(
        key="mlu",
        label="Cambricon MLU",
        aliases=("mlu", "mlu:0", "cambricon"),
        extension_modules=("torch_mlu",),
        torch_attr="mlu",
        default_device="mlu:0",
    ),
    AcceleratorSpec(
        key="musa",
        label="MUSA",
        aliases=("musa", "musa:0", "moore-threads", "mthreads"),
        extension_modules=("torch_musa",),
        torch_attr="musa",
        default_device="musa:0",
    ),
    AcceleratorSpec(
        key="xpu",
        label="XPU",
        aliases=("xpu", "xpu:0"),
        torch_attr="xpu",
        default_device="xpu:0",
    ),
    AcceleratorSpec(
        key="mps",
        label="MPS",
        aliases=("mps",),
        torch_attr="mps",
        default_device="mps",
    ),
)

AUTO_DEVICE_PRIORITY: tuple[str, ...] = ("cuda", "dcu", "npu", "mlu", "musa", "xpu", "mps")
KNOWN_DEVICE_VALUES: tuple[str, ...] = ("auto", "cpu") + tuple(
    spec.key for spec in ACCELERATOR_SPECS
)

_SPEC_BY_KEY = {spec.key: spec for spec in ACCELERATOR_SPECS}
_SPEC_BY_ALIAS = {
    alias.lower(): spec for spec in ACCELERATOR_SPECS for alias in spec.aliases
}


def _iter_specs() -> Iterator[AcceleratorSpec]:
    return iter(ACCELERATOR_SPECS)


@lru_cache(maxsize=None)
def _try_import_optional_module(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def _ensure_extension_loaded(spec: AcceleratorSpec) -> None:
    for module_name in spec.extension_modules:
        _try_import_optional_module(module_name)


def _get_backend_module(spec: AcceleratorSpec):
    _ensure_extension_loaded(spec)
    if spec.torch_attr is None:
        return None
    return getattr(torch, spec.torch_attr, None)


def _detect_version(spec: AcceleratorSpec) -> str:
    if spec.key == "cuda":
        return str(getattr(torch.version, "cuda", "") or "")
    if spec.key == "mps":
        return ""
    backend_module = _get_backend_module(spec)
    version = getattr(backend_module, "__version__", "") if backend_module is not None else ""
    return str(version or "")


def _detect_device_count(backend_module) -> int:
    if backend_module is None:
        return 0
    device_count = getattr(backend_module, "device_count", None)
    if callable(device_count):
        try:
            return int(device_count())
        except Exception:
            return 0
    return 0


def _detect_current_name(backend_module) -> str:
    if backend_module is None:
        return ""

    current_device = getattr(backend_module, "current_device", None)
    get_device_name = getattr(backend_module, "get_device_name", None)
    if not callable(current_device) or not callable(get_device_name):
        return ""
    try:
        return str(get_device_name(current_device()) or "")
    except Exception:
        return ""


def get_accelerator_info(spec_or_key: AcceleratorSpec | str) -> AcceleratorInfo:
    spec = _SPEC_BY_KEY[str(spec_or_key)] if isinstance(spec_or_key, str) else spec_or_key
    backend_module = _get_backend_module(spec)
    if backend_module is None:
        return AcceleratorInfo(
            key=spec.key,
            label=spec.label,
            available=False,
            reason="backend module unavailable",
        )

    is_available = getattr(backend_module, "is_available", None)
    if not callable(is_available):
        return AcceleratorInfo(
            key=spec.key,
            label=spec.label,
            available=False,
            reason="is_available() not exposed",
        )

    try:
        available = bool(is_available())
    except Exception:
        return AcceleratorInfo(
            key=spec.key,
            label=spec.label,
            available=False,
            reason="availability probe failed",
        )

    if not available:
        return AcceleratorInfo(
            key=spec.key,
            label=spec.label,
            available=False,
            reason="runtime reports unavailable",
            version=_detect_version(spec),
        )

    return AcceleratorInfo(
        key=spec.key,
        label=spec.label,
        available=True,
        device_count=_detect_device_count(backend_module),
        current_name=_detect_current_name(backend_module),
        version=_detect_version(spec),
    )


def list_accelerators() -> list[AcceleratorInfo]:
    return [get_accelerator_info(spec) for spec in _iter_specs()]


def canonicalize_device_request(requested: str | None) -> str:
    raw = str(requested or "auto").strip()
    if not raw:
        return "auto"

    lowered = raw.lower()
    if lowered in {"auto", "cpu"}:
        return lowered

    head = lowered.split(":", 1)[0]
    spec = _SPEC_BY_ALIAS.get(lowered) or _SPEC_BY_ALIAS.get(head)
    if spec is not None:
        if ":" in lowered:
            return lowered
        return spec.key

    return raw


def get_device_label(device_value: str) -> str:
    normalized = canonicalize_device_request(device_value)
    if normalized == "auto":
        return "Auto"
    if normalized == "cpu":
        return "CPU"

    head = normalized.split(":", 1)[0].lower()
    spec = _SPEC_BY_ALIAS.get(normalized.lower()) or _SPEC_BY_ALIAS.get(head)
    if spec is not None:
        return spec.label
    return normalized


def resolve_device(requested: str | None = "auto") -> ResolvedDevice:
    normalized = canonicalize_device_request(requested)

    if normalized == "auto":
        for backend_key in AUTO_DEVICE_PRIORITY:
            info = get_accelerator_info(backend_key)
            if not info.available:
                continue
            spec = _SPEC_BY_KEY[backend_key]
            target = spec.default_device or backend_key
            try:
                device = torch.device(target)
            except Exception:
                continue
            return ResolvedDevice(
                requested="auto",
                resolved=device,
                backend=backend_key,
                label=info.label,
                message=f"auto-selected {device}",
            )
        return ResolvedDevice(
            requested="auto",
            resolved=torch.device("cpu"),
            backend="cpu",
            label="CPU",
            message="no accelerator available; using cpu",
        )

    if normalized == "cpu":
        return ResolvedDevice(
            requested=normalized,
            resolved=torch.device("cpu"),
            backend="cpu",
            label="CPU",
            message="cpu requested",
        )

    base = normalized.split(":", 1)[0].lower()
    spec = _SPEC_BY_ALIAS.get(normalized.lower()) or _SPEC_BY_ALIAS.get(base)
    if spec is not None:
        info = get_accelerator_info(spec)
        if info.available:
            target = normalized if ":" in normalized else (spec.default_device or spec.key)
            try:
                return ResolvedDevice(
                    requested=str(requested or normalized),
                    resolved=torch.device(target),
                    backend=spec.key,
                    label=info.label,
                    message=f"using {target}",
                )
            except Exception:
                return ResolvedDevice(
                    requested=str(requested or normalized),
                    resolved=torch.device("cpu"),
                    backend="cpu",
                    label="CPU",
                    used_fallback=True,
                    message=f"{spec.label} reported available but torch.device('{target}') failed; using cpu",
                )
        return ResolvedDevice(
            requested=str(requested or normalized),
            resolved=torch.device("cpu"),
            backend="cpu",
            label="CPU",
            used_fallback=True,
            message=f"{spec.label} unavailable; using cpu",
        )

    try:
        device = torch.device(str(requested))
    except Exception:
        return ResolvedDevice(
            requested=str(requested or normalized),
            resolved=torch.device("cpu"),
            backend="cpu",
            label="CPU",
            used_fallback=True,
            message=f"unrecognized device '{requested}'; using cpu",
        )

    return ResolvedDevice(
        requested=str(requested),
        resolved=device,
        backend=device.type,
        label=get_device_label(device.type),
        message=f"using custom device {device}",
    )


def get_mixed_precision_context(device: torch.device, enabled: bool = True):
    if not enabled:
        return nullcontext()

    device_type = getattr(device, "type", "cpu")
    if device_type == "cpu":
        return nullcontext()

    backend_module = getattr(torch, device_type, None)
    amp_module = getattr(backend_module, "amp", None)
    autocast = getattr(amp_module, "autocast", None)
    if callable(autocast):
        try:
            return autocast()
        except TypeError:
            return autocast(enabled=True)
        except Exception:
            pass

    amp = getattr(torch, "amp", None)
    generic_autocast = getattr(amp, "autocast", None)
    if callable(generic_autocast):
        try:
            return generic_autocast(device_type=device_type, enabled=True)
        except TypeError:
            try:
                return generic_autocast(device_type, enabled=True)
            except Exception:
                pass
        except Exception:
            pass

    if device_type == "cuda":
        cuda_amp = getattr(getattr(torch, "cuda", None), "amp", None)
        cuda_autocast = getattr(cuda_amp, "autocast", None)
        if callable(cuda_autocast):
            return cuda_autocast()

    return nullcontext()


def get_settings_device_choices(current_value: str | None = None) -> list[tuple[str, str]]:
    current = canonicalize_device_request(current_value)
    choices = [("auto", "auto"), ("cpu", "cpu")]
    for spec in _iter_specs():
        choices.append((spec.key, spec.key))

    if current not in {value for _label, value in choices}:
        choices.append((current, current))
    return choices


def get_training_device_choices(current_value: str | None = None) -> list[tuple[str, str]]:
    current = canonicalize_device_request(current_value)
    choices = [
        ("Auto (自动选择加速器)", "auto"),
        ("CUDA", "cuda"),
        ("DCU (Hygon K100)", "dcu"),
        ("NPU (Ascend)", "npu"),
        ("MLU (Cambricon)", "mlu"),
        ("MUSA", "musa"),
        ("CPU", "cpu"),
    ]

    extra_values = ("xpu", "mps")
    for value in extra_values:
        choices.append((get_device_label(value), value))

    if current not in {value for _label, value in choices}:
        choices.append((current, current))
    return choices


def format_accelerator_status() -> str:
    accelerators = [info for info in list_accelerators() if info.available]
    if accelerators:
        primary = accelerators[0]
        parts = [f"Available accelerator: {primary.label}"]
        if primary.version:
            parts.append(f"  - Version: {primary.version}")
        if primary.device_count:
            parts.append(f"  - Device count: {primary.device_count}")
        if primary.current_name:
            parts.append(f"  - Current: {primary.current_name}")
        if len(accelerators) > 1:
            others = ", ".join(info.label for info in accelerators[1:])
            parts.append(f"  - Also available: {others}")
        return "\n".join(parts)

    known = " / ".join(spec.label for spec in _iter_specs())
    return (
        "No accelerator detected\n"
        "  - Falling back to CPU\n"
        f"  - Probed backends: {known}"
    )
