"""Packed 4-bit KV utilities for ViT attention experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

_QJL_PROJECTION_SEED = 1729


@dataclass(frozen=True)
class PackedTensor4Bit:
    """Packed 4-bit tensor representation grouped along the last dimension."""

    packed_codes: torch.Tensor
    scales: torch.Tensor
    original_shape: Tuple[int, ...]
    padded_shape: Tuple[int, ...]
    group_size: int
    groups_per_token: int
    token_axis: int
    residual_mode: str = "none"
    qjl_dim: int = 0
    residual_norms: torch.Tensor | None = None
    residual_signs: torch.Tensor | None = None

    def storage_size_bytes(self) -> int:
        total = (self.packed_codes.element_size() * self.packed_codes.numel()) + (
            self.scales.element_size() * self.scales.numel()
        )
        if self.residual_norms is not None:
            total += self.residual_norms.element_size() * self.residual_norms.numel()
        if self.residual_signs is not None:
            total += self.residual_signs.element_size() * self.residual_signs.numel()
        return int(total)


@dataclass(frozen=True)
class PackedKV4Bit:
    """Packed 4-bit K/V pair for blockwise attention experiments."""

    packed_k: PackedTensor4Bit
    packed_v: PackedTensor4Bit

    @property
    def token_axis(self) -> int:
        return int(self.packed_k.token_axis)

    @property
    def group_size(self) -> int:
        return int(self.packed_k.group_size)

    def storage_size_bytes(self) -> int:
        return int(self.packed_k.storage_size_bytes() + self.packed_v.storage_size_bytes())


def _canonical_token_axis(ndim: int, token_axis: int) -> int:
    resolved = int(token_axis)
    if resolved < 0:
        resolved += int(ndim)
    if resolved < 0 or resolved >= int(ndim):
        raise ValueError(f"token_axis {token_axis} is out of range for ndim={ndim}")
    return resolved


def _validate_group_size(group_size: int) -> int:
    resolved = int(group_size)
    if resolved <= 0:
        raise ValueError("group_size must be positive")
    return resolved


def _resolve_residual_mode(residual_mode: str) -> str:
    normalized = str(residual_mode).strip().lower()
    if normalized in {"", "none"}:
        return "none"
    if normalized == "qjl_sign_norm":
        return normalized
    raise ValueError(f"Unsupported residual_mode: {residual_mode}")


def _resolve_qjl_dim(qjl_dim: int, *, group_size: int, residual_mode: str) -> int:
    del group_size
    if residual_mode == "none":
        return 0
    resolved = int(qjl_dim)
    if resolved <= 0:
        raise ValueError("qjl_dim must be positive when residual_mode='qjl_sign_norm'")
    return resolved


def _pad_last_dim(x: torch.Tensor, *, group_size: int) -> tuple[torch.Tensor, int]:
    last_dim = int(x.shape[-1])
    padded_last_dim = ((last_dim + group_size - 1) // group_size) * group_size
    pad_amount = padded_last_dim - last_dim
    if pad_amount > 0:
        x = F.pad(x, (0, pad_amount))
    return x, padded_last_dim


def _pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    if codes.dtype != torch.uint8:
        raise TypeError("codes must use torch.uint8 before nibble packing")

    if int(codes.shape[-1]) % 2 != 0:
        codes = F.pad(codes, (0, 1), value=0)

    low = codes[..., 0::2]
    high = codes[..., 1::2]
    return low | (high << 4)


def _pack_bits(bits: torch.Tensor) -> torch.Tensor:
    if bits.dtype != torch.uint8:
        raise TypeError("bits must use torch.uint8 before bit packing")

    if int(bits.shape[-1]) % 8 != 0:
        bits = F.pad(bits, (0, 8 - (int(bits.shape[-1]) % 8)), value=0)

    shifts = torch.arange(8, device=bits.device, dtype=torch.int64)
    grouped = bits.reshape(*bits.shape[:-1], -1, 8).to(torch.int64)
    packed = torch.sum(grouped * (1 << shifts), dim=-1)
    return packed.to(torch.uint8)


def _unpack_nibbles(packed_codes: torch.Tensor, *, group_size: int) -> torch.Tensor:
    low = packed_codes & 0x0F
    high = (packed_codes >> 4) & 0x0F
    codes = torch.stack((low, high), dim=-1).reshape(*packed_codes.shape[:-1], -1)
    return codes[..., :group_size]


def _unpack_bits(packed_bits: torch.Tensor, *, bit_count: int) -> torch.Tensor:
    shifts = torch.arange(8, device=packed_bits.device, dtype=torch.int64)
    expanded = ((packed_bits.to(torch.int64).unsqueeze(-1) >> shifts) & 0x01).to(torch.uint8)
    bits = expanded.reshape(*packed_bits.shape[:-1], -1)
    return bits[..., :bit_count]


def _qjl_projection(group_size: int, qjl_dim: int, *, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(_QJL_PROJECTION_SEED) + (int(group_size) * 31) + int(qjl_dim))
    projection = torch.randn((int(qjl_dim), int(group_size)), generator=generator, dtype=torch.float32)
    projection = F.normalize(projection, dim=-1)
    return projection.to(device=device)


def pack_tensor_4bit(
    tensor: torch.Tensor,
    *,
    group_size: int = 32,
    token_axis: int = -2,
    residual_mode: str = "none",
    qjl_dim: int = 0,
) -> PackedTensor4Bit:
    """Symmetrically quantize a tensor into packed 4-bit groups on the last dim."""

    if tensor.ndim < 2:
        raise ValueError("pack_tensor_4bit expects a tensor with at least 2 dimensions")

    resolved_group_size = _validate_group_size(group_size)
    resolved_token_axis = _canonical_token_axis(tensor.ndim, token_axis)
    resolved_residual_mode = _resolve_residual_mode(residual_mode)
    resolved_qjl_dim = _resolve_qjl_dim(
        qjl_dim,
        group_size=resolved_group_size,
        residual_mode=resolved_residual_mode,
    )
    x = tensor.detach().to(torch.float32)
    original_shape = tuple(int(dim) for dim in x.shape)
    x_padded, padded_last_dim = _pad_last_dim(x, group_size=resolved_group_size)
    groups_per_token = padded_last_dim // resolved_group_size

    grouped = x_padded.reshape(*x_padded.shape[:-1], groups_per_token, resolved_group_size)
    scales = grouped.abs().amax(dim=-1).clamp_min(1e-8) / 7.0
    quantized = torch.round(grouped / scales.unsqueeze(-1)).clamp(-8, 7).to(torch.int16)
    codes = (quantized + 8).to(torch.uint8)
    packed_codes = _pack_nibbles(codes)
    residual_norms = None
    residual_signs = None
    if resolved_residual_mode == "qjl_sign_norm":
        reconstructed = quantized.to(torch.float32) * scales.unsqueeze(-1)
        residual = grouped - reconstructed
        residual_norms = residual.norm(dim=-1).contiguous()
        projection = _qjl_projection(resolved_group_size, resolved_qjl_dim, device=residual.device)
        residual_sketch = torch.matmul(residual, projection.transpose(0, 1))
        residual_signs = _pack_bits((residual_sketch >= 0).to(torch.uint8)).contiguous()

    return PackedTensor4Bit(
        packed_codes=packed_codes.contiguous(),
        scales=scales.contiguous(),
        original_shape=original_shape,
        padded_shape=tuple(list(original_shape[:-1]) + [padded_last_dim]),
        group_size=resolved_group_size,
        groups_per_token=groups_per_token,
        token_axis=resolved_token_axis,
        residual_mode=resolved_residual_mode,
        qjl_dim=resolved_qjl_dim,
        residual_norms=residual_norms,
        residual_signs=residual_signs,
    )


def unpack_tensor_4bit(
    packed: PackedTensor4Bit,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reconstruct the full tensor from packed 4-bit storage."""

    codes = _unpack_nibbles(packed.packed_codes, group_size=int(packed.group_size))
    quantized = codes.to(torch.int16) - 8
    reconstructed = quantized.to(torch.float32) * packed.scales.unsqueeze(-1)
    if packed.residual_mode == "qjl_sign_norm":
        if packed.residual_norms is None or packed.residual_signs is None:
            raise ValueError("Packed tensor is missing residual metadata for residual_mode='qjl_sign_norm'")
        sign_bits = _unpack_bits(packed.residual_signs, bit_count=int(packed.qjl_dim))
        sign_values = (sign_bits.to(torch.float32) * 2.0) - 1.0
        projection = _qjl_projection(int(packed.group_size), int(packed.qjl_dim), device=reconstructed.device)
        residual_direction = torch.matmul(sign_values, projection)
        residual_direction = residual_direction / residual_direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        reconstructed = reconstructed + (residual_direction * packed.residual_norms.unsqueeze(-1))
    reconstructed = reconstructed.reshape(*packed.padded_shape)[..., : packed.original_shape[-1]]
    return reconstructed.to(dtype=dtype)


def slice_packed_tensor(
    packed: PackedTensor4Bit,
    *,
    start: int,
    end: int,
    axis: int | None = None,
) -> PackedTensor4Bit:
    """Slice a packed tensor along a non-group axis without unpacking."""

    resolved_axis = packed.token_axis if axis is None else _canonical_token_axis(len(packed.original_shape), axis)
    if resolved_axis == len(packed.original_shape) - 1:
        raise ValueError("Slicing along the packed last dimension is not supported; use unpack_tensor_4bit")

    start_index = max(0, int(start))
    end_index = min(int(packed.original_shape[resolved_axis]), int(end))
    if end_index < start_index:
        end_index = start_index

    packed_codes = packed.packed_codes.narrow(resolved_axis, start_index, end_index - start_index).contiguous()
    scales = packed.scales.narrow(resolved_axis, start_index, end_index - start_index).contiguous()
    original_shape = list(packed.original_shape)
    padded_shape = list(packed.padded_shape)
    original_shape[resolved_axis] = end_index - start_index
    padded_shape[resolved_axis] = end_index - start_index

    return PackedTensor4Bit(
        packed_codes=packed_codes,
        scales=scales,
        original_shape=tuple(original_shape),
        padded_shape=tuple(padded_shape),
        group_size=int(packed.group_size),
        groups_per_token=int(packed.groups_per_token),
        token_axis=int(packed.token_axis if axis is None else resolved_axis),
        residual_mode=str(packed.residual_mode),
        qjl_dim=int(packed.qjl_dim),
        residual_norms=(
            None
            if packed.residual_norms is None
            else packed.residual_norms.narrow(resolved_axis, start_index, end_index - start_index).contiguous()
        ),
        residual_signs=(
            None
            if packed.residual_signs is None
            else packed.residual_signs.narrow(resolved_axis, start_index, end_index - start_index).contiguous()
        ),
    )


def unpack_tensor_4bit_block(
    packed: PackedTensor4Bit,
    *,
    start: int,
    end: int,
    axis: int | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decode a slice of the packed tensor along the token axis."""

    sliced = slice_packed_tensor(packed, start=start, end=end, axis=axis)
    return unpack_tensor_4bit(sliced, dtype=dtype)


def pack_kv_per_head(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    group_size: int = 32,
    token_axis: int = -2,
    residual_mode: str = "none",
    qjl_dim: int = 0,
) -> PackedKV4Bit:
    """Pack K/V tensors that already expose a per-head layout such as [B, H, T, D]."""

    if tuple(key.shape) != tuple(value.shape):
        raise ValueError("key and value tensors must share the same shape")

    return PackedKV4Bit(
        packed_k=pack_tensor_4bit(
            key,
            group_size=group_size,
            token_axis=token_axis,
            residual_mode=residual_mode,
            qjl_dim=qjl_dim,
        ),
        packed_v=pack_tensor_4bit(
            value,
            group_size=group_size,
            token_axis=token_axis,
            residual_mode=residual_mode,
            qjl_dim=qjl_dim,
        ),
    )


def unpack_kv_block(
    packed_kv: PackedKV4Bit,
    *,
    start: int,
    end: int,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode a K/V block along the configured token axis."""

    return (
        unpack_tensor_4bit_block(
            packed_kv.packed_k,
            start=start,
            end=end,
            axis=packed_kv.token_axis,
            dtype=dtype,
        ),
        unpack_tensor_4bit_block(
            packed_kv.packed_v,
            start=start,
            end=end,
            axis=packed_kv.token_axis,
            dtype=dtype,
        ),
    )
