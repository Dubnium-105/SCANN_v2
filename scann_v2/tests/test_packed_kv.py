from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scann.experiments.packed_kv import (
    pack_kv_per_head,
    pack_tensor_4bit,
    unpack_kv_block,
    unpack_tensor_4bit,
    unpack_tensor_4bit_block,
)


def test_pack_tensor_4bit_uses_uint8_codes_and_preserves_shape():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 5, 16)

    packed = pack_tensor_4bit(x, group_size=8, token_axis=-2)
    restored = unpack_tensor_4bit(packed)

    assert packed.packed_codes.dtype == torch.uint8
    assert tuple(restored.shape) == tuple(x.shape)
    assert packed.storage_size_bytes() < (x.numel() * x.element_size())
    assert torch.mean(torch.abs(restored - x)).item() < 0.12


def test_pack_tensor_4bit_supports_last_dim_padding():
    torch.manual_seed(1)
    x = torch.randn(2, 4, 3, 10)

    packed = pack_tensor_4bit(x, group_size=8, token_axis=-2)
    restored = unpack_tensor_4bit(packed)

    assert packed.padded_shape[-1] == 16
    assert tuple(restored.shape) == (2, 4, 3, 10)
    assert torch.max(torch.abs(restored - x)).item() < 0.5


def test_unpack_tensor_4bit_block_matches_slice_of_full_unpack():
    torch.manual_seed(2)
    x = torch.randn(2, 2, 7, 12)
    packed = pack_tensor_4bit(x, group_size=4, token_axis=-2)

    full = unpack_tensor_4bit(packed)
    block = unpack_tensor_4bit_block(packed, start=2, end=5, dtype=torch.float32)

    assert tuple(block.shape) == (2, 2, 3, 12)
    assert torch.allclose(block, full[:, :, 2:5, :], atol=1e-6, rtol=0.0)


def test_pack_kv_per_head_and_unpack_kv_block_round_trip_blockwise():
    torch.manual_seed(3)
    key = torch.randn(1, 4, 6, 8)
    value = torch.randn(1, 4, 6, 8)

    packed_kv = pack_kv_per_head(key, value, group_size=4, token_axis=-2)
    key_block, value_block = unpack_kv_block(packed_kv, start=1, end=4, dtype=torch.float32)

    full_key = unpack_tensor_4bit(packed_kv.packed_k)
    full_value = unpack_tensor_4bit(packed_kv.packed_v)

    assert tuple(key_block.shape) == (1, 4, 3, 8)
    assert tuple(value_block.shape) == (1, 4, 3, 8)
    assert torch.allclose(key_block, full_key[:, :, 1:4, :], atol=1e-6, rtol=0.0)
    assert torch.allclose(value_block, full_value[:, :, 1:4, :], atol=1e-6, rtol=0.0)
