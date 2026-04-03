from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scann.experiments.packed_kv import pack_kv_per_head, unpack_tensor_4bit
from scann.experiments.vit_attention_compression import (
    decode_kv_block,
    online_softmax_update,
    streaming_packed_attention,
)


def _dense_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *, scale: float) -> torch.Tensor:
    logits = torch.matmul(query, key.transpose(-2, -1)) * scale
    weights = torch.softmax(logits, dim=-1)
    return torch.matmul(weights, value)


def test_decode_kv_block_matches_full_unpack_slice():
    torch.manual_seed(0)
    key = torch.randn(2, 3, 7, 8)
    value = torch.randn(2, 3, 7, 8)
    packed_kv = pack_kv_per_head(key, value, group_size=4, token_axis=-2)

    full_key = unpack_tensor_4bit(packed_kv.packed_k)
    full_value = unpack_tensor_4bit(packed_kv.packed_v)
    key_block, value_block = decode_kv_block(packed_kv, start=2, end=6, dtype=torch.float32)

    assert torch.allclose(key_block, full_key[:, :, 2:6, :], atol=1e-6, rtol=0.0)
    assert torch.allclose(value_block, full_value[:, :, 2:6, :], atol=1e-6, rtol=0.0)


def test_online_softmax_update_matches_dense_two_block_reference():
    torch.manual_seed(1)
    query = torch.randn(1, 2, 4, 8)
    key = torch.randn(1, 2, 6, 8)
    value = torch.randn(1, 2, 6, 8)
    scale = query.shape[-1] ** -0.5

    logits = torch.matmul(query, key.transpose(-2, -1)) * scale
    expected = torch.matmul(torch.softmax(logits, dim=-1), value)

    first_logits = logits[..., :3]
    second_logits = logits[..., 3:]
    first_value = value[..., :3, :]
    second_value = value[..., 3:, :]

    running_max = torch.full((*query.shape[:-1], 1), -torch.inf, dtype=query.dtype)
    running_denom = torch.zeros_like(running_max)
    running_weighted = torch.zeros_like(query)

    running_max, running_denom, running_weighted = online_softmax_update(
        logits_block=first_logits,
        value_block=first_value,
        running_max=running_max,
        running_denom=running_denom,
        running_weighted_values=running_weighted,
    )
    running_max, running_denom, running_weighted = online_softmax_update(
        logits_block=second_logits,
        value_block=second_value,
        running_max=running_max,
        running_denom=running_denom,
        running_weighted_values=running_weighted,
    )

    actual = running_weighted / running_denom.clamp_min(1e-12)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_streaming_packed_attention_matches_dense_attention_on_unpacked_reference():
    torch.manual_seed(2)
    query = torch.randn(2, 4, 5, 8)
    key = torch.randn(2, 4, 9, 8)
    value = torch.randn(2, 4, 9, 8)
    packed_kv = pack_kv_per_head(key, value, group_size=4, token_axis=-2)
    unpacked_key = unpack_tensor_4bit(packed_kv.packed_k)
    unpacked_value = unpack_tensor_4bit(packed_kv.packed_v)
    scale = query.shape[-1] ** -0.5

    expected = _dense_attention(query, unpacked_key, unpacked_value, scale=scale)
    actual = streaming_packed_attention(
        query,
        packed_kv,
        block_size=3,
        scale=scale,
        compute_dtype=torch.float32,
        output_dtype=torch.float32,
    )

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_streaming_packed_attention_supports_single_block_fallback():
    torch.manual_seed(3)
    query = torch.randn(1, 2, 3, 4, dtype=torch.float32)
    key = torch.randn(1, 2, 5, 4, dtype=torch.float32)
    value = torch.randn(1, 2, 5, 4, dtype=torch.float32)
    packed_kv = pack_kv_per_head(key, value, group_size=4, token_axis=-2)

    output = streaming_packed_attention(query, packed_kv, block_size=64)

    assert tuple(output.shape) == (1, 2, 3, 4)
    assert output.dtype == query.dtype
