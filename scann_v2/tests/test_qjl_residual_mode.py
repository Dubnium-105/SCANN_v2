from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scann.experiments import legacy_runner
from scann.experiments.packed_kv import pack_kv_per_head, pack_tensor_4bit, unpack_kv_block, unpack_tensor_4bit


def test_pack_tensor_4bit_qjl_sign_norm_adds_residual_metadata():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 5, 16)

    packed_none = pack_tensor_4bit(x, group_size=8, token_axis=-2, residual_mode="none")
    packed_residual = pack_tensor_4bit(
        x,
        group_size=8,
        token_axis=-2,
        residual_mode="qjl_sign_norm",
        qjl_dim=6,
    )
    restored = unpack_tensor_4bit(packed_residual)

    assert packed_residual.residual_mode == "qjl_sign_norm"
    assert packed_residual.qjl_dim == 6
    assert packed_residual.residual_norms is not None
    assert packed_residual.residual_signs is not None
    assert packed_residual.residual_signs.dtype == torch.uint8
    assert packed_residual.storage_size_bytes() > packed_none.storage_size_bytes()
    assert tuple(restored.shape) == tuple(x.shape)
    assert torch.isfinite(restored).all()


def test_pack_tensor_4bit_qjl_sign_norm_requires_positive_qjl_dim():
    x = torch.randn(1, 2, 3, 8)

    try:
        pack_tensor_4bit(x, group_size=4, token_axis=-2, residual_mode="qjl_sign_norm", qjl_dim=0)
    except ValueError as exc:
        assert "qjl_dim" in str(exc)
    else:
        raise AssertionError("Expected qjl_sign_norm to require qjl_dim > 0")


def test_pack_kv_per_head_supports_qjl_residual_block_decode():
    torch.manual_seed(1)
    key = torch.randn(1, 4, 6, 8)
    value = torch.randn(1, 4, 6, 8)

    packed_kv = pack_kv_per_head(
        key,
        value,
        group_size=4,
        token_axis=-2,
        residual_mode="qjl_sign_norm",
        qjl_dim=4,
    )
    key_block, value_block = unpack_kv_block(packed_kv, start=1, end=4, dtype=torch.float32)

    assert packed_kv.packed_k.residual_mode == "qjl_sign_norm"
    assert packed_kv.packed_v.qjl_dim == 4
    assert tuple(key_block.shape) == (1, 4, 3, 8)
    assert tuple(value_block.shape) == (1, 4, 3, 8)
    assert torch.isfinite(key_block).all()
    assert torch.isfinite(value_block).all()


def test_vit_packed_kv_attention_model_runs_with_qjl_residual_mode():
    torch.manual_seed(2)
    base_model = legacy_runner.create_experiment_model("vit_b_16", pretrained=False, image_size=224).eval()
    packed_model = legacy_runner.create_vit_packed_kv_attention_model(
        base_model,
        layer_selector="first_n",
        count=2,
        group_size=8,
        block_size=4,
        quantize_k=True,
        quantize_v=True,
        preserve_cls_token=True,
        residual_mode="qjl_sign_norm",
        qjl_dim=8,
    ).eval()
    x = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = packed_model(x)

    config = getattr(packed_model, "_vit_packed_kv_attention_config")
    assert tuple(output.shape) == (1, 2)
    assert torch.isfinite(output).all()
    assert config.residual_mode == "qjl_sign_norm"
    assert config.qjl_dim == 8


def test_apply_attention_compression_rejects_qjl_residual_without_dimension():
    config = legacy_runner.LegacyExperimentConfig(
        experiment_name="vit_bad_qjl",
        model_name="vit_b_16",
        image_size=224,
        attention_compression_mode="vit_packed_kv",
        residual_mode="qjl_sign_norm",
        qjl_dim=0,
    )

    try:
        legacy_runner._apply_attention_compression_from_config(torch.nn.Identity(), config, for_training=False)
    except ValueError as exc:
        assert "qjl_dim > 0" in str(exc)
    else:
        raise AssertionError("Expected qjl_sign_norm configuration to require qjl_dim > 0")
