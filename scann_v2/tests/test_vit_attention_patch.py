from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scann.experiments import legacy_runner
from scann.experiments.vit_attention_compression import (
    ViTAttentionPatchConfig,
    create_vit_attention_compression_model,
    iter_vit_attention_modules,
    selected_vit_attention_specs,
)


def test_iter_vit_attention_modules_returns_encoder_attention_metadata():
    model = legacy_runner.create_experiment_model("vit_b_16", pretrained=False, image_size=224).eval()

    specs = iter_vit_attention_modules(model)

    assert len(specs) == 12
    assert specs[0].block_index == 0
    assert specs[0].attention_name.endswith("encoder_layer_0.self_attention")
    assert specs[0].num_heads == 12
    assert specs[0].head_dim == 64


def test_selected_vit_attention_specs_supports_first_last_middle_and_explicit():
    model = legacy_runner.create_experiment_model("vit_b_16", pretrained=False, image_size=224).eval()

    first_specs = selected_vit_attention_specs(
        model,
        config=ViTAttentionPatchConfig(layer_selector="first_n", count=2),
    )
    last_specs = selected_vit_attention_specs(
        model,
        config=ViTAttentionPatchConfig(layer_selector="last_n", count=2),
    )
    middle_specs = selected_vit_attention_specs(
        model,
        config=ViTAttentionPatchConfig(layer_selector="middle", count=4),
    )
    explicit_specs = selected_vit_attention_specs(
        model,
        config=ViTAttentionPatchConfig(layer_selector="explicit_indices", explicit_indices=(1, 3, 8)),
    )

    assert [spec.block_index for spec in first_specs] == [0, 1]
    assert [spec.block_index for spec in last_specs] == [10, 11]
    assert [spec.block_index for spec in middle_specs] == [4, 5, 6, 7]
    assert [spec.block_index for spec in explicit_specs] == [1, 3, 8]


def test_create_vit_attention_compression_model_patches_selected_layers_only():
    base_model = legacy_runner.create_experiment_model("vit_b_16", pretrained=False, image_size=224).eval()

    patched_model = create_vit_attention_compression_model(
        base_model,
        config=ViTAttentionPatchConfig(layer_selector="first_n", count=2),
    )

    assert getattr(patched_model, "_vit_attention_patched_indices") == [0, 1]
    assert hasattr(patched_model.encoder.layers[0], "_vit_attention_patch_original_forward")
    assert hasattr(patched_model.encoder.layers[1], "_vit_attention_patch_original_forward")
    assert not hasattr(patched_model.encoder.layers[2], "_vit_attention_patch_original_forward")


def test_patched_vit_forward_matches_baseline_with_passthrough_adapter():
    torch.manual_seed(0)
    base_model = legacy_runner.create_experiment_model("vit_b_16", pretrained=False, image_size=224).eval()
    reference_model = copy.deepcopy(base_model).eval()
    patched_model = legacy_runner.create_vit_attention_compression_model(
        base_model,
        layer_selector="all",
    ).eval()
    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        expected = reference_model(x)
        actual = patched_model(x)

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
