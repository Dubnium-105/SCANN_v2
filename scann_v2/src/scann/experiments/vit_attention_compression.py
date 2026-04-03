"""ViT attention patch scaffold for compression experiments."""

from __future__ import annotations

import copy
import types
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ViTAttentionModuleSpec:
    """Structured metadata for one ViT encoder self-attention module."""

    block_index: int
    block_name: str
    attention_name: str
    hidden_dim: int
    num_heads: int
    head_dim: int


@dataclass(frozen=True)
class ViTAttentionPatchConfig:
    """Selector config for choosing which ViT attention blocks to patch."""

    layer_selector: str = "all"
    count: int | None = None
    explicit_indices: Sequence[int] = ()


AttentionAdapter = Callable[[torch.Tensor, nn.Module, ViTAttentionModuleSpec], torch.Tensor]


def _iter_vit_encoder_blocks(model: nn.Module) -> list[tuple[int, str, nn.Module]]:
    encoder = getattr(model, "encoder", None)
    layers = getattr(encoder, "layers", None)
    if layers is None or not isinstance(layers, nn.Sequential):
        raise RuntimeError("Expected a torchvision ViT model with encoder.layers")

    blocks: list[tuple[int, str, nn.Module]] = []
    for index, (block_name, block) in enumerate(layers.named_children()):
        attention = getattr(block, "self_attention", None)
        if attention is None:
            continue
        blocks.append((index, block_name, block))

    if not blocks:
        raise RuntimeError("No ViT encoder self-attention blocks were found")
    return blocks


def iter_vit_attention_modules(model: nn.Module) -> list[ViTAttentionModuleSpec]:
    """Enumerate torchvision ViT encoder self-attention modules with metadata."""

    specs: list[ViTAttentionModuleSpec] = []
    for block_index, block_name, block in _iter_vit_encoder_blocks(model):
        attention = getattr(block, "self_attention")
        specs.append(
            ViTAttentionModuleSpec(
                block_index=block_index,
                block_name=f"encoder.layers.{block_name}",
                attention_name=f"encoder.layers.{block_name}.self_attention",
                hidden_dim=int(attention.embed_dim),
                num_heads=int(attention.num_heads),
                head_dim=int(attention.head_dim),
            )
        )
    return specs


def _resolve_layer_indices(total_layers: int, config: ViTAttentionPatchConfig) -> list[int]:
    if total_layers <= 0:
        return []

    selector = str(config.layer_selector).strip().lower()
    count = int(config.count) if config.count is not None else None

    if selector == "all":
        return list(range(total_layers))

    if selector == "explicit_indices":
        indices = sorted({int(index) for index in config.explicit_indices})
        if not indices:
            raise ValueError("explicit_indices selector requires at least one layer index")
        return [index for index in indices if 0 <= index < total_layers]

    resolved_count = max(1, count or 1)

    if selector == "first_n":
        return list(range(min(total_layers, resolved_count)))

    if selector == "last_n":
        start = max(0, total_layers - resolved_count)
        return list(range(start, total_layers))

    if selector == "middle":
        middle_count = min(total_layers, count or max(total_layers // 2, 1))
        start = max(0, (total_layers - middle_count) // 2)
        end = min(total_layers, start + middle_count)
        return list(range(start, end))

    raise ValueError(f"Unsupported ViT layer selector: {config.layer_selector}")


def vit_attention_passthrough_adapter(
    normalized_input: torch.Tensor,
    attention_module: nn.Module,
    module_spec: ViTAttentionModuleSpec,
) -> torch.Tensor:
    """Default adapter that preserves the original attention behavior."""

    del module_spec
    output, _ = attention_module(
        normalized_input,
        normalized_input,
        normalized_input,
        need_weights=False,
    )
    return output


def patch_vit_attention_modules(
    model: nn.Module,
    *,
    config: ViTAttentionPatchConfig | None = None,
    attention_adapter: AttentionAdapter | None = None,
) -> list[ViTAttentionModuleSpec]:
    """Patch selected ViT encoder blocks with a custom attention adapter."""

    patch_config = config or ViTAttentionPatchConfig()
    adapter = attention_adapter or vit_attention_passthrough_adapter
    specs = iter_vit_attention_modules(model)
    selected_indices = set(_resolve_layer_indices(len(specs), patch_config))
    patched_specs: list[ViTAttentionModuleSpec] = []

    for spec, (_, _, block) in zip(specs, _iter_vit_encoder_blocks(model)):
        if spec.block_index not in selected_indices:
            continue

        if not hasattr(block, "_vit_attention_patch_original_forward"):
            setattr(block, "_vit_attention_patch_original_forward", block.forward)

        def _forward(self, input_tensor, _adapter=adapter, _spec=spec):
            torch._assert(
                input_tensor.dim() == 3,
                f"Expected (batch_size, seq_length, hidden_dim) got {input_tensor.shape}",
            )
            x = self.ln_1(input_tensor)
            x = _adapter(x, self.self_attention, _spec)
            x = self.dropout(x)
            x = x + input_tensor

            y = self.ln_2(x)
            y = self.mlp(y)
            return x + y

        block.forward = types.MethodType(_forward, block)
        setattr(block, "_vit_attention_patch_spec", spec)
        setattr(block, "_vit_attention_patch_enabled", True)
        patched_specs.append(spec)

    setattr(model, "_vit_attention_patch_config", patch_config)
    setattr(model, "_vit_attention_patched_indices", [spec.block_index for spec in patched_specs])
    return patched_specs


def create_vit_attention_compression_model(
    base_model: nn.Module,
    *,
    config: ViTAttentionPatchConfig | None = None,
    attention_adapter: AttentionAdapter | None = None,
) -> nn.Module:
    """Clone a ViT model and patch selected encoder attention blocks."""

    patched_model = copy.deepcopy(base_model).eval()
    patched_specs = patch_vit_attention_modules(
        patched_model,
        config=config,
        attention_adapter=attention_adapter,
    )
    setattr(patched_model, "_vit_attention_patch_specs", patched_specs)
    return patched_model


def selected_vit_attention_specs(
    model: nn.Module,
    *,
    config: ViTAttentionPatchConfig | None = None,
) -> list[ViTAttentionModuleSpec]:
    """Return the metadata for the layers that would be patched under a config."""

    patch_config = config or ViTAttentionPatchConfig()
    specs = iter_vit_attention_modules(model)
    selected_indices = set(_resolve_layer_indices(len(specs), patch_config))
    return [spec for spec in specs if spec.block_index in selected_indices]
