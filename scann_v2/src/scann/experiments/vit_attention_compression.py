"""ViT attention patch scaffold for compression experiments."""

from __future__ import annotations

import copy
import math
import types
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .packed_kv import PackedKV4Bit, pack_tensor_4bit, unpack_kv_block, unpack_tensor_4bit_block


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


@dataclass(frozen=True)
class PackedKVAttentionConfig:
    """Runtime config for packed K/V streaming attention inside ViT blocks."""

    group_size: int = 32
    block_size: int = 64
    quantize_k: bool = True
    quantize_v: bool = True
    preserve_cls_token: bool = False
    compute_dtype: torch.dtype = torch.float32


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


def decode_kv_block(
    packed_kv: PackedKV4Bit,
    *,
    start: int,
    end: int,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode one K/V token block from packed storage."""

    return unpack_kv_block(packed_kv, start=start, end=end, dtype=dtype)


def online_softmax_update(
    *,
    logits_block: torch.Tensor,
    value_block: torch.Tensor,
    running_max: torch.Tensor,
    running_denom: torch.Tensor,
    running_weighted_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update streaming softmax state with one attention block."""

    block_max = logits_block.amax(dim=-1, keepdim=True)
    next_max = torch.maximum(running_max, block_max)

    prev_scale = torch.where(
        running_denom > 0,
        torch.exp(running_max - next_max),
        torch.zeros_like(running_denom),
    )
    block_scale = torch.exp(logits_block - next_max)

    next_denom = (running_denom * prev_scale) + block_scale.sum(dim=-1, keepdim=True)
    next_weighted_values = (running_weighted_values * prev_scale) + torch.matmul(block_scale, value_block)
    return next_max, next_denom, next_weighted_values


def streaming_packed_attention(
    query: torch.Tensor,
    packed_kv: PackedKV4Bit,
    *,
    block_size: int = 64,
    scale: float | None = None,
    compute_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Compute attention by blockwise K/V decode without materializing dense K/V."""

    if query.ndim != 4:
        raise ValueError(f"Expected query shaped [B, H, T, D], got {tuple(query.shape)}")

    return _streaming_attention_from_sources(
        query,
        packed_key=packed_kv.packed_k,
        packed_value=packed_kv.packed_v,
        block_size=block_size,
        scale=scale,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
    )


def _streaming_attention_update_from_sources(
    query: torch.Tensor,
    *,
    dense_key: torch.Tensor | None = None,
    dense_value: torch.Tensor | None = None,
    packed_key: object | None = None,
    packed_value: object | None = None,
    block_size: int = 64,
    scale: float | None = None,
    compute_dtype: torch.dtype = torch.float32,
    running_max: torch.Tensor | None = None,
    running_denom: torch.Tensor | None = None,
    running_weighted_values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if query.ndim != 4:
        raise ValueError(f"Expected query shaped [B, H, T, D], got {tuple(query.shape)}")
    if dense_key is None and packed_key is None:
        raise ValueError("One of dense_key or packed_key must be provided")
    if dense_value is None and packed_value is None:
        raise ValueError("One of dense_value or packed_value must be provided")

    resolved_block_size = max(1, int(block_size))
    query_compute = query.to(dtype=compute_dtype)
    head_dim = int(query.shape[-1])
    scale_value = float(scale) if scale is not None else float(head_dim) ** -0.5

    if packed_key is not None:
        total_tokens = int(packed_key.original_shape[packed_key.token_axis])
    else:
        total_tokens = int(dense_key.shape[-2])

    if running_max is None or running_denom is None or running_weighted_values is None:
        running_max = torch.full(
            (*query_compute.shape[:-1], 1),
            fill_value=-torch.inf,
            dtype=compute_dtype,
            device=query_compute.device,
        )
        running_denom = torch.zeros_like(running_max)
        running_weighted_values = torch.zeros_like(query_compute)

    for start in range(0, total_tokens, resolved_block_size):
        end = min(total_tokens, start + resolved_block_size)
        if packed_key is not None:
            key_block = unpack_tensor_4bit_block(
                packed_key,
                start=start,
                end=end,
                axis=packed_key.token_axis,
                dtype=compute_dtype,
            )
        else:
            key_block = dense_key[..., start:end, :].to(dtype=compute_dtype)

        if packed_value is not None:
            value_block = unpack_tensor_4bit_block(
                packed_value,
                start=start,
                end=end,
                axis=packed_value.token_axis,
                dtype=compute_dtype,
            )
        else:
            value_block = dense_value[..., start:end, :].to(dtype=compute_dtype)

        logits_block = torch.matmul(query_compute, key_block.transpose(-2, -1)) * scale_value
        running_max, running_denom, running_weighted_values = online_softmax_update(
            logits_block=logits_block,
            value_block=value_block,
            running_max=running_max,
            running_denom=running_denom,
            running_weighted_values=running_weighted_values,
        )

    return running_max, running_denom, running_weighted_values


def _streaming_attention_from_sources(
    query: torch.Tensor,
    *,
    dense_key: torch.Tensor | None = None,
    dense_value: torch.Tensor | None = None,
    packed_key: object | None = None,
    packed_value: object | None = None,
    block_size: int = 64,
    scale: float | None = None,
    compute_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    resolved_output_dtype = output_dtype or query.dtype
    _, running_denom, running_weighted_values = _streaming_attention_update_from_sources(
        query,
        dense_key=dense_key,
        dense_value=dense_value,
        packed_key=packed_key,
        packed_value=packed_value,
        block_size=block_size,
        scale=scale,
        compute_dtype=compute_dtype,
    )
    output = running_weighted_values / running_denom.clamp_min(1e-12)
    return output.to(dtype=resolved_output_dtype)


def _project_self_attention_qkv(
    normalized_input: torch.Tensor,
    attention_module: nn.MultiheadAttention,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not bool(attention_module.batch_first):
        raise RuntimeError("Packed KV ViT attention adapter currently expects batch_first=True")
    if attention_module.bias_k is not None or attention_module.bias_v is not None:
        raise RuntimeError("Packed KV ViT attention adapter does not support bias_k/bias_v")
    if bool(attention_module.add_zero_attn):
        raise RuntimeError("Packed KV ViT attention adapter does not support add_zero_attn")
    if attention_module.kdim not in (None, attention_module.embed_dim):
        raise RuntimeError("Packed KV ViT attention adapter expects self-attention with kdim==embed_dim")
    if attention_module.vdim not in (None, attention_module.embed_dim):
        raise RuntimeError("Packed KV ViT attention adapter expects self-attention with vdim==embed_dim")
    if attention_module.training and float(attention_module.dropout) > 0.0:
        raise RuntimeError(
            "Packed KV streaming adapter currently supports eval mode or zero attention dropout only"
        )

    embed_dim = int(attention_module.embed_dim)
    if attention_module.in_proj_weight is not None:
        q_weight, k_weight, v_weight = attention_module.in_proj_weight.chunk(3, dim=0)
        if attention_module.in_proj_bias is not None:
            q_bias, k_bias, v_bias = attention_module.in_proj_bias.chunk(3, dim=0)
        else:
            q_bias = k_bias = v_bias = None
    else:
        q_weight = attention_module.q_proj_weight
        k_weight = attention_module.k_proj_weight
        v_weight = attention_module.v_proj_weight
        q_bias = k_bias = v_bias = None

    q = F.linear(normalized_input, q_weight, q_bias)
    k = F.linear(normalized_input, k_weight, k_bias)
    v = F.linear(normalized_input, v_weight, v_bias)

    batch_size, token_count, _ = q.shape
    head_dim = embed_dim // int(attention_module.num_heads)
    reshape_dims = (batch_size, token_count, int(attention_module.num_heads), head_dim)
    q = q.reshape(reshape_dims).permute(0, 2, 1, 3).contiguous()
    k = k.reshape(reshape_dims).permute(0, 2, 1, 3).contiguous()
    v = v.reshape(reshape_dims).permute(0, 2, 1, 3).contiguous()
    return q, k, v


def _merge_attention_heads(attended_values: torch.Tensor) -> torch.Tensor:
    batch_size, num_heads, token_count, head_dim = attended_values.shape
    return attended_values.permute(0, 2, 1, 3).reshape(batch_size, token_count, num_heads * head_dim)


def build_packed_kv_attention_adapter(
    config: PackedKVAttentionConfig | None = None,
) -> AttentionAdapter:
    runtime_config = config or PackedKVAttentionConfig()

    def _adapter(
        normalized_input: torch.Tensor,
        attention_module: nn.Module,
        module_spec: ViTAttentionModuleSpec,
    ) -> torch.Tensor:
        if not isinstance(attention_module, nn.MultiheadAttention):
            raise TypeError("Packed KV attention adapter expects nn.MultiheadAttention modules")

        q, k, v = _project_self_attention_qkv(normalized_input, attention_module)
        setattr(attention_module, "_vit_last_module_spec", module_spec)
        setattr(attention_module, "_vit_last_token_count", int(q.shape[-2]))
        setattr(attention_module, "_vit_last_quantize_k", bool(runtime_config.quantize_k))
        setattr(attention_module, "_vit_last_quantize_v", bool(runtime_config.quantize_v))
        setattr(attention_module, "_vit_last_preserve_cls_token", bool(runtime_config.preserve_cls_token))
        cls_k = cls_v = None
        patch_k = k
        patch_v = v
        if runtime_config.preserve_cls_token and int(k.shape[-2]) > 0:
            cls_k = k[..., :1, :]
            cls_v = v[..., :1, :]
            patch_k = k[..., 1:, :]
            patch_v = v[..., 1:, :]

        packed_key = (
            pack_tensor_4bit(patch_k, group_size=runtime_config.group_size, token_axis=-2)
            if runtime_config.quantize_k and int(patch_k.shape[-2]) > 0
            else None
        )
        packed_value = (
            pack_tensor_4bit(patch_v, group_size=runtime_config.group_size, token_axis=-2)
            if runtime_config.quantize_v and int(patch_v.shape[-2]) > 0
            else None
        )
        dense_key = None if packed_key is not None else patch_k
        dense_value = None if packed_value is not None else patch_v
        packed_storage_bytes = 0
        if packed_key is not None:
            packed_storage_bytes += int(packed_key.storage_size_bytes())
        if packed_value is not None:
            packed_storage_bytes += int(packed_value.storage_size_bytes())
        setattr(attention_module, "_vit_last_packed_kv_size_bytes", packed_storage_bytes)

        scale = float(q.shape[-1]) ** -0.5
        running_max = running_denom = running_weighted_values = None
        if cls_k is not None and cls_v is not None:
            running_max, running_denom, running_weighted_values = _streaming_attention_update_from_sources(
                q,
                dense_key=cls_k,
                dense_value=cls_v,
                block_size=1,
                scale=scale,
                compute_dtype=runtime_config.compute_dtype,
            )
        if cls_k is None or int(patch_k.shape[-2]) > 0:
            running_max, running_denom, running_weighted_values = _streaming_attention_update_from_sources(
                q,
                dense_key=dense_key,
                dense_value=dense_value,
                packed_key=packed_key,
                packed_value=packed_value,
                block_size=runtime_config.block_size,
                scale=scale,
                compute_dtype=runtime_config.compute_dtype,
                running_max=running_max,
                running_denom=running_denom,
                running_weighted_values=running_weighted_values,
            )
        attended = running_weighted_values / running_denom.clamp_min(1e-12)
        if normalized_input.device.type == "cuda":
            setattr(
                attention_module,
                "_vit_last_attention_memory_bytes",
                int(torch.cuda.memory_allocated(normalized_input.device)),
            )
        else:
            setattr(attention_module, "_vit_last_attention_memory_bytes", 0)

        merged = _merge_attention_heads(attended).to(dtype=normalized_input.dtype)
        return attention_module.out_proj(merged)

    return _adapter


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


def create_vit_packed_kv_compression_model(
    base_model: nn.Module,
    *,
    patch_config: ViTAttentionPatchConfig | None = None,
    packed_kv_config: PackedKVAttentionConfig | None = None,
) -> nn.Module:
    """Clone a ViT model and patch selected encoder blocks with packed KV attention."""

    adapter = build_packed_kv_attention_adapter(config=packed_kv_config)
    patched_model = create_vit_attention_compression_model(
        base_model,
        config=patch_config,
        attention_adapter=adapter,
    )
    setattr(patched_model, "_vit_packed_kv_attention_config", packed_kv_config or PackedKVAttentionConfig())
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
