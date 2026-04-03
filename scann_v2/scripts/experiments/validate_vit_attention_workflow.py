"""Lightweight regression validation for the full-resolution ViT packed-KV workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scann.experiments import legacy_runner


def run_vit_attention_regression(*, image_size: int = 224, device: str = "cpu") -> dict[str, Any]:
    runtime_device = torch.device(device)
    model_name = "vit_b_16"
    summary_required = {
        "peak_gpu_memory_attention_only_mb",
        "packed_kv_size_mb",
        "token_count",
        "num_patched_layers",
        "residual_mode",
        "qjl_dim",
        "streaming_enabled",
        "materialize_attention_matrix",
    }
    ablation_required = {
        "layer_scope",
        "kv_target",
        "cls_policy",
        "streaming_enabled",
        "materialize_attention_matrix",
        "residual_mode",
        "qjl_dim",
    }

    torch.manual_seed(0)
    baseline_model = legacy_runner.create_experiment_model(
        model_name,
        pretrained=False,
        image_size=image_size,
    ).to(runtime_device).eval()
    packed_model = legacy_runner.create_vit_packed_kv_attention_model(
        baseline_model,
        layer_selector="all",
        group_size=8,
        block_size=8,
        quantize_k=True,
        quantize_v=True,
        preserve_cls_token=True,
        residual_mode="qjl_sign_norm",
        qjl_dim=8,
    ).to(runtime_device).eval()

    x = torch.randn(1, 3, image_size, image_size, device=runtime_device)
    with torch.no_grad():
        baseline_output = baseline_model(x)
        packed_output = packed_model(x)

    ablation_config = legacy_runner.LegacyExperimentConfig(
        experiment_name="vit_attention_regression",
        model_name=model_name,
        image_size=image_size,
        enabled_layer_indices=[0, 5, 11],
    )
    ablation_variants = legacy_runner._build_vit_attention_ablation_variants(ablation_config)
    variant_names = {variant["variant"] for variant in ablation_variants}
    required_variants = {
        "baseline_dense",
        "all_layers_k_only_4bit",
        "all_layers_kv_4bit",
        "all_layers_kv_4bit_cls_preserved",
        "all_layers_kv_4bit_qjl",
    }

    return {
        "model_name": model_name,
        "device": runtime_device.type,
        "baseline_forward_ok": bool(torch.isfinite(baseline_output).all().item()),
        "full_module_forward_ok": bool(torch.isfinite(packed_output).all().item()),
        "baseline_output_shape": list(baseline_output.shape),
        "full_module_output_shape": list(packed_output.shape),
        "summary_schema_complete": summary_required.issubset(set(legacy_runner.SUMMARY_COLUMNS)),
        "ablation_schema_complete": ablation_required.issubset(set(legacy_runner.VIT_ATTENTION_ABLATION_COLUMNS)),
        "all_layers_variants_present": required_variants.issubset(variant_names),
        "patched_layers": list(getattr(packed_model, "_vit_attention_patched_indices", [])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run lightweight regression validation for ViT packed-KV workflow")
    parser.add_argument("--image-size", type=int, default=224, help="Square smoke input size")
    parser.add_argument("--device", default="cpu", help="Smoke device, for example cpu or cuda")
    args = parser.parse_args()

    summary = run_vit_attention_regression(image_size=args.image_size, device=args.device)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(
        bool(summary[key])
        for key in (
            "baseline_forward_ok",
            "full_module_forward_ok",
            "summary_schema_complete",
            "ablation_schema_complete",
            "all_layers_variants_present",
        )
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
