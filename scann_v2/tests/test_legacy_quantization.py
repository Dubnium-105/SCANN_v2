from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scann.experiments import legacy_runner


def test_create_int4_weight_only_model_replaces_linear_modules():
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Sequential(nn.Linear(8, 2)),
    ).eval()
    x = torch.randn(3, 4)

    quantized = legacy_runner.create_int4_weight_only_model(model)
    output = quantized(x)

    assert output.shape == (3, 2)
    assert not any(isinstance(module, nn.Linear) for module in quantized.modules())
    assert any(isinstance(module, legacy_runner.Int4WeightOnlyLinear) for module in quantized.modules())


def test_run_legacy_quantization_smoke_writes_csv(tmp_path, monkeypatch):
    output_root = tmp_path / "experiments"
    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / "legacy_swin_t_pretrained_gpu_best.pt"
    checkpoint_path.write_bytes(b"placeholder")

    config = legacy_runner.LegacyExperimentConfig(
        experiment_name="legacy_swin_t_pretrained_gpu",
        output_root=str(output_root),
        model_name="swin_t",
        batch_size=2,
        image_size=224,
        resize_mode="resize",
        normalize=True,
        input_mode="new_old_diff",
    )
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).eval()

    def _fake_load_bundle(checkpoint_path, config_override=None):
        return (
            {"threshold": 0.6, "model_name": "swin_t"},
            config,
            model,
        )

    def _fake_quantize_dynamic(model_obj, modules, dtype):
        tagged = model_obj
        tagged.quant_variant = "dynamic_int8"
        return tagged

    def _fake_create_int4(model_obj):
        tagged = model_obj
        tagged.quant_variant = "custom_weight_only_int4"
        return tagged

    def _fake_benchmark(
        model_obj,
        config_obj,
        split="test",
        device="cpu",
        threshold=0.5,
        input_dtype=None,
        clone_model=True,
    ):
        variant = getattr(model_obj, "quant_variant", "fp32_baseline")
        metric_map = {
            "fp32_baseline": (0.96, 0.95, 12.0, 512.0),
            "dynamic_int8": (0.95, 0.94, 8.0, 420.0),
            "custom_weight_only_int4": (0.94, 0.93, 10.0, 430.0),
        }
        accuracy, f1, ms_per_image, peak_cpu_memory_mb = metric_map[variant]
        return {
            "accuracy": accuracy,
            "precision": accuracy,
            "recall": accuracy,
            "f1": f1,
            "roc_auc": 0.99,
            "ms_per_image": ms_per_image,
            "peak_cpu_memory_mb": peak_cpu_memory_mb,
            "peak_gpu_memory_mb": 0.0,
        }

    monkeypatch.setattr(legacy_runner, "_load_legacy_checkpoint_bundle", _fake_load_bundle)
    monkeypatch.setattr(legacy_runner, "benchmark_legacy_model_inference", _fake_benchmark)
    monkeypatch.setattr(legacy_runner, "create_int4_weight_only_model", _fake_create_int4)
    monkeypatch.setattr(legacy_runner.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(legacy_runner.torch.ao.quantization, "quantize_dynamic", _fake_quantize_dynamic)

    summary = legacy_runner.run_legacy_quantization_smoke(
        {
            "experiment_name": "legacy_swin_t_pretrained_gpu",
            "output_root": str(output_root),
            "model_name": "swin_t",
        }
    )

    assert summary["best_quantized_variant"] == "dynamic_int8"
    csv_path = Path(summary["comparison_csv_path"])
    assert csv_path.exists()

    with csv_path.open("r", encoding="utf-8", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))

    assert {row["variant"] for row in rows} == {
        "fp32_baseline",
        "dynamic_int8",
        "standard_int4_g32",
        "standard_int4_g64",
        "standard_int4_g128",
        "int8_weight_only",
        "int8_dynact_int8_weight",
        "turboquant_mse_b3",
        "turboquant_mse_b4",
        "turboquant_prod_qjl_b4_m16",
        "turboquant_prod_qjl_b4_m32",
        "custom_weight_only_int4",
    }
    by_variant = {row["variant"]: row for row in rows}
    assert by_variant["dynamic_int8"]["status"] == "ok"
    assert by_variant["standard_int4_g32"]["status"] == "unsupported_in_environment"
    assert by_variant["int8_weight_only"]["status"] == "unsupported_in_environment"
    assert by_variant["turboquant_mse_b3"]["status"] == "unsupported_in_environment"
    assert by_variant["turboquant_prod_qjl_b4_m16"]["status"] == "unsupported_in_environment"

    summary_path = output_root / "results" / "legacy_swin_t_pretrained_gpu_quantization_smoke_summary.json"
    assert summary_path.exists()
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved["best_quantized_variant"] == "dynamic_int8"
