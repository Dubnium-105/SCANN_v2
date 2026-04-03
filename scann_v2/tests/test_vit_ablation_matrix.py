from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scann.experiments import legacy_runner


def test_build_vit_attention_ablation_variants_includes_required_full_module_rows():
    config = legacy_runner.LegacyExperimentConfig(
        experiment_name="vit_ablation",
        model_name="vit_b_16",
        image_size=224,
        enabled_layer_indices=[1, 3, 8],
    )

    variants = legacy_runner._build_vit_attention_ablation_variants(config)
    variant_names = {variant["variant"] for variant in variants}

    assert "baseline_dense" in variant_names
    assert "all_layers_k_only_4bit" in variant_names
    assert "all_layers_kv_4bit" in variant_names
    assert "all_layers_kv_4bit_cls_preserved" in variant_names
    assert "all_layers_kv_4bit_qjl" in variant_names

    first_variant = next(variant for variant in variants if variant["variant"] == "first_25pct_kv_4bit")
    middle_variant = next(variant for variant in variants if variant["variant"] == "middle_50pct_kv_4bit")
    last_variant = next(variant for variant in variants if variant["variant"] == "last_25pct_kv_4bit")
    custom_variant = next(variant for variant in variants if variant["variant"] == "custom_indices_kv_4bit")

    assert first_variant["attention_layer_selector"] == "first_n"
    assert first_variant["attention_layer_count"] == 3
    assert middle_variant["attention_layer_selector"] == "middle"
    assert middle_variant["attention_layer_count"] == 6
    assert last_variant["attention_layer_selector"] == "last_n"
    assert last_variant["attention_layer_count"] == 3
    assert custom_variant["attention_layer_selector"] == "explicit_indices"
    assert custom_variant["enabled_layer_indices"] == [1, 3, 8]


def test_run_vit_attention_ablation_writes_csv_with_required_metadata(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "demo.pt"
    checkpoint_path.write_bytes(b"placeholder")
    csv_path = tmp_path / "vit_attention_ablation.csv"
    base_config = {
        "experiment_name": "legacy_vit_b16_fullres_ablation",
        "dataset_dir": str(tmp_path),
        "manifest_path": str(tmp_path / "manifest.json"),
        "output_root": str(tmp_path),
        "model_name": "vit_b_16",
        "image_size": 224,
        "enabled_layer_indices": [0, 5, 11],
        "group_size": 32,
        "token_block_size": 64,
    }

    def _fake_benchmark(checkpoint_file, *, config_override=None, split="test", device="cpu", threshold=None, input_dtype=None):
        del checkpoint_file, input_dtype
        assert split == "test"
        assert device == "cpu"
        assert threshold == 0.61
        override = legacy_runner.load_experiment_config(config_override or {})
        multiplier = 1.0 if override.attention_compression_mode == "none" else 0.9
        return {
            "checkpoint_path": str(checkpoint_path),
            "split": split,
            "threshold": threshold,
            "accuracy": 0.95 * multiplier,
            "precision": 0.94 * multiplier,
            "recall": 0.93 * multiplier,
            "f1": 0.92 * multiplier,
            "roc_auc": 0.99 * multiplier,
            "ms_per_image": 2.0,
            "peak_cpu_memory_mb": 100.0,
            "peak_gpu_memory_mb": 200.0 if override.attention_compression_mode == "none" else 150.0,
            "peak_gpu_memory_attention_only_mb": 0.0 if override.attention_compression_mode == "none" else 70.0,
            "packed_kv_size_mb": 0.0 if override.attention_compression_mode == "none" else 10.0,
            "token_count": 197,
            "num_patched_layers": 0 if override.attention_compression_mode == "none" else 12,
            "experiment_name": override.experiment_name,
            "model_name": override.model_name,
            "attention_compression_mode": override.attention_compression_mode,
            "streaming_enabled": override.streaming_enabled,
            "materialize_attention_matrix": override.materialize_attention_matrix,
        }

    monkeypatch.setattr(legacy_runner, "benchmark_legacy_checkpoint", _fake_benchmark)

    summary = legacy_runner.run_vit_attention_ablation(
        checkpoint_path,
        base_config=base_config,
        comparison_csv_path=csv_path,
        split="test",
        device="cpu",
        threshold=0.61,
    )

    assert summary["comparison_csv_path"] == str(csv_path)
    assert csv_path.is_file()

    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8", newline="")))
    assert rows
    row_by_variant = {row["variant"]: row for row in rows}

    assert row_by_variant["baseline_dense"]["streaming_enabled"] == "False"
    assert row_by_variant["baseline_dense"]["materialize_attention_matrix"] == "True"
    assert row_by_variant["all_layers_k_only_4bit"]["layer_scope"] == "all_layers"
    assert row_by_variant["all_layers_kv_4bit"]["kv_target"] == "K/V both"
    assert row_by_variant["all_layers_kv_4bit_cls_preserved"]["cls_policy"] == "cls_preserved"
    assert row_by_variant["all_layers_kv_4bit_qjl"]["residual_mode"] == "qjl_sign_norm"
    assert row_by_variant["all_layers_kv_4bit_qjl"]["qjl_dim"] == "8"
    assert row_by_variant["custom_indices_kv_4bit"]["enabled_layer_indices"] == "[0, 5, 11]"
    assert row_by_variant["middle_50pct_kv_4bit"]["attention_layer_selector"] == "middle"
    assert row_by_variant["last_25pct_kv_4bit"]["attention_layer_count"] == "3"


def test_apply_attention_compression_rejects_non_streaming_packed_kv():
    config = legacy_runner.LegacyExperimentConfig(
        experiment_name="vit_bad_streaming",
        model_name="vit_b_16",
        image_size=224,
        attention_compression_mode="vit_packed_kv",
        streaming_enabled=False,
        materialize_attention_matrix=False,
    )

    try:
        legacy_runner._apply_attention_compression_from_config(object(), config, for_training=False)
    except ValueError as exc:
        assert "streaming_enabled=True" in str(exc)
    else:
        raise AssertionError("Expected packed-KV compression to reject streaming_enabled=False")
