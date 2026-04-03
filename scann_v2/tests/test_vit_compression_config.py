from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scann.experiments import legacy_runner


def test_load_experiment_config_supports_vit_compression_fields():
    config = legacy_runner.load_experiment_config(
        {
            "experiment_name": "vit_kv_cfg",
            "model_name": "vit_b_16",
            "image_size": 224,
            "attention_compression_mode": "vit_packed_kv",
            "attention_layer_selector": "first_n",
            "attention_layer_count": 3,
            "enabled_layer_indices": [1, "2"],
            "kv_bits": 4,
            "group_size": 16,
            "token_block_size": 32,
            "preserve_cls_token": True,
            "quantize_k": True,
            "quantize_v": False,
        }
    )

    assert config.attention_compression_mode == "vit_packed_kv"
    assert config.attention_layer_selector == "first_n"
    assert config.attention_layer_count == 3
    assert config.enabled_layer_indices == [1, 2]
    assert config.kv_bits == 4
    assert config.group_size == 16
    assert config.token_block_size == 32
    assert config.preserve_cls_token is True
    assert config.quantize_k is True
    assert config.quantize_v is False


def test_benchmark_legacy_checkpoint_passes_through_new_metrics(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "demo.pt"
    checkpoint_path.write_bytes(b"placeholder")
    config = legacy_runner.LegacyExperimentConfig(
        experiment_name="legacy_vit_b16_fullres_k4_stream",
        model_name="vit_b_16",
        image_size=224,
        attention_compression_mode="vit_packed_kv",
        attention_layer_selector="all",
        kv_bits=4,
        group_size=32,
        token_block_size=64,
        preserve_cls_token=True,
        quantize_k=True,
        quantize_v=True,
    )
    model = nn.Identity()

    def _fake_load_bundle(checkpoint_path, config_override=None):
        return ({"threshold": 0.6, "model_name": "vit_b_16"}, config, model)

    def _fake_benchmark(model_obj, config_obj, split="test", device="cpu", threshold=0.5, input_dtype=None, clone_model=True):
        assert model_obj is model
        assert config_obj is config
        return {
            "accuracy": 0.95,
            "precision": 0.95,
            "recall": 0.95,
            "f1": 0.95,
            "roc_auc": 0.99,
            "ms_per_image": 2.0,
            "peak_cpu_memory_mb": 100.0,
            "peak_gpu_memory_mb": 200.0,
            "peak_gpu_memory_attention_only_mb": 80.0,
            "packed_kv_size_mb": 12.5,
            "token_count": 197,
            "num_patched_layers": 12,
        }

    monkeypatch.setattr(legacy_runner, "_load_legacy_checkpoint_bundle", _fake_load_bundle)
    monkeypatch.setattr(legacy_runner, "benchmark_legacy_model_inference", _fake_benchmark)

    metrics = legacy_runner.benchmark_legacy_checkpoint(
        checkpoint_path,
        config_override={"attention_compression_mode": "vit_packed_kv"},
        split="test",
        device="cpu",
    )

    assert metrics["attention_compression_mode"] == "vit_packed_kv"
    assert metrics["packed_kv_size_mb"] == 12.5
    assert metrics["peak_gpu_memory_attention_only_mb"] == 80.0
    assert metrics["token_count"] == 197
    assert metrics["num_patched_layers"] == 12


def test_train_legacy_classifier_rejects_attention_compression_for_training(tmp_path, monkeypatch):
    config = {
        "experiment_name": "train_not_supported",
        "dataset_dir": str(tmp_path),
        "manifest_path": str(tmp_path / "manifest.json"),
        "output_root": str(tmp_path),
        "model_name": "vit_b_16",
        "attention_compression_mode": "vit_packed_kv",
    }

    monkeypatch.setattr(
        legacy_runner,
        "_ensure_manifest",
        lambda experiment_config: Path(experiment_config.manifest_path),
    )
    monkeypatch.setattr(
        legacy_runner,
        "create_experiment_model",
        lambda model_name, pretrained, image_size: nn.Identity(),
    )

    try:
        legacy_runner.train_legacy_classifier(config)
    except ValueError as exc:
        assert "inference/benchmark" in str(exc)
    else:
        raise AssertionError("Expected train_legacy_classifier to reject attention compression mode")
