from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scann.experiments import legacy_runner


class TinyLegacyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(256, 512)
        self.head = nn.Linear(512, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.head(torch.relu(x))


def test_export_legacy_compressed_checkpoint_variants_writes_smaller_files(monkeypatch, tmp_path):
    torch.manual_seed(0)
    model = TinyLegacyModel().eval()
    checkpoint_path = tmp_path / "tiny_dense.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": "tiny_legacy",
            "threshold": 0.5,
            "config": {
                "experiment_name": "tiny_dense",
                "model_name": "tiny_legacy",
                "image_size": 224,
            },
        },
        checkpoint_path,
    )

    monkeypatch.setattr(
        legacy_runner,
        "create_experiment_model",
        lambda model_name, pretrained, image_size: TinyLegacyModel(),
    )

    summaries = legacy_runner.export_legacy_compressed_checkpoint_variants(
        checkpoint_path,
        output_dir=tmp_path,
    )

    assert len(summaries) == 2
    summary_by_mode = {summary["checkpoint_compression_mode"]: summary for summary in summaries}
    assert summary_by_mode["custom_int8_weight_only"]["compressed_checkpoint_file_size_mb"] > 0.0
    assert summary_by_mode["packed_int4_weight_only"]["compressed_model_state_size_mb"] < summary_by_mode[
        "packed_int4_weight_only"
    ]["source_model_state_size_mb"]

    exported = torch.load(tmp_path / "tiny_dense_packed_int4_weight_only.pt", map_location="cpu")
    assert exported["checkpoint_compression_mode"] == "packed_int4_weight_only"
    assert "proj.packed_codes" in exported["state_dict"]
    assert "head.scales" in exported["state_dict"]


def test_benchmark_legacy_checkpoint_reports_compressed_checkpoint_mode(monkeypatch, tmp_path):
    torch.manual_seed(1)
    dense_model = TinyLegacyModel().eval()
    checkpoint_path = tmp_path / "tiny_dense.pt"
    torch.save(
        {
            "state_dict": dense_model.state_dict(),
            "model_name": "tiny_legacy",
            "threshold": 0.5,
            "config": {
                "experiment_name": "tiny_dense",
                "model_name": "tiny_legacy",
                "image_size": 224,
            },
        },
        checkpoint_path,
    )

    monkeypatch.setattr(
        legacy_runner,
        "create_experiment_model",
        lambda model_name, pretrained, image_size: TinyLegacyModel(),
    )

    compressed_summary = legacy_runner.export_legacy_compressed_checkpoint(
        checkpoint_path,
        checkpoint_compression_mode="packed_int4_weight_only",
        output_path=tmp_path / "tiny_dense_packed_int4_weight_only.pt",
    )

    def _fake_benchmark(model_obj, config_obj, split="test", device="cpu", threshold=0.5, input_dtype=None, clone_model=True):
        del config_obj, split, device, threshold, input_dtype, clone_model
        assert isinstance(model_obj.proj, legacy_runner.PackedInt4WeightOnlyLinear)
        assert isinstance(model_obj.head, legacy_runner.PackedInt4WeightOnlyLinear)
        return {
            "accuracy": 0.9,
            "precision": 0.9,
            "recall": 0.9,
            "f1": 0.9,
            "roc_auc": 0.95,
            "ms_per_image": 1.0,
            "peak_cpu_memory_mb": 10.0,
            "peak_gpu_memory_mb": 0.0,
            "peak_gpu_memory_attention_only_mb": 0.0,
            "packed_kv_size_mb": 0.0,
            "token_count": 0,
            "num_patched_layers": 0,
        }

    monkeypatch.setattr(legacy_runner, "benchmark_legacy_model_inference", _fake_benchmark)

    metrics = legacy_runner.benchmark_legacy_checkpoint(
        compressed_summary["checkpoint_path"],
        device="cpu",
    )

    assert metrics["checkpoint_compression_mode"] == "packed_int4_weight_only"
    assert metrics["checkpoint_file_size_mb"] > 0.0
