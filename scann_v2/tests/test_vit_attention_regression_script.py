from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_regression_script_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "experiments"
        / "validate_vit_attention_workflow.py"
    )
    spec = importlib.util.spec_from_file_location("validate_vit_attention_workflow", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load regression script module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_vit_attention_regression_returns_success_flags():
    module = _load_regression_script_module()

    summary = module.run_vit_attention_regression(image_size=224, device="cpu")

    assert summary["baseline_forward_ok"] is True
    assert summary["full_module_forward_ok"] is True
    assert summary["summary_schema_complete"] is True
    assert summary["ablation_schema_complete"] is True
    assert summary["all_layers_variants_present"] is True
    assert summary["baseline_output_shape"] == [1, 2]
    assert summary["full_module_output_shape"] == [1, 2]
    assert summary["patched_layers"] == list(range(12))
