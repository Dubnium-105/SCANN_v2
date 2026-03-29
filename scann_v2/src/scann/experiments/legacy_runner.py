"""Training and evaluation entry points for legacy v1 triplet experiments."""

from __future__ import annotations

import csv
import json
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models

from scann.ai.device_utils import resolve_device
from scann.ai.trainer import FocalLoss

from .legacy_dataset import LegacyTripletExperimentDataset
from .legacy_manifest import build_legacy_triplet_manifest, load_legacy_manifest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_DIR = PROJECT_ROOT.parent / "dataset"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments"
DEFAULT_MANIFEST_PATH = DEFAULT_OUTPUT_ROOT / "manifests" / "legacy_v1_manifest.json"
SUMMARY_COLUMNS = [
    "experiment_name",
    "model_name",
    "input_mode",
    "image_size",
    "resize_mode",
    "pretrained",
    "seed",
    "batch_size",
    "lr",
    "epochs_requested",
    "epochs_ran",
    "best_epoch",
    "best_threshold",
    "val_accuracy",
    "val_precision",
    "val_recall",
    "val_f1",
    "val_f2",
    "val_roc_auc",
    "val_pr_auc",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_f2",
    "test_roc_auc",
    "test_pr_auc",
    "test_tn",
    "test_fp",
    "test_fn",
    "test_tp",
    "manifest_path",
    "checkpoint_path",
    "history_path",
    "plots_dir",
    "split_distribution_plot",
    "learning_curve_plot",
    "val_analysis_plot",
    "test_analysis_plot",
    "val_predictions_path",
    "test_predictions_path",
]


@dataclass
class LegacyExperimentConfig:
    """Serializable config for the legacy experiment pipeline."""

    experiment_name: str = "legacy_v1_experiment"
    dataset_dir: str = str(DEFAULT_DATASET_DIR)
    manifest_path: str = str(DEFAULT_MANIFEST_PATH)
    output_root: str = str(DEFAULT_OUTPUT_ROOT)
    model_name: str = "resnet18"
    pretrained: bool = False
    input_mode: str = "new_old_diff"
    image_size: int = 224
    resize_mode: str = "resize"
    normalize: bool = True
    batch_size: int = 32
    epochs: int = 30
    lr: float = 2e-4
    weight_decay: float = 1e-3
    optimizer: str = "AdamW"
    scheduler: str = "cosine"
    step_size: int = 10
    step_gamma: float = 0.5
    early_stopping_patience: int = 10
    selection_metric: str = "f1"
    threshold_metric: str = "f1"
    loss_name: str = "cross_entropy"
    focal_gamma: float = 2.0
    focal_alpha: list[float] = field(default_factory=lambda: [1.0, 1.5])
    use_weighted_sampler: bool = True
    augment: bool = True
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.5
    enable_rotate_90: bool = True
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    summary_csv_name: str = "experiment_results.csv"
    log_every_n_steps: int = 25


def _resolve_relative_path(value: str | None, base_dir: Path) -> str:
    if not value:
        return ""
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def load_experiment_config(config: str | Path | dict[str, Any]) -> LegacyExperimentConfig:
    """Load a config from json/yaml or build one from a mapping."""

    base_dir = Path.cwd()
    raw: dict[str, Any]
    if isinstance(config, dict):
        raw = dict(config)
    else:
        config_path = Path(config).resolve()
        base_dir = config_path.parent
        suffix = config_path.suffix.lower()
        if suffix == ".json":
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("YAML config requires PyYAML to be installed") from exc
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        else:
            raise ValueError(f"Unsupported config format: {config_path}")

    for key in ("dataset_dir", "manifest_path", "output_root"):
        if key in raw:
            raw[key] = _resolve_relative_path(raw.get(key), base_dir)

    return LegacyExperimentConfig(**raw)


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_model_name(model_name: str) -> str:
    normalized = str(model_name).strip().lower()
    aliases = {
        "cnn_baseline": "resnet18",
        "legacy_cnn": "resnet18",
        "legacy_resnet18": "resnet18",
        "vit_baseline": "vit_b_16",
        "vit_b16": "vit_b_16",
    }
    return aliases.get(normalized, normalized)


def create_experiment_model(model_name: str, *, pretrained: bool) -> nn.Module:
    """Create a torchvision classifier for the experiment."""

    normalized = _normalize_model_name(model_name)
    if normalized == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model
    if normalized == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model
    if normalized == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model
    if normalized == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, 2)
        return model
    raise ValueError(f"Unsupported model name: {model_name}")


def _criterion_from_config(config: LegacyExperimentConfig) -> nn.Module:
    loss_name = str(config.loss_name).strip().lower()
    if loss_name == "focal":
        return FocalLoss(gamma=float(config.focal_gamma), alpha=list(config.focal_alpha))
    return nn.CrossEntropyLoss()


def _optimizer_from_config(model: nn.Module, config: LegacyExperimentConfig) -> optim.Optimizer:
    optimizer_name = str(config.optimizer).strip().lower()
    if optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)
    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    return optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)


def _scheduler_from_config(
    optimizer: optim.Optimizer,
    config: LegacyExperimentConfig,
) -> optim.lr_scheduler._LRScheduler | None:
    scheduler_name = str(config.scheduler).strip().lower()
    if scheduler_name == "none":
        return None
    if scheduler_name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, int(config.step_size)),
            gamma=float(config.step_gamma),
        )
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(config.epochs)))


def _compute_binary_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int32)
    probs = np.asarray(probs, dtype=np.float64)
    preds = (probs >= threshold).astype(np.int32)

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in cm.ravel()]
    eps = 1e-12

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2.0 * precision * recall) / (precision + recall + eps)
    f2 = (5.0 * precision * recall) / ((4.0 * precision) + recall + eps)

    if len(np.unique(labels)) > 1:
        roc_auc = float(roc_auc_score(labels, probs))
        pr_auc = float(average_precision_score(labels, probs))
    else:
        roc_auc = 0.0
        pr_auc = 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f2": float(f2),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "threshold": float(threshold),
    }


def _choose_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
    *,
    metric_name: str,
) -> tuple[float, dict[str, Any]]:
    labels = np.asarray(labels, dtype=np.int32)
    probs = np.asarray(probs, dtype=np.float64)

    if probs.size == 0:
        raise ValueError("Cannot choose a threshold from empty predictions")

    thresholds = np.unique(np.concatenate([probs, np.array([0.5], dtype=np.float64)]))
    if thresholds.size > 257:
        quantiles = np.linspace(0.0, 1.0, 257)
        thresholds = np.unique(np.quantile(thresholds, quantiles))

    metric_key = str(metric_name).strip().lower()
    best_threshold = 0.5
    best_metrics = _compute_binary_metrics(labels, probs, threshold=best_threshold)
    best_score = float(best_metrics.get(metric_key, best_metrics["f1"]))

    for threshold in thresholds:
        metrics = _compute_binary_metrics(labels, probs, threshold=float(threshold))
        score = float(metrics.get(metric_key, metrics["f1"]))
        if score > best_score + 1e-9:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_score = score
        elif abs(score - best_score) <= 1e-9 and metrics["recall"] > best_metrics["recall"] + 1e-9:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_score = score

    return best_threshold, best_metrics


def _build_datasets(config: LegacyExperimentConfig) -> dict[str, LegacyTripletExperimentDataset]:
    return {
        "train": LegacyTripletExperimentDataset(
            config.manifest_path,
            split="train",
            dataset_root=config.dataset_dir,
            input_mode=config.input_mode,
            image_size=config.image_size,
            resize_mode=config.resize_mode,
            normalize=config.normalize,
            augment=config.augment,
            horizontal_flip_prob=config.horizontal_flip_prob,
            vertical_flip_prob=config.vertical_flip_prob,
            enable_rotate_90=config.enable_rotate_90,
        ),
        "val": LegacyTripletExperimentDataset(
            config.manifest_path,
            split="val",
            dataset_root=config.dataset_dir,
            input_mode=config.input_mode,
            image_size=config.image_size,
            resize_mode=config.resize_mode,
            normalize=config.normalize,
            augment=False,
        ),
        "test": LegacyTripletExperimentDataset(
            config.manifest_path,
            split="test",
            dataset_root=config.dataset_dir,
            input_mode=config.input_mode,
            image_size=config.image_size,
            resize_mode=config.resize_mode,
            normalize=config.normalize,
            augment=False,
        ),
    }


def _build_loaders(
    datasets: dict[str, LegacyTripletExperimentDataset],
    config: LegacyExperimentConfig,
) -> dict[str, DataLoader]:
    train_labels = datasets["train"].labels()
    sampler = None
    shuffle = False
    if config.use_weighted_sampler:
        counts = {
            0: max(train_labels.count(0), 1),
            1: max(train_labels.count(1), 1),
        }
        sample_weights = torch.tensor(
            [1.0 / counts[label] for label in train_labels],
            dtype=torch.double,
        )
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
    else:
        shuffle = True

    common_kwargs = {
        "batch_size": int(config.batch_size),
        "num_workers": int(config.num_workers),
        "pin_memory": torch.cuda.is_available(),
    }
    return {
        "train": DataLoader(datasets["train"], sampler=sampler, shuffle=shuffle, **common_kwargs),
        "val": DataLoader(datasets["val"], shuffle=False, **common_kwargs),
        "test": DataLoader(datasets["test"], shuffle=False, **common_kwargs),
    }


def _label_summary(labels: list[int]) -> str:
    negative = labels.count(0)
    positive = labels.count(1)
    total = len(labels)
    return f"total={total}, real={positive}, bogus={negative}"


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _torch_runtime_summary() -> str:
    cuda_build = getattr(torch.version, "cuda", None)
    return (
        f"python={sys.executable} | "
        f"torch={torch.__version__} | "
        f"torch_cuda_build={cuda_build or 'none'} | "
        f"cuda_available={torch.cuda.is_available()}"
    )


def _run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    *,
    epoch: int,
    total_epochs: int,
    log_every_n_steps: int,
) -> tuple[float, float]:
    model.train()
    loss_sum = 0.0
    seen = 0
    step_count = max(len(loader), 1)
    epoch_start = time.perf_counter()

    for step_index, (x, y) in enumerate(loader, start=1):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        loss_sum += float(loss.item()) * batch_size
        seen += batch_size
        if log_every_n_steps > 0 and (step_index == 1 or step_index % log_every_n_steps == 0 or step_index == step_count):
            running_loss = loss_sum / max(seen, 1)
            logger.info(
                "Epoch %02d/%02d | train step %03d/%03d | lr=%.3e | running_loss=%.5f",
                epoch,
                total_epochs,
                step_index,
                step_count,
                float(optimizer.param_groups[0]["lr"]),
                float(running_loss),
            )

    return loss_sum / max(seen, 1), time.perf_counter() - epoch_start


@torch.no_grad()
def _run_eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    loss_sum = 0.0
    seen = 0
    probs_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)
        probs = torch.softmax(logits, dim=1)[:, 1]

        batch_size = x.size(0)
        loss_sum += float(loss.item()) * batch_size
        seen += batch_size
        probs_list.append(probs.cpu().numpy())
        labels_list.append(y.cpu().numpy())

    if not probs_list:
        raise ValueError("Evaluation loader produced no batches")

    avg_loss = loss_sum / max(seen, 1)
    return avg_loss, np.concatenate(probs_list), np.concatenate(labels_list)


def _prepare_output_paths(config: LegacyExperimentConfig) -> dict[str, Path]:
    output_root = Path(config.output_root).resolve()
    manifest_dir = output_root / "manifests"
    results_dir = output_root / "results"
    checkpoint_dir = output_root / "checkpoints"
    plots_dir = output_root / "plots"
    prediction_dir = results_dir / "predictions"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    return {
        "output_root": output_root,
        "manifest_path": Path(config.manifest_path).resolve(),
        "checkpoint_path": checkpoint_dir / f"{config.experiment_name}_best.pt",
        "history_path": results_dir / f"{config.experiment_name}_history.csv",
        "summary_path": results_dir / f"{config.experiment_name}_summary.json",
        "config_snapshot_path": results_dir / f"{config.experiment_name}_config.json",
        "summary_csv_path": results_dir / config.summary_csv_name,
        "plots_dir": plots_dir,
        "split_distribution_plot": plots_dir / f"{config.experiment_name}_split_distribution.png",
        "learning_curve_plot": plots_dir / f"{config.experiment_name}_learning_curves.png",
        "val_analysis_plot": plots_dir / f"{config.experiment_name}_val_analysis.png",
        "test_analysis_plot": plots_dir / f"{config.experiment_name}_test_analysis.png",
        "prediction_dir": prediction_dir,
        "val_predictions_path": prediction_dir / f"{config.experiment_name}_val_predictions.csv",
        "test_predictions_path": prediction_dir / f"{config.experiment_name}_test_predictions.csv",
    }


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _append_summary_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=SUMMARY_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})


def _save_split_distribution_plot(manifest_path: Path, output_path: Path) -> None:
    manifest = load_legacy_manifest(manifest_path)
    split_counts = manifest.get("summary", {}).get("split_counts", {})
    split_names = ["train", "val", "test"]
    real_counts = [int(split_counts.get(split_name, {}).get("real", 0)) for split_name in split_names]
    bogus_counts = [int(split_counts.get(split_name, {}).get("bogus", 0)) for split_name in split_names]

    x = np.arange(len(split_names), dtype=np.float64)
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.bar(x - width / 2.0, real_counts, width=width, label="Real", color="#2b8cbe")
    ax.bar(x + width / 2.0, bogus_counts, width=width, label="Bogus", color="#de2d26")
    ax.set_xticks(x)
    ax.set_xticklabels([split_name.upper() for split_name in split_names])
    ax.set_ylabel("Sample Count")
    ax.set_title("Legacy V1 Split Distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _save_learning_curve_plot(history_rows: list[dict[str, Any]], output_path: Path) -> None:
    if not history_rows:
        return

    epochs = [int(row["epoch"]) for row in history_rows]
    train_loss = [float(row["train_loss"]) for row in history_rows]
    val_loss = [float(row["val_loss"]) for row in history_rows]
    val_f1 = [float(row["val_f1"]) for row in history_rows]
    val_f2 = [float(row["val_f2"]) for row in history_rows]
    val_roc_auc = [float(row["val_roc_auc"]) for row in history_rows]
    val_pr_auc = [float(row["val_pr_auc"]) for row in history_rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=160)

    axes[0].plot(epochs, train_loss, label="Train Loss", color="#2b8cbe", linewidth=2)
    axes[0].plot(epochs, val_loss, label="Val Loss", color="#de2d26", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Learning Curves")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    axes[1].plot(epochs, val_f1, label="Val F1", color="#31a354", linewidth=2)
    axes[1].plot(epochs, val_f2, label="Val F2", color="#756bb1", linewidth=2)
    axes[1].plot(epochs, val_roc_auc, label="Val ROC-AUC", color="#636363", linewidth=1.8)
    axes[1].plot(epochs, val_pr_auc, label="Val PR-AUC", color="#fd8d3c", linewidth=1.8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title("Validation Metrics")
    axes[1].grid(alpha=0.2)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _save_analysis_figure(
    labels: np.ndarray,
    probs: np.ndarray,
    *,
    threshold: float,
    split_name: str,
    output_path: Path,
) -> None:
    metrics = _compute_binary_metrics(labels, probs, threshold=threshold)
    preds = (np.asarray(probs) >= float(threshold)).astype(np.int32)
    cm = confusion_matrix(np.asarray(labels, dtype=np.int32), preds, labels=[0, 1])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=160)

    ax_cm = axes[0, 0]
    im = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Bogus", "Real"])
    ax_cm.set_yticklabels(["Bogus", "Real"])
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    ax_cm.set_title(f"{split_name.upper()} Confusion Matrix")
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax_cm.text(col, row, int(cm[row, col]), ha="center", va="center", color="#111111")
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    ax_roc = axes[0, 1]
    if len(np.unique(labels)) > 1:
        fpr, tpr, _ = roc_curve(labels, probs)
        ax_roc.plot(fpr, tpr, color="#2b8cbe", linewidth=2, label=f"AUC={metrics['roc_auc']:.4f}")
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1)
    ax_roc.set_xlim(0.0, 1.0)
    ax_roc.set_ylim(0.0, 1.0)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"{split_name.upper()} ROC Curve")
    ax_roc.grid(alpha=0.2)
    if ax_roc.lines:
        ax_roc.legend(loc="lower right")

    ax_pr = axes[1, 0]
    if len(np.unique(labels)) > 1:
        precision, recall, _ = precision_recall_curve(labels, probs)
        ax_pr.plot(recall, precision, color="#31a354", linewidth=2, label=f"AP={metrics['pr_auc']:.4f}")
    ax_pr.set_xlim(0.0, 1.0)
    ax_pr.set_ylim(0.0, 1.0)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"{split_name.upper()} Precision-Recall")
    ax_pr.grid(alpha=0.2)
    if ax_pr.lines:
        ax_pr.legend(loc="lower left")

    ax_hist = axes[1, 1]
    labels_arr = np.asarray(labels, dtype=np.int32)
    probs_arr = np.asarray(probs, dtype=np.float64)
    bogus_probs = probs_arr[labels_arr == 0]
    real_probs = probs_arr[labels_arr == 1]
    bins = np.linspace(0.0, 1.0, 21)
    if bogus_probs.size:
        ax_hist.hist(bogus_probs, bins=bins, alpha=0.6, color="#de2d26", label="Bogus")
    if real_probs.size:
        ax_hist.hist(real_probs, bins=bins, alpha=0.6, color="#2b8cbe", label="Real")
    ax_hist.axvline(float(threshold), color="#111111", linestyle="--", linewidth=1.5, label=f"Threshold={threshold:.3f}")
    ax_hist.set_xlabel("P(Real)")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title(
        f"{split_name.upper()} Score Distribution\n"
        f"Acc={metrics['accuracy']:.3f}  Prec={metrics['precision']:.3f}  Rec={metrics['recall']:.3f}"
    )
    ax_hist.grid(alpha=0.2)
    ax_hist.legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _save_predictions_csv(
    output_path: Path,
    entries: list[dict[str, Any]],
    labels: np.ndarray,
    probs: np.ndarray,
    *,
    threshold: float,
    split_name: str,
) -> None:
    fieldnames = [
        "split",
        "relative_path",
        "group_key",
        "label",
        "label_name",
        "prob_real",
        "pred_label",
        "pred_label_name",
        "correct",
        "candidate_id",
        "is_manual",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for entry, label, prob in zip(entries, labels.tolist(), probs.tolist()):
            pred_label = 1 if float(prob) >= float(threshold) else 0
            writer.writerow(
                {
                    "split": split_name,
                    "relative_path": entry.get("relative_path", ""),
                    "group_key": entry.get("group_key", ""),
                    "label": int(label),
                    "label_name": "real" if int(label) == 1 else "bogus",
                    "prob_real": float(prob),
                    "pred_label": pred_label,
                    "pred_label_name": "real" if pred_label == 1 else "bogus",
                    "correct": int(pred_label == int(label)),
                    "candidate_id": entry.get("candidate_id", ""),
                    "is_manual": entry.get("is_manual", ""),
                }
            )


def _ensure_manifest(config: LegacyExperimentConfig) -> Path:
    manifest_path = Path(config.manifest_path).resolve()
    if manifest_path.is_file():
        return manifest_path

    logger.info("Manifest not found, building one at %s", manifest_path)
    build_legacy_triplet_manifest(
        config.dataset_dir,
        manifest_path,
        seed=config.seed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )
    return manifest_path


def train_legacy_classifier(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Run the legacy v1 classifier experiment end to end."""

    experiment_config = load_experiment_config(config)
    _set_random_seed(experiment_config.seed)
    _ensure_manifest(experiment_config)
    paths = _prepare_output_paths(experiment_config)

    resolved_device = resolve_device(experiment_config.device)
    device = resolved_device.resolved
    logger.info("Starting experiment: %s", experiment_config.experiment_name)
    logger.info("Training device: %s", resolved_device.message)
    logger.info("Runtime: %s", _torch_runtime_summary())
    if device.type == "cpu" and getattr(torch.version, "cuda", None) is None:
        logger.warning("Current interpreter uses a CPU-only PyTorch build; GPU cannot be selected in this environment")
    logger.info("Dataset root: %s", experiment_config.dataset_dir)
    logger.info("Manifest: %s", paths["manifest_path"])
    logger.info(
        "Config: model=%s pretrained=%s input_mode=%s image_size=%d batch=%d epochs=%d optimizer=%s lr=%.3e scheduler=%s",
        _normalize_model_name(experiment_config.model_name),
        experiment_config.pretrained,
        experiment_config.input_mode,
        int(experiment_config.image_size),
        int(experiment_config.batch_size),
        int(experiment_config.epochs),
        experiment_config.optimizer,
        float(experiment_config.lr),
        experiment_config.scheduler,
    )

    datasets = _build_datasets(experiment_config)
    loaders = _build_loaders(datasets, experiment_config)
    logger.info("Split train: %s", _label_summary(datasets["train"].labels()))
    logger.info("Split val: %s", _label_summary(datasets["val"].labels()))
    logger.info("Split test: %s", _label_summary(datasets["test"].labels()))

    model = create_experiment_model(
        experiment_config.model_name,
        pretrained=experiment_config.pretrained,
    ).to(device)
    logger.info("Model parameters: %s", f"{_parameter_count(model):,}")
    criterion = _criterion_from_config(experiment_config).to(device)
    optimizer = _optimizer_from_config(model, experiment_config)
    scheduler = _scheduler_from_config(optimizer, experiment_config)

    best_score = float("-inf")
    best_epoch = 0
    best_threshold = 0.5
    best_val_metrics: dict[str, Any] | None = None
    history_rows: list[dict[str, Any]] = []
    patience = 0
    epochs_ran = 0

    for epoch in range(1, int(experiment_config.epochs) + 1):
        train_loss, train_seconds = _run_train_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            epoch=epoch,
            total_epochs=int(experiment_config.epochs),
            log_every_n_steps=max(0, int(experiment_config.log_every_n_steps)),
        )
        val_loss, val_probs, val_labels = _run_eval_epoch(model, loaders["val"], criterion, device)
        threshold, val_metrics = _choose_threshold(
            val_labels,
            val_probs,
            metric_name=experiment_config.threshold_metric,
        )

        metric_key = str(experiment_config.selection_metric).strip().lower()
        score = float(val_metrics.get(metric_key, val_metrics["f1"]))
        current_lr = float(optimizer.param_groups[0]["lr"])
        history_rows.append(
            {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_threshold": float(threshold),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_precision": float(val_metrics["precision"]),
                "val_recall": float(val_metrics["recall"]),
                "val_f1": float(val_metrics["f1"]),
                "val_f2": float(val_metrics["f2"]),
                "val_roc_auc": float(val_metrics["roc_auc"]),
                "val_pr_auc": float(val_metrics["pr_auc"]),
            }
        )
        epochs_ran = epoch
        logger.info(
            "Epoch %02d/%02d done | %.1fs | train_loss=%.5f | val_loss=%.5f | val_acc=%.4f | val_f1=%.4f | val_f2=%.4f | val_roc_auc=%.4f | val_pr_auc=%.4f | threshold=%.4f",
            epoch,
            int(experiment_config.epochs),
            float(train_seconds),
            float(train_loss),
            float(val_loss),
            float(val_metrics["accuracy"]),
            float(val_metrics["f1"]),
            float(val_metrics["f2"]),
            float(val_metrics["roc_auc"]),
            float(val_metrics["pr_auc"]),
            float(threshold),
        )

        if score > best_score + 1e-9:
            best_score = score
            best_epoch = epoch
            best_threshold = threshold
            best_val_metrics = dict(val_metrics)
            checkpoint = {
                "state_dict": model.state_dict(),
                "model_name": _normalize_model_name(experiment_config.model_name),
                "threshold": float(best_threshold),
                "best_epoch": int(best_epoch),
                "selection_metric": experiment_config.selection_metric,
                "config": asdict(experiment_config),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            torch.save(checkpoint, paths["checkpoint_path"])
            patience = 0
            logger.info(
                "New best checkpoint | epoch=%d | metric=%s | score=%.4f | saved=%s",
                epoch,
                metric_key,
                float(score),
                paths["checkpoint_path"],
            )
        else:
            patience += 1
            logger.info(
                "No improvement | epoch=%d | metric=%s | score=%.4f | patience=%d/%d",
                epoch,
                metric_key,
                float(score),
                int(patience),
                int(experiment_config.early_stopping_patience),
            )

        if scheduler is not None:
            scheduler.step()

        if patience >= int(experiment_config.early_stopping_patience):
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    if best_val_metrics is None:
        raise RuntimeError("Training finished without producing a valid checkpoint")

    val_result = evaluate_legacy_checkpoint(
        paths["checkpoint_path"],
        split="val",
        batch_size=experiment_config.batch_size,
        num_workers=experiment_config.num_workers,
        device=experiment_config.device,
        return_outputs=True,
    )
    test_result = evaluate_legacy_checkpoint(
        paths["checkpoint_path"],
        split="test",
        batch_size=experiment_config.batch_size,
        num_workers=experiment_config.num_workers,
        device=experiment_config.device,
        return_outputs=True,
    )
    _save_split_distribution_plot(paths["manifest_path"], paths["split_distribution_plot"])
    _save_learning_curve_plot(history_rows, paths["learning_curve_plot"])
    _save_analysis_figure(
        np.asarray(val_result["labels"]),
        np.asarray(val_result["probs"]),
        threshold=float(best_threshold),
        split_name="val",
        output_path=paths["val_analysis_plot"],
    )
    _save_analysis_figure(
        np.asarray(test_result["labels"]),
        np.asarray(test_result["probs"]),
        threshold=float(best_threshold),
        split_name="test",
        output_path=paths["test_analysis_plot"],
    )
    _save_predictions_csv(
        paths["val_predictions_path"],
        list(val_result["entries"]),
        np.asarray(val_result["labels"]),
        np.asarray(val_result["probs"]),
        threshold=float(best_threshold),
        split_name="val",
    )
    _save_predictions_csv(
        paths["test_predictions_path"],
        list(test_result["entries"]),
        np.asarray(test_result["labels"]),
        np.asarray(test_result["probs"]),
        threshold=float(best_threshold),
        split_name="test",
    )

    summary = {
        "experiment_name": experiment_config.experiment_name,
        "model_name": _normalize_model_name(experiment_config.model_name),
        "input_mode": experiment_config.input_mode,
        "image_size": int(experiment_config.image_size),
        "resize_mode": experiment_config.resize_mode,
        "pretrained": bool(experiment_config.pretrained),
        "seed": int(experiment_config.seed),
        "batch_size": int(experiment_config.batch_size),
        "lr": float(experiment_config.lr),
        "epochs_requested": int(experiment_config.epochs),
        "epochs_ran": int(epochs_ran),
        "best_epoch": int(best_epoch),
        "best_threshold": float(best_threshold),
        "val_accuracy": float(val_result["accuracy"]),
        "val_precision": float(val_result["precision"]),
        "val_recall": float(val_result["recall"]),
        "val_f1": float(val_result["f1"]),
        "val_f2": float(val_result["f2"]),
        "val_roc_auc": float(val_result["roc_auc"]),
        "val_pr_auc": float(val_result["pr_auc"]),
        "test_accuracy": float(test_result["accuracy"]),
        "test_precision": float(test_result["precision"]),
        "test_recall": float(test_result["recall"]),
        "test_f1": float(test_result["f1"]),
        "test_f2": float(test_result["f2"]),
        "test_roc_auc": float(test_result["roc_auc"]),
        "test_pr_auc": float(test_result["pr_auc"]),
        "test_tn": int(test_result["tn"]),
        "test_fp": int(test_result["fp"]),
        "test_fn": int(test_result["fn"]),
        "test_tp": int(test_result["tp"]),
        "manifest_path": str(paths["manifest_path"]),
        "checkpoint_path": str(paths["checkpoint_path"]),
        "history_path": str(paths["history_path"]),
        "plots_dir": str(paths["plots_dir"]),
        "split_distribution_plot": str(paths["split_distribution_plot"]),
        "learning_curve_plot": str(paths["learning_curve_plot"]),
        "val_analysis_plot": str(paths["val_analysis_plot"]),
        "test_analysis_plot": str(paths["test_analysis_plot"]),
        "val_predictions_path": str(paths["val_predictions_path"]),
        "test_predictions_path": str(paths["test_predictions_path"]),
    }

    logger.info(
        "Final summary | best_epoch=%d | val_f1=%.4f | val_f2=%.4f | test_f1=%.4f | test_f2=%.4f | test_recall=%.4f",
        int(best_epoch),
        float(val_result["f1"]),
        float(val_result["f2"]),
        float(test_result["f1"]),
        float(test_result["f2"]),
        float(test_result["recall"]),
    )
    logger.info("Plots directory: %s", paths["plots_dir"])
    logger.info("Summary json: %s", paths["summary_path"])

    paths["config_snapshot_path"].write_text(
        json.dumps(asdict(experiment_config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_history(paths["history_path"], history_rows)
    paths["summary_path"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _append_summary_csv(paths["summary_csv_path"], summary)
    return summary


def evaluate_legacy_checkpoint(
    checkpoint_path: str | Path,
    *,
    split: str = "test",
    manifest_path: str | Path | None = None,
    dataset_dir: str | Path | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    device: str | None = None,
    return_outputs: bool = False,
) -> dict[str, Any]:
    """Load a saved checkpoint and evaluate one split."""

    checkpoint_file = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    config = load_experiment_config(checkpoint.get("config") or {})

    if manifest_path is not None:
        config.manifest_path = str(Path(manifest_path).resolve())
    if dataset_dir is not None:
        config.dataset_dir = str(Path(dataset_dir).resolve())
    if batch_size is not None:
        config.batch_size = int(batch_size)
    if num_workers is not None:
        config.num_workers = int(num_workers)
    if device is not None:
        config.device = str(device)

    resolved_device = resolve_device(config.device)
    runtime_device = resolved_device.resolved
    logger.info("Evaluation device: %s", resolved_device.message)

    dataset = LegacyTripletExperimentDataset(
        config.manifest_path,
        split=split,
        dataset_root=config.dataset_dir,
        input_mode=config.input_mode,
        image_size=config.image_size,
        resize_mode=config.resize_mode,
        normalize=config.normalize,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    model = create_experiment_model(checkpoint.get("model_name", config.model_name), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(runtime_device)

    criterion = _criterion_from_config(config).to(runtime_device)
    loss, probs, labels = _run_eval_epoch(model, loader, criterion, runtime_device)
    threshold = float(checkpoint.get("threshold", 0.5))
    metrics = _compute_binary_metrics(labels, probs, threshold=threshold)
    metrics["loss"] = float(loss)
    metrics["split"] = str(split).strip().lower()
    metrics["checkpoint_path"] = str(checkpoint_file)
    if return_outputs:
        metrics["probs"] = probs
        metrics["labels"] = labels
        metrics["entries"] = [dataset.entry(index) for index in range(len(dataset))]
    return metrics
