"""训练流程冒烟测试（历史脚本重构版）。"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models

from scann.ai.dataset import TripletPNGDataset


def _write_triplet_png(path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    img = np.zeros((80, 240), dtype=np.uint8)
    img[:, :80] = rng.integers(20, 100, size=(80, 80), dtype=np.uint8)
    img[:, 80:160] = rng.integers(80, 160, size=(80, 80), dtype=np.uint8)
    img[:, 160:] = rng.integers(120, 220, size=(80, 80), dtype=np.uint8)
    Image.fromarray(img).save(path)


def test_training_smoke_with_weighted_sampler(tmp_path):
    _ = pytest.importorskip("torch")

    pos_dir = tmp_path / "positive"
    neg_dir = tmp_path / "negative"
    pos_dir.mkdir()
    neg_dir.mkdir()

    for i in range(4):
        _write_triplet_png(pos_dir / f"pos_{i}.png", seed=100 + i)
    for i in range(4):
        _write_triplet_png(neg_dir / f"neg_{i}.png", seed=200 + i)

    samples = [(str(p), 1) for p in sorted(pos_dir.glob("*.png"))] + [
        (str(p), 0) for p in sorted(neg_dir.glob("*.png"))
    ]

    train_set = TripletPNGDataset(samples=samples, split="train", resize=64, augment=True)
    labels = [y for _, y in samples]
    count_neg = labels.count(0)
    count_pos = labels.count(1)
    weights = [1.0 / max(count_neg, 1), 1.0 / max(count_pos, 1)]
    sample_weights = torch.tensor([weights[y] for y in labels], dtype=torch.double)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_set), replacement=True)
    train_loader = DataLoader(train_set, batch_size=4, sampler=sampler, num_workers=0)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    x, y = next(iter(train_loader))
    y = y.float().unsqueeze(1)
    logits = model(x)
    loss = criterion(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss).item()
