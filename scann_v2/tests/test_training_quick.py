"""快速训练回归测试（历史脚本重构版）。"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import models

from scann.ai.dataset import TripletPNGDataset
from scann.ai.trainer import FocalLoss


def _write_triplet_png(path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    image = np.zeros((80, 240), dtype=np.uint8)
    image[:, :80] = rng.integers(0, 80, size=(80, 80), dtype=np.uint8)
    image[:, 80:160] = rng.integers(80, 160, size=(80, 80), dtype=np.uint8)
    image[:, 160:] = rng.integers(160, 255, size=(80, 80), dtype=np.uint8)
    Image.fromarray(image).save(path)


def test_training_quick_dataset_and_model_forward(tmp_path):
    _ = pytest.importorskip("torch")

    pos = tmp_path / "positive"
    neg = tmp_path / "negative"
    pos.mkdir()
    neg.mkdir()

    for i in range(2):
        _write_triplet_png(pos / f"pos_{i}.png", seed=1000 + i)
        _write_triplet_png(neg / f"neg_{i}.png", seed=2000 + i)

    samples = [(str(p), 1) for p in sorted(pos.glob("*.png"))] + [
        (str(p), 0) for p in sorted(neg.glob("*.png"))
    ]
    train_dataset = TripletPNGDataset(samples=samples, split="train", resize=64, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    criterion = FocalLoss(gamma=2.0, alpha=[1.0, 1.5])
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

    images, labels = next(iter(train_loader))
    logits = model(images)
    loss = criterion(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert logits.shape == (2, 2)
    assert torch.isfinite(loss).item()
