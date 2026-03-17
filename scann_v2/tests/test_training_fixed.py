"""训练流程修复回归测试（历史脚本重构版）。"""

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
    img = np.zeros((80, 240), dtype=np.uint8)
    img[:, :80] = rng.integers(0, 60, size=(80, 80), dtype=np.uint8)
    img[:, 80:160] = rng.integers(60, 140, size=(80, 80), dtype=np.uint8)
    img[:, 160:] = rng.integers(140, 255, size=(80, 80), dtype=np.uint8)
    Image.fromarray(img).save(path)


def test_training_fixed_focal_loss_smoke(tmp_path):
    _ = pytest.importorskip("torch")

    pos_dir = tmp_path / "positive"
    neg_dir = tmp_path / "negative"
    pos_dir.mkdir()
    neg_dir.mkdir()

    for i in range(3):
        _write_triplet_png(pos_dir / f"p_{i}.png", seed=10 + i)
        _write_triplet_png(neg_dir / f"n_{i}.png", seed=20 + i)

    samples = [(str(p), 1) for p in sorted(pos_dir.glob("*.png"))] + [
        (str(p), 0) for p in sorted(neg_dir.glob("*.png"))
    ]
    dataset = TripletPNGDataset(samples=samples, split="train", resize=64, augment=True)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    device = torch.device("cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = FocalLoss(gamma=2.0, alpha=[1.0, 1.5]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

    model.train()
    images, labels = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert outputs.shape[1] == 2
    assert torch.isfinite(loss).item()
