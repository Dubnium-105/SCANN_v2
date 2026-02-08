"""训练器模块单元测试"""

import numpy as np
import pytest


class TestFocalLoss:
    """测试 Focal Loss"""

    def test_focal_loss_output_scalar(self):
        torch = pytest.importorskip("torch")
        from scann.ai.trainer import FocalLoss

        loss_fn = FocalLoss(alpha=1.0, gamma=2.0)
        logits = torch.randn(8, 2)
        labels = torch.randint(0, 2, (8,))
        loss = loss_fn(logits, labels)
        assert loss.dim() == 0  # 标量

    def test_focal_loss_nonnegative(self):
        torch = pytest.importorskip("torch")
        from scann.ai.trainer import FocalLoss

        loss_fn = FocalLoss()
        logits = torch.randn(16, 2)
        labels = torch.randint(0, 2, (16,))
        loss = loss_fn(logits, labels)
        assert loss.item() >= 0


class TestMetrics:
    """测试指标计算"""

    def test_confusion_matrix_shape(self):
        from scann.ai.trainer import compute_confusion_matrix

        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm.shape == (2, 2)

    def test_compute_metrics_keys(self):
        from scann.ai.trainer import compute_metrics

        y_true = [0, 1, 1, 0, 1, 0]
        y_pred = [0, 1, 0, 0, 1, 1]
        m = compute_metrics(y_true, y_pred)
        assert "precision" in m
        assert "recall" in m
        assert "f1" in m
        assert "accuracy" in m

    def test_perfect_prediction(self):
        from scann.ai.trainer import compute_metrics

        y = [0, 1, 1, 0]
        m = compute_metrics(y, y)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)

    def test_find_threshold_for_recall(self):
        from scann.ai.trainer import find_threshold_for_recall

        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        labels = [0, 0, 1, 1, 1]
        thresh = find_threshold_for_recall(scores, labels, target_recall=0.9)
        assert 0.0 <= thresh <= 1.0
