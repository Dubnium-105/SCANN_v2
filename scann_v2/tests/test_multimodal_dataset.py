from __future__ import annotations

import numpy as np

from scann.ai.dataset import MultimodalRecordDataset


def test_multimodal_dataset_preserves_view_alignment_and_missing_mask():
    base = np.arange(16, dtype=np.float32).reshape(4, 4)
    dataset = MultimodalRecordDataset(
        [
            {
                "data": np.stack(
                    [base, base + 100, base + 200],
                    axis=0,
                ),
                "detail_type": "supernova",
                "candidate_features": {
                    "snr": 8.0,
                    "fwhm": None,
                },
            }
        ],
        split="validation",
        resize=4,
        feature_names=("snr", "fwhm"),
        augment=False,
    )

    sample = dataset[0]

    assert sample["images"].shape == (3, 3, 4, 4)
    assert np.allclose(
        (
            sample["images"][1, 0]
            - sample["images"][0, 0]
        ).numpy(),
        100.0,
    )
    assert sample["structured_values"].tolist() == [8.0, 0.0]
    assert sample["structured_mask"].tolist() == [True, False]
    assert sample["review_action"].item() >= 0
    assert sample["phenomenon_family"].item() >= 0
    assert sample["detail_type"].item() >= 0
