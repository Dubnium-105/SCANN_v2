from unittest.mock import Mock

import numpy as np

from scann.services.detection_window_scanner import sliding_window_detect


def test_sliding_window_detect_returns_thresholded_candidates_for_small_image():
    inference_engine = Mock()
    inference_engine.is_ready = True
    inference_engine.is_v1 = False
    inference_engine.threshold = 0.6
    inference_engine._channel_order = (0, 1, 2)
    inference_engine.classify_patches.return_value = [0.75]

    seen = []

    def prepare_triplet_patch_fn(new_data, old_data, x, y, size):
        seen.append((x, y, size))
        return np.ones((3, 224, 224), dtype=np.float32)

    def nms_candidates_fn(candidates, min_dist):
        return candidates

    result = sliding_window_detect(
        np.ones((16, 16), dtype=np.float32),
        np.ones((16, 16), dtype=np.float32),
        inference_engine=inference_engine,
        patch_size=16,
        is_v1_model=False,
        extract_patch_fn=lambda image, x, y, size: np.ones((size, size), dtype=np.float32),
        prepare_triplet_patch_fn=prepare_triplet_patch_fn,
        nms_candidates_fn=nms_candidates_fn,
    )

    assert seen == [(8, 8, 16)]
    assert len(result) == 1
    assert (result[0].x, result[0].y) == (8, 8)
    assert result[0].ai_score == 0.75


def test_sliding_window_detect_v1_retries_without_valid_filter_when_needed():
    inference_engine = Mock()
    inference_engine.is_ready = True
    inference_engine.is_v1 = True
    inference_engine.threshold = 0.5
    inference_engine._channel_order = (0, 1, 2)
    inference_engine.classify_patches.return_value = [0.9, 0.2, 0.8, 0.1]

    old_data = np.zeros((12, 12), dtype=np.float32)
    new_data = np.ones((12, 12), dtype=np.float32)

    def extract_patch_fn(image, x, y, size):
        if image is old_data:
            return np.zeros((size, size), dtype=np.float32)
        return np.ones((size, size), dtype=np.float32)

    result = sliding_window_detect(
        new_data,
        old_data,
        inference_engine=inference_engine,
        patch_size=6,
        is_v1_model=True,
        extract_patch_fn=extract_patch_fn,
        prepare_triplet_patch_fn=lambda new_img, old_img, x, y, size: np.ones((3, 224, 224), dtype=np.float32),
        nms_candidates_fn=lambda candidates, min_dist: candidates,
    )

    assert [(candidate.x, candidate.y, candidate.ai_score) for candidate in result] == [
        (3, 3, 0.9),
        (9, 3, 0.8),
    ]