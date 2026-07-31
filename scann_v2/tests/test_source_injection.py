from __future__ import annotations

import numpy as np

from scann.ai.source_injection import (
    inject_point_sources,
    recovery_curve,
)
from scann.core.candidate_detector import DetectionParams, detect_candidates


def test_source_injection_is_deterministic_and_non_destructive():
    rng = np.random.default_rng(3)
    image = rng.normal(100.0, 2.0, size=(128, 128)).astype(np.float32)
    original = image.copy()

    first = inject_point_sources(
        image,
        count=6,
        seed=17,
        snr_values=(8.0,),
        fwhm_values=(3.0,),
    )
    second = inject_point_sources(
        image,
        count=6,
        seed=17,
        snr_values=(8.0,),
        fwhm_values=(3.0,),
    )

    assert np.array_equal(image, original)
    assert np.array_equal(first.image, second.image)
    assert first.sources == second.sources
    assert len(first.truth_boxes()) == 6


def test_significance_detector_recovers_positive_and_negative_sources():
    rng = np.random.default_rng(11)
    old_image = rng.normal(100.0, 1.0, size=(160, 160)).astype(np.float32)
    positive = inject_point_sources(
        old_image,
        count=2,
        seed=7,
        snr_values=(30.0,),
        fwhm_values=(3.0,),
        polarity_values=(1, -1),
    )

    candidates = detect_candidates(
        positive.image,
        old_image,
        params=DetectionParams(
            detector="significance_v1",
            significance_sigma=4.0,
            min_area=1,
            max_area=100,
            edge_margin=3,
            topk=20,
        ),
    )

    assert candidates
    assert all(candidate.features.snr >= 4.0 for candidate in candidates)
    assert {candidate.polarity for candidate in candidates}.issubset({-1, 1})


def test_recovery_curve_preserves_empty_edge_support():
    image = np.zeros((96, 96), dtype=np.float32)
    injection = inject_point_sources(
        image,
        count=3,
        seed=4,
        snr_values=(5.0,),
        fwhm_values=(2.0,),
    )
    curve = recovery_curve(
        injection.sources,
        [injection.sources[0].source_id],
    )

    assert curve["total"]["support"] == 3
    assert curve["total"]["recovered"] == 1
    assert "5.0" in curve["by_snr"]
