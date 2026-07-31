from __future__ import annotations

import numpy as np

from scann.ai.detection_trace import (
    DETECTION_TRACE_VERSION,
    DetectionTrace,
    image_statistics,
)
from scann.core.candidate_detector import DetectionParams
from scann.services.detection_service import DetectionPipeline


def test_image_statistics_handles_nonfinite_pixels():
    image = np.asarray([[1.0, np.nan], [np.inf, 3.0]], dtype=np.float32)
    statistics = image_statistics(image)

    assert statistics["shape"] == [2, 2]
    assert statistics["finite_fraction"] == 0.5
    assert statistics["median"] == 2.0


def test_detection_trace_is_emitted_without_changing_legacy_result():
    old_image = np.full((64, 64), 100.0, dtype=np.float32)
    new_image = old_image.copy()
    new_image[30:34, 30:34] += 200.0
    pipeline = DetectionPipeline(
        detection_params=DetectionParams(
            thresh=50,
            kill_flat=False,
            kill_dipole=False,
        )
    )

    result = pipeline.process_pair(
        "trace-task",
        new_image,
        old_image,
        skip_align=True,
    )

    assert result.trace is not None
    assert result.trace.trace_version == DETECTION_TRACE_VERSION
    assert result.trace.detector_version == "legacy"
    assert result.trace.stage_counts["standard"] == len(result.candidates)
    assert result.trace.stage_counts["final"] == len(result.candidates)
    assert result.trace.alignment["reason"] == "skip_align"
    assert result.trace.finished_at is not None


def test_trace_serialization_does_not_expose_monotonic_clock():
    trace = DetectionTrace(
        pair_name="task",
        detector_version="legacy",
        detection_mode="patch",
    )
    trace.finish()
    payload = trace.to_dict()

    assert "_started_monotonic" not in payload
    assert payload["final_candidate_count"] == 0
