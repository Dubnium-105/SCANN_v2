from __future__ import annotations

import json

from scann.ai.candidate_evaluation import (
    EvaluationBox,
    evaluate_candidate_records,
    match_boxes,
    write_evaluation_artifact,
)


def test_match_boxes_uses_iou_or_center_distance():
    truth = [
        EvaluationBox(x=10, y=10, width=8, height=8),
        EvaluationBox(x=40, y=40, width=8, height=8),
    ]
    candidates = [
        EvaluationBox(x=11, y=11, width=8, height=8),
        EvaluationBox(x=100, y=100, width=8, height=8),
    ]

    matches, missed, unmatched = match_boxes(truth, candidates)

    assert len(matches) == 1
    assert missed == [1]
    assert unmatched == [1]


def test_candidate_metrics_report_recall_volume_and_unsupported_classes():
    metrics, per_task = evaluate_candidate_records(
        [
            {
                "task_id": "task-1",
                "truth": [
                    {
                        "x": 10,
                        "y": 10,
                        "width": 8,
                        "height": 8,
                        "detail_type": "supernova",
                    }
                ],
                "candidates": [
                    {
                        "x": 11,
                        "y": 11,
                        "width": 8,
                        "height": 8,
                        "score": 0.8,
                    }
                ],
                "trace": {
                    "duration_ms": 12.0,
                    "stage_counts": {"standard": 3, "post_ai": 1},
                },
            },
            {
                "task_id": "task-2",
                "truth": [],
                "candidates": [],
                "raw_candidate_count": 0,
            },
        ]
    )

    assert metrics["recall"] == 1.0
    assert metrics["task_count"] == 2
    assert metrics["raw_candidates_per_task"]["p95"] is not None
    assert metrics["detail_type_metrics"]["supernova"]["support"] == 1
    assert per_task[1]["recall"] is None


def test_evaluation_artifact_contains_hashed_manifest(tmp_path):
    result = write_evaluation_artifact(
        tmp_path,
        run_id="evaluation-1",
        run_type="candidate",
        config={"seed": 42},
        metrics={"recall": 1.0},
        per_task=[{"task_id": "task-1", "recall": 1.0}],
    )

    manifest = json.loads(
        (tmp_path / "evaluation-1" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["manifest_sha256"] == result["manifest_sha256"]
    assert manifest["files"]["metrics.json"]["sha256"]
    assert (tmp_path / "evaluation-1" / "per_task.jsonl").is_file()
