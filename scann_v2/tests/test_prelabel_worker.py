from __future__ import annotations

from pathlib import Path

import numpy as np

from scann.core.models import FitsHeader, FitsImage
from scann.native_annotation.prelabel_service import WorkerClaimResponse
from scann.native_annotation.prelabel_worker import (
    DetectionPrelabelProcessor,
    PrelabelProcessingResult,
    PrelabelTaskAssetResolver,
    PrelabelWorkerConfig,
    PrelabelWorkerRunner,
    WorkerDetectionConfig,
)


class _FakeClient:
    def __init__(self, jobs: list[WorkerClaimResponse] | None = None) -> None:
        self.jobs = list(jobs or [])
        self.completed: list[tuple[str, PrelabelProcessingResult]] = []
        self.failed: list[tuple[str, str, bool]] = []
        self.heartbeats: list[str] = []
        self.fetched: list[tuple[str, str]] = []

    def claim_job(self):
        if not self.jobs:
            return None
        return self.jobs.pop(0)

    def complete_job(self, job_id: str, result: PrelabelProcessingResult):
        self.completed.append((job_id, result))
        return None

    def fail_job(self, job_id: str, *, error_message: str, retryable: bool = False):
        self.failed.append((job_id, error_message, retryable))
        return None

    def heartbeat_job(self, job_id: str):
        self.heartbeats.append(job_id)
        return None

    def fetch_job_fits(self, job_id: str, view_name: str) -> bytes:
        self.fetched.append((job_id, view_name))
        return b"unused"


class _FakeProcessor:
    def __init__(self, result: PrelabelProcessingResult | None = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[str] = []

    def process(self, job, assets):
        self.calls.append(job.job_id)
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def _worker_config(tmp_path: Path) -> PrelabelWorkerConfig:
    return PrelabelWorkerConfig(
        server_url="http://127.0.0.1:8000",
        worker_token="worker-secret",
        worker_id="worker-1",
        display_name="worker-1",
        host_name="pc-01",
        device_label="RTX",
        dataset_root=tmp_path,
        detection=WorkerDetectionConfig(model_path="dummy.pt", model_version="detector-v1"),
        idle_poll_seconds=0.01,
        heartbeat_interval_seconds=60.0,
        request_timeout_seconds=5.0,
    )


def test_prelabel_worker_runner_reports_completion(tmp_path) -> None:
    config = _worker_config(tmp_path)
    job = WorkerClaimResponse(
        job_id="job-1",
        task_id="task-1",
        model_version="detector-v1",
        input_fingerprint="abc",
        paths={"new": "new/task-1.fts"},
    )
    result = PrelabelProcessingResult(
        source_view="new",
        ai_suggestion="real",
        ai_confidence=0.91,
        annotations=[],
        metadata={"candidate_count": 0},
    )
    client = _FakeClient([job])
    processor = _FakeProcessor(result=result)
    runner = PrelabelWorkerRunner(config, client=client, processor=processor)

    handled = runner.run_once()

    assert handled is True
    assert processor.calls == ["job-1"]
    assert client.completed == [("job-1", result)]
    assert client.failed == []


def test_worker_config_does_not_advertise_auto_backbone(tmp_path) -> None:
    config = _worker_config(tmp_path)

    assert config.detection.model_backbone == "auto"
    assert config.supported_model_backbones == []


def test_prelabel_worker_runner_reports_failure(tmp_path) -> None:
    config = _worker_config(tmp_path)
    job = WorkerClaimResponse(
        job_id="job-2",
        task_id="task-2",
        model_version="detector-v1",
        input_fingerprint="xyz",
        paths={"new": "new/task-2.fts"},
    )
    client = _FakeClient([job])
    processor = _FakeProcessor(error=RuntimeError("model exploded"))
    runner = PrelabelWorkerRunner(config, client=client, processor=processor)

    handled = runner.run_once()

    assert handled is True
    assert client.completed == []
    assert client.failed == [("job-2", "model exploded", False)]


def test_task_asset_resolver_falls_back_to_remote_fetch(tmp_path, monkeypatch) -> None:
    config = _worker_config(tmp_path / "missing-root")
    job = WorkerClaimResponse(
        job_id="job-3",
        task_id="task-3",
        model_version="detector-v1",
        input_fingerprint="zzz",
        paths={"new": "new/task-3.fts"},
    )
    client = _FakeClient()
    resolver = PrelabelTaskAssetResolver(config, client, job)

    def _fake_from_bytes(_data: bytes) -> FitsImage:
        return FitsImage(
            data=np.ones((8, 8), dtype=np.float32),
            header=FitsHeader(raw={}),
        )

    monkeypatch.setattr(
        PrelabelTaskAssetResolver,
        "_read_fits_from_bytes",
        staticmethod(_fake_from_bytes),
    )

    image = resolver.load_fits("new")

    assert image is not None
    assert image.data.shape == (8, 8)
    assert client.fetched == [("job-3", "new")]


def test_detection_processor_candidate_mapping(tmp_path, monkeypatch) -> None:
    config = _worker_config(tmp_path)
    processor = DetectionPrelabelProcessor.__new__(DetectionPrelabelProcessor)
    processor.config = config
    processor.inference_engine = None

    class _FakePipeline:
        def __init__(self) -> None:
            self.detection_params = type("Params", (), {"topk": 20})()

        def process_pair(self, **_kwargs):
            class _Candidate:
                x = 40
                y = 50
                ai_score = 0.88
                bbox_x = 33
                bbox_y = 46
                bbox_width = 11
                bbox_height = 7

            class _Result:
                candidates = [_Candidate()]
                error = ""

            return _Result()

    processor.pipeline = _FakePipeline()

    class _FakeAssets:
        def load_fits(self, view_name: str):
            if view_name == "new":
                return FitsImage(data=np.zeros((100, 120), dtype=np.float32), header=FitsHeader(raw={}))
            return FitsImage(data=np.zeros((100, 120), dtype=np.float32), header=FitsHeader(raw={}))

    result = processor.process(
        WorkerClaimResponse(
            job_id="job-4",
            task_id="task-4",
            model_version="detector-v1",
            candidate_limit=5,
            confidence_threshold=0.5,
            input_fingerprint="123",
            paths={"new": "new/task-4.fts"},
        ),
        _FakeAssets(),
    )

    assert result.ai_suggestion == "real"
    assert result.ai_confidence == 0.88
    assert len(result.annotations) == 1
    assert result.annotations[0].x == 33
    assert result.annotations[0].y == 46
    assert result.annotations[0].width == 11
    assert result.annotations[0].height == 7
    assert result.annotations[0].label == "real"
    assert result.metadata["candidate_limit"] == 5
    assert result.metadata["confidence_threshold"] == 0.5


def test_detection_processor_applies_threshold_and_limit(tmp_path) -> None:
    config = _worker_config(tmp_path)
    processor = DetectionPrelabelProcessor.__new__(DetectionPrelabelProcessor)
    processor.config = config

    class _FakeEngine:
        def __init__(self) -> None:
            self.threshold = 0.25

    processor.inference_engine = _FakeEngine()

    class _FakePipeline:
        def __init__(self) -> None:
            self.detection_params = type("Params", (), {"topk": 20})()

        def process_pair(self, **_kwargs):
            candidates = []
            for index, score in enumerate([0.92, 0.61, 0.40]):
                candidate = type(
                    "Candidate",
                    (),
                    {
                        "x": 20 + index * 10,
                        "y": 30 + index * 10,
                        "ai_score": score,
                    },
                )()
                candidates.append(candidate)

            return type("Result", (), {"candidates": candidates, "error": ""})()

    processor.pipeline = _FakePipeline()

    class _FakeAssets:
        def load_fits(self, _view_name: str):
            return FitsImage(data=np.zeros((120, 120), dtype=np.float32), header=FitsHeader(raw={}))

    result = processor.process(
        WorkerClaimResponse(
            job_id="job-5",
            task_id="task-5",
            model_version="detector-v1",
            candidate_limit=2,
            confidence_threshold=0.6,
            input_fingerprint="abc",
            paths={"new": "new/task-5.fts"},
        ),
        _FakeAssets(),
    )

    assert len(result.annotations) == 2
    assert [round(item.confidence, 2) for item in result.annotations] == [0.92, 0.61]
    assert result.metadata["raw_candidate_count"] == 3
    assert result.metadata["candidate_count"] == 2
    assert processor.inference_engine.threshold == 0.25
    assert processor.pipeline.detection_params.topk == 20
