from __future__ import annotations

import json
from pathlib import Path

from scann.native_annotation.training_job_worker import (
    RemoteTrainingWorkerConfig,
    TrainingExecutionConfig,
    TrainingExecutionResult,
    TrainingJobWorkerRunner,
)
from scann.native_annotation.training_lifecycle_service import TrainingArtifactUploadResponse, TrainingWorkerClaimResponse


class _FakeTrainingClient:
    def __init__(self, jobs: list[TrainingWorkerClaimResponse] | None = None) -> None:
        self.jobs = list(jobs or [])
        self.snapshots: dict[str, bytes] = {}
        self.uploads: list[tuple[str, bytes]] = []
        self.completions: list[tuple[str, str, TrainingExecutionResult]] = []
        self.failures: list[tuple[str, str, bool]] = []
        self.heartbeats: list[str] = []

    def claim_job(self):
        if not self.jobs:
            return None
        return self.jobs.pop(0)

    def heartbeat_job(self, job_id: str):
        self.heartbeats.append(job_id)
        return None

    def fetch_snapshot(self, job_id: str) -> bytes:
        return self.snapshots[job_id]

    def upload_artifact(self, job_id: str, model_path: Path) -> TrainingArtifactUploadResponse:
        data = model_path.read_bytes()
        self.uploads.append((job_id, data))
        return TrainingArtifactUploadResponse(
            job_id=job_id,
            artifact_path=f".scann_control/models/{job_id}/best_model.pth",
        )

    def complete_job(self, job_id: str, *, artifact_path: str, result: TrainingExecutionResult):
        self.completions.append((job_id, artifact_path, result))
        return {"job_id": job_id}

    def fail_job(self, job_id: str, *, error_message: str, retryable: bool = False):
        self.failures.append((job_id, error_message, retryable))
        return None


def _worker_config(tmp_path: Path) -> RemoteTrainingWorkerConfig:
    return RemoteTrainingWorkerConfig(
        server_url="http://127.0.0.1:8000",
        worker_token="training-worker-secret",
        worker_id="trainer-1",
        display_name="trainer-1",
        host_name="gpu-pc-01",
        device_label="RTX-4090",
        execution=TrainingExecutionConfig(
            dataset_root=tmp_path / "dataset",
            output_root=tmp_path / "worker-output",
            task_types=["classification"],
            model_backbones=["ViT_B_16"],
            device="cpu",
        ),
        idle_poll_seconds=0.01,
        heartbeat_interval_seconds=60.0,
        request_timeout_seconds=5.0,
    )


def test_training_job_worker_runner_reports_completion(tmp_path) -> None:
    config = _worker_config(tmp_path)
    job = TrainingWorkerClaimResponse(
        job_id="train-job-1",
        snapshot_id="snapshot-1",
        task_type="classification",
        model_version="cls-v3",
        model_id="cls-v3-run-001",
        model_backbone="ViT_B_16",
        train_config={"epochs": 2},
    )
    client = _FakeTrainingClient([job])
    client.snapshots[job.job_id] = json.dumps({"version": "2.3", "images": [{"id": "PGC 17069"}]}).encode("utf-8")

    def _trainer(params: dict) -> TrainingExecutionResult:
        save_path = Path(params["save_path"])
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(b"trained-model")
        return TrainingExecutionResult(
            model_path=save_path,
            metrics={"f1": 0.95},
            metadata={"epochs": params["epochs"], "task_type": params["task_type"]},
        )

    runner = TrainingJobWorkerRunner(config, client=client, trainer=_trainer)

    handled = runner.run_once()

    assert handled is True
    snapshot_path = config.execution.output_root / "snapshots" / f"{job.snapshot_id}.json"
    assert snapshot_path.exists()
    assert json.loads(snapshot_path.read_text(encoding="utf-8"))["images"][0]["id"] == "PGC 17069"
    assert client.uploads == [("train-job-1", b"trained-model")]
    assert len(client.completions) == 1
    completed_job_id, artifact_path, result = client.completions[0]
    assert completed_job_id == "train-job-1"
    assert artifact_path.endswith("/best_model.pth")
    assert result.metrics["f1"] == 0.95
    assert client.failures == []


def test_training_job_worker_runner_reports_failure(tmp_path) -> None:
    config = _worker_config(tmp_path)
    job = TrainingWorkerClaimResponse(
        job_id="train-job-2",
        snapshot_id="snapshot-2",
        task_type="classification",
        model_version="cls-v4",
        model_id="cls-v4-run-001",
        model_backbone="ResNet18",
        train_config={"epochs": 1},
    )
    client = _FakeTrainingClient([job])
    client.snapshots[job.job_id] = b'{"version":"2.3","images":[]}'

    def _trainer(_params: dict) -> TrainingExecutionResult:
        raise RuntimeError("training exploded")

    runner = TrainingJobWorkerRunner(config, client=client, trainer=_trainer)

    handled = runner.run_once()

    assert handled is True
    assert client.completions == []
    assert client.failures == [("train-job-2", "training exploded", False)]
