from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import requests

from scann.ai.training_worker import TrainingWorker

from .training_lifecycle_service import (
    TrainingArtifactUploadResponse,
    TrainingWorkerClaimRequest,
    TrainingWorkerClaimResponse,
    TrainingWorkerCompleteRequest,
    TrainingWorkerFailRequest,
    TrainingWorkerHeartbeatRequest,
    TrainingWorkerJobAckResponse,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingExecutionConfig:
    dataset_root: Path
    output_root: Path
    task_types: list[str] = field(default_factory=lambda: ["classification"])
    model_backbones: list[str] = field(default_factory=list)
    device: str = "auto"


@dataclass(frozen=True)
class RemoteTrainingWorkerConfig:
    server_url: str
    worker_token: str
    worker_id: str
    display_name: str
    host_name: str
    device_label: str | None
    execution: TrainingExecutionConfig
    idle_poll_seconds: float = 10.0
    heartbeat_interval_seconds: float = 60.0
    request_timeout_seconds: float = 120.0


@dataclass(frozen=True)
class TrainingExecutionResult:
    model_path: Path
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class TrainingJobWorkerClient:
    def __init__(self, config: RemoteTrainingWorkerConfig, session: requests.Session | None = None) -> None:
        self.config = config
        self._session = session or requests.Session()

    def _url(self, path: str) -> str:
        return f"{self.config.server_url.rstrip('/')}{path}"

    def _headers(self) -> dict[str, str]:
        return {"X-SCANN-Worker-Token": self.config.worker_token}

    def claim_job(self) -> TrainingWorkerClaimResponse | None:
        payload = TrainingWorkerClaimRequest(
            worker_id=self.config.worker_id,
            display_name=self.config.display_name,
            host_name=self.config.host_name,
            device_label=self.config.device_label,
            capabilities={
                "task_types": self.config.execution.task_types,
                "model_backbones": self.config.execution.model_backbones,
            },
        )
        response = self._session.post(
            self._url("/api/training-jobs/claim"),
            headers=self._headers(),
            json=payload.model_dump(),
            timeout=self.config.request_timeout_seconds,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return TrainingWorkerClaimResponse.model_validate(response.json())

    def heartbeat_job(self, job_id: str) -> TrainingWorkerJobAckResponse:
        payload = TrainingWorkerHeartbeatRequest(worker_id=self.config.worker_id)
        response = self._session.post(
            self._url(f"/api/training-jobs/{job_id}/heartbeat"),
            headers=self._headers(),
            json=payload.model_dump(),
            timeout=self.config.request_timeout_seconds,
        )
        response.raise_for_status()
        return TrainingWorkerJobAckResponse.model_validate(response.json())

    def fetch_snapshot(self, job_id: str) -> bytes:
        response = self._session.get(
            self._url(f"/api/training-jobs/{job_id}/snapshot"),
            headers=self._headers(),
            params={"worker_id": self.config.worker_id},
            timeout=self.config.request_timeout_seconds,
        )
        response.raise_for_status()
        return response.content

    def upload_artifact(self, job_id: str, model_path: Path) -> TrainingArtifactUploadResponse:
        response = self._session.post(
            self._url(f"/api/training-jobs/{job_id}/artifact"),
            headers={
                **self._headers(),
                "Content-Type": "application/octet-stream",
            },
            params={
                "worker_id": self.config.worker_id,
                "filename": model_path.name,
            },
            data=model_path.read_bytes(),
            timeout=max(self.config.request_timeout_seconds, 600.0),
        )
        response.raise_for_status()
        return TrainingArtifactUploadResponse.model_validate(response.json())

    def complete_job(self, job_id: str, *, artifact_path: str, result: TrainingExecutionResult) -> dict[str, Any]:
        payload = TrainingWorkerCompleteRequest(
            worker_id=self.config.worker_id,
            artifact_path=artifact_path,
            metrics=result.metrics,
            metadata=result.metadata,
        )
        response = self._session.post(
            self._url(f"/api/training-jobs/{job_id}/complete"),
            headers=self._headers(),
            json=payload.model_dump(),
            timeout=self.config.request_timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def fail_job(self, job_id: str, *, error_message: str, retryable: bool = False) -> TrainingWorkerJobAckResponse:
        payload = TrainingWorkerFailRequest(
            worker_id=self.config.worker_id,
            error_message=error_message,
            retryable=retryable,
        )
        response = self._session.post(
            self._url(f"/api/training-jobs/{job_id}/fail"),
            headers=self._headers(),
            json=payload.model_dump(),
            timeout=self.config.request_timeout_seconds,
        )
        response.raise_for_status()
        return TrainingWorkerJobAckResponse.model_validate(response.json())


class _HeartbeatThread:
    def __init__(self, client: TrainingJobWorkerClient, job_id: str, interval_seconds: float) -> None:
        self.client = client
        self.job_id = job_id
        self.interval_seconds = max(1.0, interval_seconds)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"training-heartbeat-{job_id[:8]}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=max(1.0, self.interval_seconds))

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            try:
                self.client.heartbeat_job(self.job_id)
            except Exception:
                logger.exception("Failed to heartbeat training job %s", self.job_id)


def _run_training_sync(params: dict[str, Any]) -> TrainingExecutionResult:
    worker = TrainingWorker(params)
    result: dict[str, Any] = {}
    errors: list[str] = []

    def _on_finished(model_path: str, metrics: dict) -> None:
        result["model_path"] = model_path
        result["metrics"] = metrics

    def _on_error(message: str) -> None:
        errors.append(message)

    worker.finished.connect(_on_finished)
    worker.error.connect(_on_error)
    worker.run()

    if errors:
        raise RuntimeError(errors[-1])
    model_path = Path(str(result.get("model_path") or "")).resolve()
    if not model_path.is_file():
        raise RuntimeError("training did not produce a checkpoint")
    return TrainingExecutionResult(
        model_path=model_path,
        metrics=result.get("metrics") if isinstance(result.get("metrics"), dict) else {},
        metadata={
            "trainer": "TrainingWorker",
            "requested_device": str(params.get("device") or "auto"),
            "task_type": str(params.get("task_type") or "classification"),
            "backbone": str(params.get("backbone") or ""),
        },
    )


class TrainingJobWorkerRunner:
    def __init__(
        self,
        config: RemoteTrainingWorkerConfig,
        *,
        client: TrainingJobWorkerClient | None = None,
        trainer: Callable[[dict[str, Any]], TrainingExecutionResult] | None = None,
    ) -> None:
        self.config = config
        self.client = client or TrainingJobWorkerClient(config)
        self.trainer = trainer or _run_training_sync

    def _snapshot_path_for_job(self, job: TrainingWorkerClaimResponse) -> Path:
        path = self.config.execution.output_root / "snapshots" / f"{job.snapshot_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _build_training_params(self, job: TrainingWorkerClaimResponse, snapshot_path: Path, output_path: Path) -> dict[str, Any]:
        params = dict(job.train_config or {})
        params["dataset_dir"] = str(self.config.execution.dataset_root)
        params["dataset_format"] = "v2"
        params["task_type"] = "classification"
        params["backbone"] = job.model_backbone
        params["device"] = params.get("device") or self.config.execution.device
        params["annotations_document_path"] = str(snapshot_path)
        params["save_path"] = str(output_path)
        return params

    def run_once(self) -> bool:
        job = self.client.claim_job()
        if job is None:
            return False

        heartbeat = _HeartbeatThread(self.client, job.job_id, self.config.heartbeat_interval_seconds)
        snapshot_path = self._snapshot_path_for_job(job)
        output_path = self.config.execution.output_root / "checkpoints" / f"{job.model_id}.pth"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            snapshot_path.write_bytes(self.client.fetch_snapshot(job.job_id))
            training_params = self._build_training_params(job, snapshot_path, output_path)
            heartbeat.start()
            result = self.trainer(training_params)
            artifact = self.client.upload_artifact(job.job_id, result.model_path)
            self.client.complete_job(job.job_id, artifact_path=artifact.artifact_path, result=result)
            logger.info("Completed training job %s with model %s", job.job_id, job.model_id)
        except Exception as exc:
            logger.exception("Training job %s failed", job.job_id)
            self.client.fail_job(job.job_id, error_message=str(exc), retryable=False)
        finally:
            heartbeat.stop()
        return True

    def run_forever(self, *, max_jobs: int | None = None) -> int:
        processed = 0
        while True:
            handled = self.run_once()
            if handled:
                processed += 1
                if max_jobs is not None and processed >= max_jobs:
                    break
                continue
            if max_jobs is not None and processed >= max_jobs:
                break
            time.sleep(max(0.5, self.config.idle_poll_seconds))
        return processed


def _env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_list(name: str, default: list[str] | None = None) -> list[str]:
    raw = _env_str(name)
    if not raw:
        return list(default or [])
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_remote_training_worker_config_from_env() -> RemoteTrainingWorkerConfig:
    server_url = _env_str("SCANN_TRAINING_SERVER_URL")
    worker_token = _env_str("SCANN_TRAINING_WORKER_TOKEN")
    dataset_root_raw = _env_str("SCANN_TRAINING_WORKER_DATASET_ROOT")
    output_root_raw = _env_str("SCANN_TRAINING_WORKER_OUTPUT_ROOT")
    if not server_url:
        raise ValueError("SCANN_TRAINING_SERVER_URL is required")
    if not worker_token:
        raise ValueError("SCANN_TRAINING_WORKER_TOKEN is required")
    if not dataset_root_raw:
        raise ValueError("SCANN_TRAINING_WORKER_DATASET_ROOT is required")
    if not output_root_raw:
        raise ValueError("SCANN_TRAINING_WORKER_OUTPUT_ROOT is required")

    host_name = _env_str("SCANN_TRAINING_WORKER_HOST_NAME") or socket.gethostname()
    worker_id = _env_str("SCANN_TRAINING_WORKER_ID") or f"{host_name}-trainer"
    display_name = _env_str("SCANN_TRAINING_WORKER_NAME") or worker_id

    execution = TrainingExecutionConfig(
        dataset_root=Path(dataset_root_raw).resolve(),
        output_root=Path(output_root_raw).resolve(),
        task_types=_env_list("SCANN_TRAINING_WORKER_TASK_TYPES", ["classification"]),
        model_backbones=_env_list("SCANN_TRAINING_WORKER_MODEL_BACKBONES", []),
        device=_env_str("SCANN_TRAINING_WORKER_DEVICE", "auto") or "auto",
    )
    execution.output_root.mkdir(parents=True, exist_ok=True)

    return RemoteTrainingWorkerConfig(
        server_url=server_url.rstrip("/"),
        worker_token=worker_token,
        worker_id=worker_id,
        display_name=display_name,
        host_name=host_name,
        device_label=_env_str("SCANN_TRAINING_WORKER_DEVICE_LABEL") or None,
        execution=execution,
        idle_poll_seconds=float(_env_str("SCANN_TRAINING_WORKER_IDLE_SECONDS", "10") or 10),
        heartbeat_interval_seconds=float(_env_str("SCANN_TRAINING_WORKER_HEARTBEAT_SECONDS", "60") or 60),
        request_timeout_seconds=float(_env_str("SCANN_TRAINING_WORKER_REQUEST_TIMEOUT_SECONDS", "120") or 120),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the SCANN training GPU worker")
    parser.add_argument("--max-jobs", type=int, default=None, help="Process at most N jobs before exiting")
    parser.add_argument("--log-level", default=_env_str("SCANN_TRAINING_WORKER_LOG_LEVEL", "INFO") or "INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    config = load_remote_training_worker_config_from_env()
    runner = TrainingJobWorkerRunner(config)
    processed = runner.run_forever(max_jobs=args.max_jobs)
    logger.info("Training worker processed %s jobs", processed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
