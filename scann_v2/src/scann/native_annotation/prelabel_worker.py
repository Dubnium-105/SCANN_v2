from __future__ import annotations

import argparse
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import numpy as np
import requests
from astropy.io import fits as astropy_fits

from scann.ai.inference import InferenceConfig, InferenceEngine
from scann.core.candidate_detector import DetectionParams
from scann.core.config import load_config
from scann.core.fits_io import read_fits
from scann.core.models import FitsHeader, FitsImage
from scann.services.detection_pipeline import DetectionPipeline

from .prelabel_service import (
    PrelabelBox,
    TaskPrelabelResponse,
    WorkerClaimRequest,
    WorkerClaimResponse,
    WorkerCompleteRequest,
    WorkerFailRequest,
    WorkerHeartbeatRequest,
    WorkerJobAckResponse,
)


logger = logging.getLogger(__name__)


_CAPABILITY_WILDCARDS = {"", "auto", "any", "*"}


@dataclass(frozen=True)
class WorkerDetectionConfig:
    model_path: str
    model_version: str
    model_id: str | None = None
    model_format: str = "auto"
    model_backbone: str = "auto"
    compute_device: str = "auto"
    batch_size: int = 64
    patch_size: int = 80
    detection_mode: str = "patch"
    hybrid_primary_mode: str = "full_image"
    hybrid_low_confidence: float = 0.5
    detection_params: DetectionParams = field(default_factory=DetectionParams)


@dataclass(frozen=True)
class PrelabelWorkerConfig:
    server_url: str
    worker_token: str
    worker_id: str
    display_name: str
    host_name: str
    device_label: str | None
    dataset_root: Path | None
    detection: WorkerDetectionConfig
    idle_poll_seconds: float = 5.0
    heartbeat_interval_seconds: float = 30.0
    request_timeout_seconds: float = 60.0

    @property
    def supported_model_versions(self) -> list[str]:
        return [self.detection.model_version]

    @property
    def supported_model_ids(self) -> list[str]:
        model_id = str(self.detection.model_id or "").strip()
        return [model_id] if model_id else []

    @property
    def supported_model_backbones(self) -> list[str]:
        model_backbone = str(self.detection.model_backbone or "").strip()
        if model_backbone.lower() in _CAPABILITY_WILDCARDS:
            return []
        return [model_backbone]


@dataclass(frozen=True)
class PrelabelProcessingResult:
    source_view: str
    ai_suggestion: str | None
    ai_confidence: float | None
    annotations: list[PrelabelBox]
    metadata: dict[str, Any] = field(default_factory=dict)


class PrelabelWorkerClient:
    def __init__(self, config: PrelabelWorkerConfig, session: requests.Session | None = None) -> None:
        self.config = config
        self._session = session or requests.Session()
        self.last_claim_detail = ""

    def _url(self, path: str) -> str:
        return f"{self.config.server_url.rstrip('/')}{path}"

    def _headers(self) -> dict[str, str]:
        return {"X-SCANN-Worker-Token": self.config.worker_token}

    def claim_job(self) -> WorkerClaimResponse | None:
        self.last_claim_detail = ""
        payload = WorkerClaimRequest(
            worker_id=self.config.worker_id,
            display_name=self.config.display_name,
            host_name=self.config.host_name,
            device_label=self.config.device_label,
            capabilities={
                "model_versions": self.config.supported_model_versions,
                "model_ids": self.config.supported_model_ids,
                "model_backbones": self.config.supported_model_backbones,
            },
        )
        response = self._session.post(
            self._url("/api/prelabel-jobs/claim"),
            headers=self._headers(),
            json=payload.model_dump(),
            timeout=self.config.request_timeout_seconds,
        )
        if response.status_code == 404:
            detail = ""
            try:
                body = response.json()
                if isinstance(body, dict):
                    detail = str(body.get("detail") or "").strip()
            except Exception:
                detail = ""
            self.last_claim_detail = detail or "No queued prelabel job"
            return None
        response.raise_for_status()
        return WorkerClaimResponse.model_validate(response.json())

    def heartbeat_job(self, job_id: str) -> WorkerJobAckResponse:
        payload = WorkerHeartbeatRequest(worker_id=self.config.worker_id)
        response = self._session.post(
            self._url(f"/api/prelabel-jobs/{job_id}/heartbeat"),
            headers=self._headers(),
            json=payload.model_dump(),
            timeout=self.config.request_timeout_seconds,
        )
        response.raise_for_status()
        return WorkerJobAckResponse.model_validate(response.json())

    def complete_job(self, job_id: str, result: PrelabelProcessingResult) -> TaskPrelabelResponse:
        payload = WorkerCompleteRequest(
            worker_id=self.config.worker_id,
            source_view=result.source_view,
            ai_suggestion=result.ai_suggestion,
            ai_confidence=result.ai_confidence,
            annotations=result.annotations,
            metadata=result.metadata,
        )
        response = self._session.post(
            self._url(f"/api/prelabel-jobs/{job_id}/complete"),
            headers=self._headers(),
            json=payload.model_dump(exclude_none=True),
            timeout=self.config.request_timeout_seconds,
        )
        response.raise_for_status()
        return TaskPrelabelResponse.model_validate(response.json())

    def fail_job(self, job_id: str, *, error_message: str, retryable: bool = False) -> WorkerJobAckResponse:
        payload = WorkerFailRequest(
            worker_id=self.config.worker_id,
            error_message=error_message,
            retryable=retryable,
        )
        response = self._session.post(
            self._url(f"/api/prelabel-jobs/{job_id}/fail"),
            headers=self._headers(),
            json=payload.model_dump(),
            timeout=self.config.request_timeout_seconds,
        )
        response.raise_for_status()
        return WorkerJobAckResponse.model_validate(response.json())

    def fetch_job_fits(self, job_id: str, view_name: str) -> bytes:
        response = self._session.get(
            self._url(f"/api/prelabel-jobs/{job_id}/fits/{view_name}"),
            headers=self._headers(),
            params={"worker_id": self.config.worker_id},
            timeout=self.config.request_timeout_seconds,
        )
        response.raise_for_status()
        return response.content


class PrelabelTaskAssetResolver:
    def __init__(self, config: PrelabelWorkerConfig, client: PrelabelWorkerClient, job: WorkerClaimResponse) -> None:
        self.config = config
        self.client = client
        self.job = job

    def _local_path(self, view_name: str) -> Path | None:
        if self.config.dataset_root is None:
            return None
        relpath = self.job.paths.get(view_name)
        if not relpath:
            return None
        candidate = (self.config.dataset_root / relpath).resolve()
        try:
            candidate.relative_to(self.config.dataset_root.resolve())
        except ValueError:
            return None
        return candidate if candidate.is_file() else None

    @staticmethod
    def _read_fits_from_bytes(data: bytes) -> FitsImage:
        with astropy_fits.open(BytesIO(data), memmap=False) as hdul:
            image_data = None
            header_dict: dict[str, Any] = {}
            for hdu in hdul:
                if hdu.data is not None:
                    image_data = np.array(hdu.data)
                    header_dict = dict(hdu.header)
                    break
        if image_data is None:
            raise ValueError("FITS data is empty")
        if image_data.ndim > 2:
            image_data = np.squeeze(image_data)
        return FitsImage(data=image_data, header=FitsHeader(raw=header_dict))

    def load_fits(self, view_name: str) -> FitsImage | None:
        relpath = self.job.paths.get(view_name)
        if not relpath:
            return None
        local_path = self._local_path(view_name)
        if local_path is not None:
            logger.info(
                "Prelabel job %s loading %s FITS from local path %s",
                self.job.job_id,
                view_name,
                local_path,
            )
            return read_fits(local_path)
        logger.info(
            "Prelabel job %s fetching %s FITS from server path %s",
            self.job.job_id,
            view_name,
            relpath,
        )
        return self._read_fits_from_bytes(self.client.fetch_job_fits(self.job.job_id, view_name))


class DetectionPrelabelProcessor:
    def __init__(self, config: PrelabelWorkerConfig) -> None:
        self.config = config
        inference_config = InferenceConfig(
            batch_size=config.detection.batch_size,
            device=config.detection.compute_device,
            model_format=config.detection.model_format,
            model_backbone=config.detection.model_backbone,
        )
        self.inference_engine = InferenceEngine(
            model_path=config.detection.model_path,
            config=inference_config,
        )
        self.pipeline = DetectionPipeline(
            detection_params=config.detection.detection_params,
            inference_engine=self.inference_engine,
            patch_size=config.detection.patch_size,
            detection_mode=config.detection.detection_mode,
            hybrid_primary_mode=config.detection.hybrid_primary_mode,
            hybrid_low_confidence=config.detection.hybrid_low_confidence,
        )

    @staticmethod
    def _clip_box(
        *,
        left: int,
        top: int,
        box_width: int,
        box_height: int,
        image_shape: tuple[int, ...],
    ) -> tuple[int, int, int, int]:
        height, width = image_shape[:2]
        safe_left = max(0, min(int(left), max(0, width - 1)))
        safe_top = max(0, min(int(top), max(0, height - 1)))
        safe_width = max(1, min(int(box_width), max(1, width - safe_left)))
        safe_height = max(1, min(int(box_height), max(1, height - safe_top)))
        return safe_left, safe_top, safe_width, safe_height

    def _candidate_to_prelabel_box(self, candidate, image_shape: tuple[int, ...]) -> PrelabelBox:
        bbox_width = getattr(candidate, "bbox_width", None)
        bbox_height = getattr(candidate, "bbox_height", None)
        if bbox_width is not None and bbox_height is not None:
            width_value = max(1, int(round(float(bbox_width))))
            height_value = max(1, int(round(float(bbox_height))))
            bbox_left = getattr(candidate, "bbox_x", None)
            bbox_top = getattr(candidate, "bbox_y", None)
            if bbox_left is None:
                bbox_left = int(round(float(candidate.x) - width_value / 2.0))
            if bbox_top is None:
                bbox_top = int(round(float(candidate.y) - height_value / 2.0))
            left, top, box_width, box_height = self._clip_box(
                left=int(bbox_left),
                top=int(bbox_top),
                box_width=width_value,
                box_height=height_value,
                image_shape=image_shape,
            )
            return PrelabelBox(
                x=float(left),
                y=float(top),
                width=float(box_width),
                height=float(box_height),
                label=None,
                detail_type=None,
                confidence=float(getattr(candidate, "ai_score", 0.0)),
            )

        patch_size = int(self.config.detection.patch_size)
        half_size = patch_size // 2

        left, top, box_width, box_height = self._clip_box(
            left=int(candidate.x) - half_size,
            top=int(candidate.y) - half_size,
            box_width=patch_size,
            box_height=patch_size,
            image_shape=image_shape,
        )
        return PrelabelBox(
            x=float(left),
            y=float(top),
            width=float(max(1, box_width)),
            height=float(max(1, box_height)),
            label=None,
            detail_type=None,
            confidence=float(getattr(candidate, "ai_score", 0.0)),
        )

    def process(self, job: WorkerClaimResponse, assets: PrelabelTaskAssetResolver) -> PrelabelProcessingResult:
        new_image = assets.load_fits("new")
        if new_image is None:
            raise ValueError("new FITS asset is missing")
        old_image = assets.load_fits("old")
        old_data = np.asarray(old_image.data if old_image is not None else np.zeros_like(new_image.data), dtype=np.float32)
        new_data = np.asarray(new_image.data, dtype=np.float32)

        requested_threshold = (
            float(job.confidence_threshold) if job.confidence_threshold is not None else None
        )
        requested_limit = int(job.candidate_limit) if job.candidate_limit is not None else None
        logger.info(
            "Processing prelabel job %s for task %s with candidate_limit=%s confidence_threshold=%s",
            job.job_id,
            job.task_id,
            requested_limit if requested_limit is not None else "-",
            f"{requested_threshold:.4f}" if requested_threshold is not None else "-",
        )
        original_threshold = getattr(self.inference_engine, "threshold", None)
        original_topk = getattr(self.pipeline.detection_params, "topk", None)

        if requested_threshold is not None and self.inference_engine is not None:
            self.inference_engine.threshold = requested_threshold
        if requested_limit is not None and original_topk is not None:
            self.pipeline.detection_params.topk = max(1, requested_limit)

        try:
            result = self.pipeline.process_pair(
                pair_name=job.task_id,
                new_data=new_data,
                old_data=old_data,
                image_path=job.paths.get("new"),
            )
        finally:
            if original_threshold is not None and self.inference_engine is not None:
                self.inference_engine.threshold = original_threshold
            if original_topk is not None:
                self.pipeline.detection_params.topk = original_topk

        candidates = list(getattr(result, "candidates", []) or [])
        if requested_threshold is not None:
            candidates = [
                candidate
                for candidate in candidates
                if float(getattr(candidate, "ai_score", 0.0)) >= requested_threshold
            ]
        if requested_limit is not None:
            candidates = candidates[: max(1, requested_limit)]

        annotations = [
            self._candidate_to_prelabel_box(candidate, new_data.shape)
            for candidate in candidates
        ]
        ai_confidence = max((item.confidence or 0.0 for item in annotations), default=None)
        return PrelabelProcessingResult(
            source_view="new",
            ai_suggestion="real" if annotations else None,
            ai_confidence=ai_confidence,
            annotations=annotations,
            metadata={
                "processor": "detection_pipeline",
                "model_version": self.config.detection.model_version,
                "model_id": self.config.detection.model_id,
                "model_backbone": self.config.detection.model_backbone,
                "detection_mode": self.config.detection.detection_mode,
                "patch_size": self.config.detection.patch_size,
                "candidate_limit": requested_limit,
                "confidence_threshold": requested_threshold,
                "candidate_count": len(annotations),
                "raw_candidate_count": len(getattr(result, "candidates", []) or []),
                "pipeline_error": getattr(result, "error", ""),
            },
        )


class _HeartbeatThread:
    def __init__(self, client: PrelabelWorkerClient, job_id: str, interval_seconds: float) -> None:
        self.client = client
        self.job_id = job_id
        self.interval_seconds = max(1.0, interval_seconds)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"prelabel-heartbeat-{job_id[:8]}", daemon=True)

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
                logger.exception("Failed to heartbeat prelabel job %s", self.job_id)


class PrelabelWorkerRunner:
    def __init__(
        self,
        config: PrelabelWorkerConfig,
        *,
        client: PrelabelWorkerClient | None = None,
        processor: DetectionPrelabelProcessor | Any | None = None,
    ) -> None:
        self.config = config
        self.client = client or PrelabelWorkerClient(config)
        self.processor = processor or DetectionPrelabelProcessor(config)
        self._idle_miss_count = 0
        self._last_idle_log_monotonic = 0.0
        self._last_idle_reason = ""

    def _process_job(self, job: WorkerClaimResponse) -> None:
        heartbeat = _HeartbeatThread(
            client=self.client,
            job_id=job.job_id,
            interval_seconds=self.config.heartbeat_interval_seconds,
        )
        heartbeat.start()
        try:
            assets = PrelabelTaskAssetResolver(self.config, self.client, job)
            result = self.processor.process(job, assets)
            self.client.complete_job(job.job_id, result)
        except Exception as exc:
            logger.exception("Prelabel job %s failed", job.job_id)
            message = str(exc).strip() or exc.__class__.__name__
            try:
                self.client.fail_job(job.job_id, error_message=message, retryable=False)
            except Exception:
                logger.exception("Failed to report error for prelabel job %s", job.job_id)
        finally:
            heartbeat.stop()

    def run_once(self) -> bool:
        job = self.client.claim_job()
        if job is None:
            self._idle_miss_count += 1
            detail = str(getattr(self.client, "last_claim_detail", "") or "No queued prelabel job").strip()
            now = time.monotonic()
            should_log = (
                self._idle_miss_count == 1
                or detail != self._last_idle_reason
                or (now - self._last_idle_log_monotonic) >= max(30.0, self.config.idle_poll_seconds * 4)
            )
            if should_log:
                logger.info(
                    "Prelabel worker idle (miss #%s): %s | server=%s worker_id=%s model_version=%s model_id=%s model_backbone=%s",
                    self._idle_miss_count,
                    detail,
                    self.config.server_url,
                    self.config.worker_id,
                    self.config.detection.model_version,
                    self.config.detection.model_id or "-",
                    self.config.detection.model_backbone or "-",
                )
                self._last_idle_reason = detail
                self._last_idle_log_monotonic = now
            return False
        self._idle_miss_count = 0
        self._last_idle_reason = ""
        logger.info("Claimed prelabel job %s for task %s", job.job_id, job.task_id)
        self._process_job(job)
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


def load_prelabel_worker_config_from_env() -> PrelabelWorkerConfig:
    server_url = _env_str("SCANN_PRELABEL_SERVER_URL")
    worker_token = _env_str("SCANN_PRELABEL_WORKER_TOKEN")
    if not server_url:
        raise ValueError("SCANN_PRELABEL_SERVER_URL is required")
    if not worker_token:
        raise ValueError("SCANN_PRELABEL_WORKER_TOKEN is required")

    host_name = _env_str("SCANN_PRELABEL_WORKER_HOST_NAME") or socket.gethostname()
    worker_id = _env_str("SCANN_PRELABEL_WORKER_ID") or f"{host_name}-worker"
    display_name = _env_str("SCANN_PRELABEL_WORKER_NAME") or worker_id
    dataset_root_raw = _env_str("SCANN_PRELABEL_WORKER_DATASET_ROOT")
    dataset_root = Path(dataset_root_raw).resolve() if dataset_root_raw else None

    config_path = _env_str("SCANN_PRELABEL_WORKER_CONFIG_PATH")
    app_config = load_config(config_path or None)
    model_path = _env_str("SCANN_PRELABEL_WORKER_MODEL_PATH") or str(getattr(app_config, "model_path", "") or "")
    if not model_path:
        raise ValueError("SCANN_PRELABEL_WORKER_MODEL_PATH or config.model_path is required")
    model_version = _env_str("SCANN_PRELABEL_WORKER_MODEL_VERSION") or Path(model_path).stem
    model_id = _env_str("SCANN_PRELABEL_WORKER_MODEL_ID") or Path(model_path).stem

    detection_params = DetectionParams(
        thresh=int(_env_str("SCANN_PRELABEL_THRESH", str(getattr(app_config, "thresh", 80))) or 80),
        min_area=int(_env_str("SCANN_PRELABEL_MIN_AREA", str(getattr(app_config, "min_area", 6))) or 6),
        max_area=int(_env_str("SCANN_PRELABEL_MAX_AREA", str(getattr(app_config, "max_area", 600))) or 600),
        sharpness_min=float(_env_str("SCANN_PRELABEL_SHARPNESS_MIN", str(getattr(app_config, "sharpness", 1.2))) or 1.2),
        sharpness_max=float(_env_str("SCANN_PRELABEL_SHARPNESS_MAX", str(getattr(app_config, "max_sharpness", 5.0))) or 5.0),
        contrast_min=int(_env_str("SCANN_PRELABEL_CONTRAST_MIN", str(getattr(app_config, "contrast", 15))) or 15),
        edge_margin=int(_env_str("SCANN_PRELABEL_EDGE_MARGIN", str(getattr(app_config, "edge_margin", 10))) or 10),
        dynamic_thresh=_env_str("SCANN_PRELABEL_DYNAMIC_THRESH", str(getattr(app_config, "dynamic_thresh", False))).lower() in {"1", "true", "yes", "on"},
        kill_flat=_env_str("SCANN_PRELABEL_KILL_FLAT", str(getattr(app_config, "kill_flat", True))).lower() in {"1", "true", "yes", "on"},
        kill_dipole=_env_str("SCANN_PRELABEL_KILL_DIPOLE", str(getattr(app_config, "kill_dipole", True))).lower() in {"1", "true", "yes", "on"},
        aspect_ratio_max=float(_env_str("SCANN_PRELABEL_ASPECT_RATIO_MAX", str(getattr(app_config, "aspect_ratio_max", 3.0))) or 3.0),
        extent_max=float(_env_str("SCANN_PRELABEL_EXTENT_MAX", str(getattr(app_config, "extent_max", 0.9))) or 0.9),
        topk=int(_env_str("SCANN_PRELABEL_TOPK", str(getattr(app_config, "topk", 20))) or 20),
    )
    detection = WorkerDetectionConfig(
        model_path=model_path,
        model_version=model_version,
        model_id=model_id,
        model_format=_env_str("SCANN_PRELABEL_WORKER_MODEL_FORMAT") or str(getattr(app_config, "model_format", "auto")),
        model_backbone=_env_str("SCANN_PRELABEL_WORKER_MODEL_BACKBONE") or str(getattr(app_config, "model_backbone", "auto")),
        compute_device=_env_str("SCANN_PRELABEL_WORKER_COMPUTE_DEVICE") or str(getattr(app_config, "compute_device", "auto")),
        batch_size=int(_env_str("SCANN_PRELABEL_WORKER_BATCH_SIZE", str(getattr(app_config, "batch_size", 64))) or 64),
        patch_size=int(_env_str("SCANN_PRELABEL_WORKER_PATCH_SIZE", str(getattr(app_config, "slice_size", 80))) or 80),
        detection_mode=_env_str("SCANN_PRELABEL_WORKER_DETECTION_MODE") or str(getattr(app_config, "detection_mode", "patch")),
        hybrid_primary_mode=_env_str("SCANN_PRELABEL_WORKER_HYBRID_PRIMARY_MODE") or str(
            getattr(app_config, "hybrid_primary_mode", "full_image")
        ),
        hybrid_low_confidence=float(
            _env_str(
                "SCANN_PRELABEL_WORKER_HYBRID_LOW_CONFIDENCE",
                str(getattr(app_config, "hybrid_low_confidence", 0.5)),
            )
            or 0.5
        ),
        detection_params=detection_params,
    )

    return PrelabelWorkerConfig(
        server_url=server_url.rstrip("/"),
        worker_token=worker_token,
        worker_id=worker_id,
        display_name=display_name,
        host_name=host_name,
        device_label=_env_str("SCANN_PRELABEL_WORKER_DEVICE_LABEL") or None,
        dataset_root=dataset_root,
        idle_poll_seconds=float(_env_str("SCANN_PRELABEL_WORKER_IDLE_SECONDS", "5") or 5),
        heartbeat_interval_seconds=float(_env_str("SCANN_PRELABEL_WORKER_HEARTBEAT_SECONDS", "30") or 30),
        request_timeout_seconds=float(_env_str("SCANN_PRELABEL_WORKER_REQUEST_TIMEOUT_SECONDS", "60") or 60),
        detection=detection,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the SCANN prelabel GPU worker")
    parser.add_argument("--max-jobs", type=int, default=None, help="Process at most N jobs before exiting")
    parser.add_argument("--log-level", default=_env_str("SCANN_PRELABEL_WORKER_LOG_LEVEL", "INFO") or "INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = load_prelabel_worker_config_from_env()
    logger.info(
        "Starting prelabel worker %s (%s) server=%s dataset_root=%s device=%s model_version=%s model_id=%s model_backbone=%s compute_device=%s",
        config.worker_id,
        config.display_name,
        config.server_url,
        str(config.dataset_root) if config.dataset_root is not None else "<remote-only>",
        config.device_label or "-",
        config.detection.model_version,
        config.detection.model_id or "-",
        config.detection.model_backbone or "-",
        config.detection.compute_device,
    )
    runner = PrelabelWorkerRunner(config)
    processed = runner.run_forever(max_jobs=args.max_jobs)
    logger.info("Prelabel worker stopped after processing %s jobs", processed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
