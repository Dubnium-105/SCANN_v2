"""Persistence facade for discovery evaluation, review, AL, and rollout state."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from scann.core.dataset_storage import DatasetStorage


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(value: Any, default: Any) -> Any:
    try:
        parsed = json.loads(value or "")
    except (TypeError, ValueError):
        return default
    return parsed


class DiscoveryStorage:
    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self.dataset_storage = DatasetStorage(self.dataset_root)
        self.dataset_storage.ensure_schema()
        self.db_path = self.dataset_storage.db_path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @staticmethod
    def _evaluation(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "run_id": str(row["run_id"]),
            "run_type": str(row["run_type"]),
            "status": str(row["status"]),
            "partition_id": row["partition_id"],
            "model_id": row["model_id"],
            "artifact_relpath": row["artifact_relpath"],
            "artifact_sha256": row["artifact_sha256"],
            "config": _json(row["config_json"], {}),
            "metrics": _json(row["metrics_json"], {}),
            "error_message": row["error_message"],
            "created_by": row["created_by"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _review_event(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "event_id": str(row["event_id"]),
            "task_id": str(row["task_id"]),
            "prelabel_id": row["prelabel_id"],
            "revision_id": row["revision_id"],
            "model_id": row["model_id"],
            "outcome": str(row["outcome"]),
            "match_algorithm_version": str(
                row["match_algorithm_version"]
            ),
            "result": _json(row["result_json"], {}),
            "created_by": row["created_by"],
            "created_at": row["created_at"],
        }

    def create_evaluation(
        self,
        *,
        run_id: str,
        run_type: str,
        status: str,
        partition_id: str | None,
        model_id: str | None,
        config: Mapping[str, Any],
        created_by: str,
    ) -> dict[str, Any]:
        now = _utc_now_iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO evaluation_runs (
                    run_id, run_type, status, partition_id, model_id,
                    config_json, metrics_json, created_by,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, '{}', ?, ?, ?)
                """,
                (
                    run_id,
                    run_type,
                    status,
                    partition_id,
                    model_id,
                    json.dumps(dict(config), ensure_ascii=False),
                    created_by,
                    now,
                    now,
                ),
            )
        result = self.get_evaluation(run_id)
        if result is None:
            raise RuntimeError("failed to create evaluation run")
        return result

    def update_evaluation(
        self,
        run_id: str,
        *,
        status: str,
        artifact_relpath: str | None = None,
        artifact_sha256: str | None = None,
        metrics: Mapping[str, Any] | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE evaluation_runs
                SET status = ?,
                    artifact_relpath = ?,
                    artifact_sha256 = ?,
                    metrics_json = ?,
                    error_message = ?,
                    updated_at = ?
                WHERE run_id = ?
                """,
                (
                    status,
                    artifact_relpath,
                    artifact_sha256,
                    json.dumps(dict(metrics or {}), ensure_ascii=False),
                    error_message,
                    _utc_now_iso(),
                    run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("evaluation run not found")
        result = self.get_evaluation(run_id)
        if result is None:
            raise RuntimeError("failed to update evaluation run")
        return result

    def get_evaluation(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM evaluation_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return self._evaluation(row) if row is not None else None

    def list_evaluations(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM evaluation_runs
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?
                """,
                (max(1, min(int(limit), 500)),),
            ).fetchall()
        return [self._evaluation(row) for row in rows]

    def create_review_event(
        self,
        *,
        event_id: str,
        task_id: str,
        prelabel_id: str | None,
        revision_id: str | None,
        model_id: str | None,
        outcome: str,
        match_algorithm_version: str,
        result: Mapping[str, Any],
        created_by: str,
    ) -> dict[str, Any]:
        now = _utc_now_iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO annotation_review_events (
                    event_id, task_id, prelabel_id, revision_id, model_id,
                    outcome, match_algorithm_version, result_json,
                    created_by, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    task_id,
                    prelabel_id,
                    revision_id,
                    model_id,
                    outcome,
                    match_algorithm_version,
                    json.dumps(dict(result), ensure_ascii=False),
                    created_by,
                    now,
                ),
            )
        stored = self.get_review_event(event_id)
        if stored is None:
            raise RuntimeError("failed to create review event")
        return stored

    def get_review_event(
        self,
        event_id: str,
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM annotation_review_events
                WHERE event_id = ?
                """,
                (event_id,),
            ).fetchone()
        return self._review_event(row) if row is not None else None

    def list_review_events(
        self,
        *,
        model_id: str | None = None,
        task_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM annotation_review_events"
        parameters: list[Any] = []
        filters: list[str] = []
        if model_id is not None:
            filters.append("model_id = ?")
            parameters.append(model_id)
        if task_id is not None:
            filters.append("task_id = ?")
            parameters.append(task_id)
        if filters:
            query += " WHERE " + " AND ".join(filters)
        query += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        parameters.append(max(1, min(int(limit), 500)))
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._review_event(row) for row in rows]

    def create_active_learning_batch(
        self,
        *,
        batch_id: str,
        batch_name: str,
        status: str,
        strategy_version: str,
        model_id: str | None,
        partition_id: str | None,
        budget: int,
        config: Mapping[str, Any],
        summary: Mapping[str, Any],
        items: Iterable[Mapping[str, Any]],
        created_by: str,
    ) -> dict[str, Any]:
        now = _utc_now_iso()
        normalized_items = [dict(item) for item in items]
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO active_learning_batches (
                    batch_id, batch_name, status, strategy_version,
                    model_id, partition_id, budget,
                    config_json, summary_json, created_by,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    batch_id,
                    batch_name,
                    status,
                    strategy_version,
                    model_id,
                    partition_id,
                    int(budget),
                    json.dumps(dict(config), ensure_ascii=False),
                    json.dumps(dict(summary), ensure_ascii=False),
                    created_by,
                    now,
                    now,
                ),
            )
            connection.executemany(
                """
                INSERT INTO active_learning_items (
                    batch_id, task_id, rank, score, group_key,
                    reasons_json, dual_review, review_status,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                [
                    (
                        batch_id,
                        str(item["task_id"]),
                        int(item["rank"]),
                        float(item["score"]),
                        str(item.get("group_key") or ""),
                        json.dumps(
                            list(item.get("reasons") or []),
                            ensure_ascii=False,
                        ),
                        1 if item.get("dual_review") else 0,
                        json.dumps(item, ensure_ascii=False),
                    )
                    for item in normalized_items
                ],
            )
        result = self.get_active_learning_batch(batch_id)
        if result is None:
            raise RuntimeError("failed to create active-learning batch")
        return result

    def get_active_learning_batch(
        self,
        batch_id: str,
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            batch = connection.execute(
                """
                SELECT * FROM active_learning_batches
                WHERE batch_id = ?
                """,
                (batch_id,),
            ).fetchone()
            if batch is None:
                return None
            rows = connection.execute(
                """
                SELECT * FROM active_learning_items
                WHERE batch_id = ?
                ORDER BY rank
                """,
                (batch_id,),
            ).fetchall()
        return {
            "batch_id": str(batch["batch_id"]),
            "batch_name": str(batch["batch_name"]),
            "status": str(batch["status"]),
            "strategy_version": str(batch["strategy_version"]),
            "model_id": batch["model_id"],
            "partition_id": batch["partition_id"],
            "budget": int(batch["budget"]),
            "config": _json(batch["config_json"], {}),
            "summary": _json(batch["summary_json"], {}),
            "created_by": batch["created_by"],
            "created_at": batch["created_at"],
            "updated_at": batch["updated_at"],
            "items": [
                {
                    "task_id": str(row["task_id"]),
                    "rank": int(row["rank"]),
                    "score": float(row["score"]),
                    "group_key": row["group_key"],
                    "reasons": _json(row["reasons_json"], []),
                    "dual_review": bool(row["dual_review"]),
                    "review_status": str(row["review_status"]),
                    "reviewed_at": row["reviewed_at"],
                    "metadata": _json(row["metadata_json"], {}),
                }
                for row in rows
            ],
        }

    def list_active_learning_batches(
        self,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT batch_id FROM active_learning_batches
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?
                """,
                (max(1, min(int(limit), 500)),),
            ).fetchall()
        return [
            result
            for row in rows
            if (
                result := self.get_active_learning_batch(
                    str(row["batch_id"])
                )
            )
            is not None
        ]

    @staticmethod
    def _deployment(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "deployment_id": str(row["deployment_id"]),
            "model_id": str(row["model_id"]),
            "stage": str(row["stage"]),
            "status": str(row["status"]),
            "traffic_fraction": float(row["traffic_fraction"]),
            "previous_deployment_id": row["previous_deployment_id"],
            "config": _json(row["config_json"], {}),
            "metrics": _json(row["metrics_json"], {}),
            "created_by": row["created_by"],
            "created_at": row["created_at"],
            "ended_at": row["ended_at"],
        }

    def create_deployment(
        self,
        *,
        deployment_id: str,
        model_id: str,
        stage: str,
        status: str,
        traffic_fraction: float,
        previous_deployment_id: str | None,
        config: Mapping[str, Any],
        metrics: Mapping[str, Any],
        created_by: str,
        end_previous: bool = True,
    ) -> dict[str, Any]:
        now = _utc_now_iso()
        with self._connect() as connection:
            if end_previous and previous_deployment_id:
                connection.execute(
                    """
                    UPDATE model_deployments
                    SET status = 'ended', ended_at = ?
                    WHERE deployment_id = ? AND status = 'active'
                    """,
                    (now, previous_deployment_id),
                )
            connection.execute(
                """
                INSERT INTO model_deployments (
                    deployment_id, model_id, stage, status,
                    traffic_fraction, previous_deployment_id,
                    config_json, metrics_json, created_by, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    deployment_id,
                    model_id,
                    stage,
                    status,
                    float(traffic_fraction),
                    previous_deployment_id,
                    json.dumps(dict(config), ensure_ascii=False),
                    json.dumps(dict(metrics), ensure_ascii=False),
                    created_by,
                    now,
                ),
            )
        result = self.get_deployment(deployment_id)
        if result is None:
            raise RuntimeError("failed to create model deployment")
        return result

    def get_deployment(
        self,
        deployment_id: str,
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM model_deployments
                WHERE deployment_id = ?
                """,
                (deployment_id,),
            ).fetchone()
        return self._deployment(row) if row is not None else None

    def latest_deployment_for_model(
        self,
        model_id: str,
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM model_deployments
                WHERE model_id = ?
                ORDER BY created_at DESC, rowid DESC
                LIMIT 1
                """,
                (model_id,),
            ).fetchone()
        return self._deployment(row) if row is not None else None

    def list_deployments(
        self,
        *,
        model_id: str | None = None,
        stage: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM model_deployments"
        parameters: list[Any] = []
        filters: list[str] = []
        if model_id is not None:
            filters.append("model_id = ?")
            parameters.append(model_id)
        if stage is not None:
            filters.append("stage = ?")
            parameters.append(stage)
        if status is not None:
            filters.append("status = ?")
            parameters.append(status)
        if filters:
            query += " WHERE " + " AND ".join(filters)
        query += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        parameters.append(max(1, min(int(limit), 500)))
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._deployment(row) for row in rows]

    def end_deployment(self, deployment_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE model_deployments
                SET status = 'ended', ended_at = ?
                WHERE deployment_id = ? AND status = 'active'
                """,
                (_utc_now_iso(), deployment_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("active deployment not found")
        result = self.get_deployment(deployment_id)
        if result is None:
            raise RuntimeError("failed to end deployment")
        return result
