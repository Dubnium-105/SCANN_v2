"""Read-only integrity audit for a SCANN dataset database and its files."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from scann.ai.dataset_partition import verify_partition_manifest
from scann.core.annotation_models import AnnotationLabel, DETAIL_TYPE_TO_LABEL, DetailType
from scann.core.dataset_storage import DEFAULT_DATASET_DB_FILE


EXPECTED_TABLES: tuple[str, ...] = (
    "raw_assets",
    "tasks",
    "task_artifacts",
    "task_annotation_boxes_current",
    "annotation_revisions",
    "annotation_revision_boxes",
    "prelabel_jobs",
    "task_ai_prelabels",
    "task_ai_prelabel_boxes",
    "worker_nodes",
    "dataset_partitions",
    "evaluation_runs",
    "annotation_review_events",
    "active_learning_batches",
    "active_learning_items",
    "model_deployments",
    "dataset_snapshots",
    "training_jobs",
    "training_runs",
    "model_registry",
    "schema_migrations",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class AuditIssue:
    severity: str
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetAuditReport:
    dataset_root: str
    database_path: str
    started_at: str
    finished_at: str = ""
    status: str = "ok"
    integrity_check: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    files: dict[str, Any] = field(default_factory=dict)
    annotations: dict[str, Any] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)
    foreign_key_violation_count: int = 0
    issues: list[AuditIssue] = field(default_factory=list)

    def finish(self) -> None:
        self.finished_at = _utc_now_iso()
        severities = {issue.severity for issue in self.issues}
        if "error" in severities:
            self.status = "error"
        elif "warning" in severities:
            self.status = "warning"
        else:
            self.status = "ok"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatasetAuditor:
    def __init__(
        self,
        dataset_root: Path,
        *,
        db_path: Path | None = None,
        max_issue_details: int = 50,
        verify_model_hashes: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self.db_path = (
            Path(db_path).resolve()
            if db_path is not None
            else (self.dataset_root / DEFAULT_DATASET_DB_FILE).resolve()
        )
        self.max_issue_details = max(1, int(max_issue_details))
        self.verify_model_hashes = bool(verify_model_hashes)
        self.report = DatasetAuditReport(
            dataset_root=str(self.dataset_root),
            database_path=str(self.db_path),
            started_at=_utc_now_iso(),
        )

    def _issue(
        self,
        severity: str,
        code: str,
        message: str,
        **details: Any,
    ) -> None:
        self.report.issues.append(
            AuditIssue(
                severity=severity,
                code=code,
                message=message,
                details=details,
            )
        )

    @staticmethod
    def _table_names(connection: sqlite3.Connection) -> set[str]:
        return {
            str(row[0])
            for row in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                """
            ).fetchall()
        }

    def _resolve_dataset_file(self, relpath: str) -> Path | None:
        raw = str(relpath or "").strip()
        if not raw:
            return None
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.dataset_root / candidate
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.dataset_root)
        except ValueError:
            return None
        return resolved

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _audit_schema(self, connection: sqlite3.Connection, table_names: set[str]) -> None:
        missing_tables = sorted(set(EXPECTED_TABLES) - table_names)
        for table_name in missing_tables:
            self._issue(
                "error",
                "missing_table",
                f"required table is missing: {table_name}",
                table=table_name,
            )

        for table_name in EXPECTED_TABLES:
            if table_name not in table_names:
                continue
            count = connection.execute(
                f'SELECT COUNT(*) FROM "{table_name}"'
            ).fetchone()[0]
            self.report.counts[table_name] = int(count)

        integrity_rows = [
            str(row[0])
            for row in connection.execute("PRAGMA integrity_check").fetchall()
        ]
        self.report.integrity_check = integrity_rows
        if integrity_rows != ["ok"]:
            self._issue(
                "error",
                "sqlite_integrity_failed",
                "SQLite integrity_check did not return ok",
                results=integrity_rows[: self.max_issue_details],
            )

        foreign_key_rows = connection.execute("PRAGMA foreign_key_check").fetchall()
        self.report.foreign_key_violation_count = len(foreign_key_rows)
        if foreign_key_rows:
            self._issue(
                "error",
                "foreign_key_violation",
                f"found {len(foreign_key_rows)} foreign-key violations",
                rows=[
                    list(row)
                    for row in foreign_key_rows[: self.max_issue_details]
                ],
            )

    def _audit_file_rows(
        self,
        connection: sqlite3.Connection,
        *,
        table_name: str,
        id_sql: str,
        path_column: str,
        where_sql: str = "",
        severity: str,
        result_key: str,
    ) -> None:
        if table_name not in self.report.counts:
            return
        rows = connection.execute(
            f"""
            SELECT {id_sql} AS audit_id, "{path_column}"
            FROM "{table_name}"
            {where_sql}
            """
        ).fetchall()
        missing: list[dict[str, str]] = []
        outside: list[dict[str, str]] = []
        present = 0
        for row in rows:
            record_id = str(row[0])
            relpath = str(row[1] or "")
            resolved = self._resolve_dataset_file(relpath)
            if resolved is None:
                outside.append({"id": record_id, "path": relpath})
            elif resolved.is_file():
                present += 1
            else:
                missing.append({"id": record_id, "path": relpath})

        self.report.files[result_key] = {
            "checked": len(rows),
            "present": present,
            "missing": len(missing),
            "outside_dataset_root": len(outside),
        }
        if missing:
            self._issue(
                severity,
                f"{result_key}_missing",
                f"{len(missing)} {result_key} files are missing",
                records=missing[: self.max_issue_details],
            )
        if outside:
            self._issue(
                "error",
                f"{result_key}_outside_dataset_root",
                f"{len(outside)} {result_key} paths resolve outside the dataset root",
                records=outside[: self.max_issue_details],
            )

    def _audit_files(self, connection: sqlite3.Connection) -> None:
        self._audit_file_rows(
            connection,
            table_name="raw_assets",
            id_sql='"asset_id"',
            path_column="relpath",
            where_sql="WHERE status = 'active'",
            severity="error",
            result_key="raw_assets",
        )
        self._audit_file_rows(
            connection,
            table_name="task_artifacts",
            id_sql="task_id || ':' || artifact_role",
            path_column="relpath",
            severity="error",
            result_key="task_artifacts",
        )
        self._audit_file_rows(
            connection,
            table_name="dataset_snapshots",
            id_sql='"snapshot_id"',
            path_column="document_relpath",
            severity="warning",
            result_key="dataset_snapshots",
        )
        self._audit_dataset_partitions(connection)

    def _audit_dataset_partitions(
        self,
        connection: sqlite3.Connection,
    ) -> None:
        if "dataset_partitions" not in self.report.counts:
            return
        rows = connection.execute(
            """
            SELECT
                partition_id,
                manifest_relpath,
                manifest_sha256,
                is_active
            FROM dataset_partitions
            ORDER BY created_at, rowid
            """
        ).fetchall()
        invalid: list[dict[str, Any]] = []
        present = 0
        for row in rows:
            partition_id = str(row[0])
            relpath = str(row[1] or "")
            expected_sha256 = str(row[2] or "")
            resolved = self._resolve_dataset_file(relpath)
            reason = ""
            if resolved is None:
                reason = "outside_dataset_root"
            elif not resolved.is_file():
                reason = "missing"
            else:
                try:
                    payload = json.loads(resolved.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    payload = None
                if not isinstance(payload, dict):
                    reason = "invalid_json"
                elif str(payload.get("partition_id") or "") != partition_id:
                    reason = "partition_id_mismatch"
                elif str(payload.get("manifest_sha256") or "") != expected_sha256:
                    reason = "registry_checksum_mismatch"
                elif not verify_partition_manifest(payload):
                    reason = "manifest_checksum_invalid"
                else:
                    present += 1
            if reason:
                invalid.append(
                    {
                        "partition_id": partition_id,
                        "path": relpath,
                        "is_active": bool(row[3]),
                        "reason": reason,
                    }
                )

        self.report.files["dataset_partitions"] = {
            "checked": len(rows),
            "present_and_valid": present,
            "invalid": len(invalid),
        }
        if invalid:
            self._issue(
                "error",
                "dataset_partitions_invalid",
                f"{len(invalid)} dataset partition manifests are invalid",
                records=invalid[: self.max_issue_details],
            )

    def _audit_annotation_table(
        self,
        connection: sqlite3.Connection,
        *,
        table_name: str,
        result_key: str,
    ) -> None:
        if table_name not in self.report.counts:
            return
        valid_labels = {item.value for item in AnnotationLabel}
        valid_detail_types = {item.value for item in DetailType}
        expected_labels = {
            detail_type.value: label.value
            for detail_type, label in DETAIL_TYPE_TO_LABEL.items()
        }
        rows = connection.execute(
            f"""
            SELECT label, detail_type
            FROM "{table_name}"
            """
        ).fetchall()
        missing_label = 0
        missing_detail_type = 0
        invalid_labels: dict[str, int] = {}
        invalid_detail_types: dict[str, int] = {}
        label_conflicts: dict[str, int] = {}
        for row in rows:
            label = str(row[0] or "").strip().lower()
            detail_type = str(row[1] or "").strip().lower()
            if not label:
                missing_label += 1
            elif label not in valid_labels:
                invalid_labels[label] = invalid_labels.get(label, 0) + 1
            if not detail_type:
                missing_detail_type += 1
            elif detail_type not in valid_detail_types:
                invalid_detail_types[detail_type] = invalid_detail_types.get(detail_type, 0) + 1
            expected = expected_labels.get(detail_type)
            if label in valid_labels and expected is not None and label != expected:
                key = f"{detail_type}:{label}->{expected}"
                label_conflicts[key] = label_conflicts.get(key, 0) + 1

        result = {
            "total": len(rows),
            "missing_label": missing_label,
            "missing_detail_type": missing_detail_type,
            "invalid_labels": invalid_labels,
            "invalid_detail_types": invalid_detail_types,
            "label_conflicts": label_conflicts,
        }
        self.report.annotations[result_key] = result
        if invalid_labels:
            self._issue(
                "error",
                "invalid_annotation_labels",
                f"{result_key} contains invalid coarse labels",
                values=invalid_labels,
            )
        if invalid_detail_types:
            self._issue(
                "error",
                "invalid_annotation_detail_types",
                f"{result_key} contains invalid detail types",
                values=invalid_detail_types,
            )
        if label_conflicts:
            self._issue(
                "warning",
                "annotation_label_conflicts",
                f"{result_key} contains coarse/detail label conflicts",
                values=label_conflicts,
            )
        if missing_label:
            self._issue(
                "warning",
                "annotation_labels_missing",
                f"{result_key} contains {missing_label} boxes without a coarse label",
                count=missing_label,
            )
        if missing_detail_type:
            self._issue(
                "warning",
                "annotation_detail_types_missing",
                f"{result_key} contains {missing_detail_type} boxes without a detail type",
                count=missing_detail_type,
            )

    def _audit_cached_counts(self, connection: sqlite3.Connection) -> None:
        if {
            "tasks",
            "task_annotation_boxes_current",
        }.issubset(self.report.counts):
            rows = connection.execute(
                """
                SELECT
                    t.task_id,
                    t.current_annotation_count,
                    COUNT(b.box_index) AS actual_count
                FROM tasks t
                LEFT JOIN task_annotation_boxes_current b ON b.task_id = t.task_id
                GROUP BY t.task_id
                HAVING t.current_annotation_count != COUNT(b.box_index)
                """
            ).fetchall()
            if rows:
                self._issue(
                    "error",
                    "task_annotation_count_mismatch",
                    f"{len(rows)} tasks have inconsistent cached annotation counts",
                    records=[
                        {
                            "task_id": str(row[0]),
                            "cached": int(row[1]),
                            "actual": int(row[2]),
                        }
                        for row in rows[: self.max_issue_details]
                    ],
                )

        if {
            "task_ai_prelabels",
            "task_ai_prelabel_boxes",
        }.issubset(self.report.counts):
            rows = connection.execute(
                """
                SELECT
                    p.prelabel_id,
                    p.box_count,
                    COUNT(b.box_index) AS actual_count
                FROM task_ai_prelabels p
                LEFT JOIN task_ai_prelabel_boxes b ON b.prelabel_id = p.prelabel_id
                GROUP BY p.prelabel_id
                HAVING p.box_count != COUNT(b.box_index)
                """
            ).fetchall()
            if rows:
                self._issue(
                    "error",
                    "prelabel_box_count_mismatch",
                    f"{len(rows)} prelabels have inconsistent cached box counts",
                    records=[
                        {
                            "prelabel_id": str(row[0]),
                            "cached": int(row[1]),
                            "actual": int(row[2]),
                        }
                        for row in rows[: self.max_issue_details]
                    ],
                )

    @staticmethod
    def _artifact_metadata(metadata_json: str) -> dict[str, Any]:
        try:
            metadata = json.loads(metadata_json or "{}")
        except (TypeError, ValueError):
            return {}
        if not isinstance(metadata, dict):
            return {}
        artifact = metadata.get("artifact")
        return artifact if isinstance(artifact, dict) else {}

    def _audit_models(self, connection: sqlite3.Connection) -> None:
        if "model_registry" not in self.report.counts:
            return
        rows = connection.execute(
            """
            SELECT model_id, artifact_path, is_promoted, metadata_json
            FROM model_registry
            ORDER BY created_at
            """
        ).fetchall()
        missing: list[dict[str, Any]] = []
        hash_mismatches: list[dict[str, Any]] = []
        valid = 0
        hashes_verified = 0
        for row in rows:
            model_id = str(row[0])
            artifact_path = str(row[1] or "")
            is_promoted = bool(row[2])
            resolved = self._resolve_dataset_file(artifact_path)
            if resolved is None or not resolved.is_file():
                missing.append(
                    {
                        "model_id": model_id,
                        "artifact_path": artifact_path,
                        "is_promoted": is_promoted,
                    }
                )
                continue
            valid += 1
            artifact_metadata = self._artifact_metadata(str(row[3] or "{}"))
            expected_sha256 = str(artifact_metadata.get("sha256") or "").strip().lower()
            if self.verify_model_hashes and expected_sha256:
                hashes_verified += 1
                actual_sha256 = self._sha256(resolved)
                if actual_sha256 != expected_sha256:
                    hash_mismatches.append(
                        {
                            "model_id": model_id,
                            "expected_sha256": expected_sha256,
                            "actual_sha256": actual_sha256,
                        }
                    )

        self.report.models = {
            "registered": len(rows),
            "valid_artifact": valid,
            "missing_artifact": len(missing),
            "hashes_verified": hashes_verified,
            "hash_mismatches": len(hash_mismatches),
        }
        if missing:
            promoted_missing = [item for item in missing if item["is_promoted"]]
            self._issue(
                "error" if promoted_missing else "warning",
                "model_artifacts_missing",
                f"{len(missing)} registered model artifacts are missing",
                records=missing[: self.max_issue_details],
            )
        if hash_mismatches:
            self._issue(
                "error",
                "model_artifact_hash_mismatch",
                f"{len(hash_mismatches)} model artifacts failed SHA256 verification",
                records=hash_mismatches[: self.max_issue_details],
            )

    def run(self) -> DatasetAuditReport:
        if not self.db_path.is_file():
            self._issue(
                "error",
                "database_missing",
                "dataset database file does not exist",
                path=str(self.db_path),
            )
            self.report.finish()
            return self.report

        connection = sqlite3.connect(
            f"{self.db_path.as_uri()}?mode=ro",
            uri=True,
            timeout=30,
        )
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("PRAGMA foreign_keys=ON")
            table_names = self._table_names(connection)
            self._audit_schema(connection, table_names)
            self._audit_files(connection)
            self._audit_annotation_table(
                connection,
                table_name="task_annotation_boxes_current",
                result_key="current_boxes",
            )
            self._audit_annotation_table(
                connection,
                table_name="annotation_revision_boxes",
                result_key="revision_boxes",
            )
            self._audit_cached_counts(connection)
            self._audit_models(connection)
        except sqlite3.DatabaseError as exc:
            self._issue(
                "error",
                "database_read_failed",
                "failed to read dataset database",
                error=str(exc),
            )
        finally:
            connection.close()

        self.report.finish()
        return self.report


def audit_dataset(
    dataset_root: Path,
    *,
    db_path: Path | None = None,
    max_issue_details: int = 50,
    verify_model_hashes: bool = True,
) -> DatasetAuditReport:
    return DatasetAuditor(
        dataset_root,
        db_path=db_path,
        max_issue_details=max_issue_details,
        verify_model_hashes=verify_model_hashes,
    ).run()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only integrity audit for a SCANN dataset",
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--db-path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-details", type=int, default=50)
    parser.add_argument("--skip-model-hashes", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return a non-zero exit code when warnings are present",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    report = audit_dataset(
        args.dataset_root,
        db_path=args.db_path,
        max_issue_details=args.max_details,
        verify_model_hashes=not args.skip_model_hashes,
    )
    payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    if args.output is not None:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    if not args.quiet:
        print(payload)
    if report.status == "error":
        return 2
    if report.status == "warning" and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
