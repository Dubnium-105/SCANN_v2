"""Versioned SQLite schema migrations for the dataset database.

The base dataset schema is still created by :class:`DatasetStorage` so that
legacy and freshly-created databases remain compatible.  All schema changes
after that baseline must be registered here and are applied transactionally.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable


MigrationHandler = Callable[[sqlite3.Connection], None]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class SchemaMigration:
    """One immutable, versioned database migration."""

    migration_id: int
    name: str
    checksum_source: str
    apply: MigrationHandler

    @property
    def checksum(self) -> str:
        payload = (
            f"{int(self.migration_id)}\n"
            f"{str(self.name).strip()}\n"
            f"{self.checksum_source}"
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def ensure_migration_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            migration_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            checksum TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )
        """
    )


def _validate_migrations(migrations: Iterable[SchemaMigration]) -> list[SchemaMigration]:
    ordered = sorted(list(migrations), key=lambda item: int(item.migration_id))
    ids: set[int] = set()
    names: set[str] = set()
    for migration in ordered:
        migration_id = int(migration.migration_id)
        name = str(migration.name).strip()
        if migration_id <= 0:
            raise ValueError("schema migration IDs must be positive")
        if not name:
            raise ValueError(f"schema migration {migration_id} has an empty name")
        if migration_id in ids:
            raise ValueError(f"duplicate schema migration ID: {migration_id}")
        if name in names:
            raise ValueError(f"duplicate schema migration name: {name}")
        ids.add(migration_id)
        names.add(name)
    return ordered


def apply_schema_migrations(
    connection: sqlite3.Connection,
    migrations: Iterable[SchemaMigration],
) -> list[int]:
    """Apply pending migrations and return the IDs applied by this call.

    Every migration runs inside a savepoint.  A failed migration rolls back its
    own statements and is not recorded.  Previously-applied migrations must
    keep the exact same name and checksum.
    """

    ordered = _validate_migrations(migrations)
    ensure_migration_table(connection)

    rows = connection.execute(
        """
        SELECT migration_id, name, checksum
        FROM schema_migrations
        ORDER BY migration_id
        """
    ).fetchall()
    applied = {
        int(row[0]): {
            "name": str(row[1]),
            "checksum": str(row[2]),
        }
        for row in rows
    }
    known_ids = {int(item.migration_id) for item in ordered}
    unknown_ids = sorted(set(applied) - known_ids)
    if unknown_ids:
        raise RuntimeError(
            "database contains schema migrations newer than this application: "
            + ", ".join(str(item) for item in unknown_ids)
        )

    applied_now: list[int] = []
    for migration in ordered:
        migration_id = int(migration.migration_id)
        existing = applied.get(migration_id)
        if existing is not None:
            if existing["name"] != migration.name or existing["checksum"] != migration.checksum:
                raise RuntimeError(
                    f"schema migration {migration_id} does not match its applied checksum"
                )
            continue

        savepoint = f"schema_migration_{migration_id}"
        connection.execute(f"SAVEPOINT {savepoint}")
        try:
            migration.apply(connection)
            connection.execute(
                """
                INSERT INTO schema_migrations (
                    migration_id,
                    name,
                    checksum,
                    applied_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    migration_id,
                    migration.name,
                    migration.checksum,
                    _utc_now_iso(),
                ),
            )
            connection.execute(f"RELEASE SAVEPOINT {savepoint}")
        except Exception:
            connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            connection.execute(f"RELEASE SAVEPOINT {savepoint}")
            raise
        applied_now.append(migration_id)

    return applied_now


def _record_dataset_schema_baseline(_connection: sqlite3.Connection) -> None:
    """No-op marker for the schema that predates versioned migrations."""


def _create_dataset_partitions(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE dataset_partitions (
            partition_id TEXT PRIMARY KEY,
            partition_name TEXT NOT NULL,
            manifest_relpath TEXT NOT NULL UNIQUE,
            manifest_sha256 TEXT NOT NULL,
            taxonomy_version TEXT NOT NULL,
            split_strategy TEXT NOT NULL,
            seed INTEGER NOT NULL,
            task_count INTEGER NOT NULL DEFAULT 0,
            train_task_count INTEGER NOT NULL DEFAULT 0,
            validation_task_count INTEGER NOT NULL DEFAULT 0,
            test_task_count INTEGER NOT NULL DEFAULT 0,
            is_active INTEGER NOT NULL DEFAULT 0,
            activated_at TEXT,
            created_by TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX idx_dataset_partitions_created_at
        ON dataset_partitions(created_at)
        """
    )
    connection.execute(
        """
        CREATE UNIQUE INDEX idx_dataset_partitions_one_active
        ON dataset_partitions(is_active)
        WHERE is_active = 1
        """
    )


DATASET_SCHEMA_MIGRATIONS: tuple[SchemaMigration, ...] = (
    SchemaMigration(
        migration_id=1,
        name="dataset_schema_baseline",
        checksum_source=(
            "SCANN dataset schema through commit caf95d44: raw assets, tasks, "
            "artifacts, annotations, revisions, prelabels, workers, snapshots, "
            "training jobs/runs, and model registry."
        ),
        apply=_record_dataset_schema_baseline,
    ),
    SchemaMigration(
        migration_id=2,
        name="create_dataset_partitions",
        checksum_source=(
            "Create immutable dataset_partitions registry with manifest path/hash, "
            "taxonomy, split strategy, task counts, activation state, and metadata."
        ),
        apply=_create_dataset_partitions,
    ),
)


def apply_dataset_schema_migrations(connection: sqlite3.Connection) -> list[int]:
    return apply_schema_migrations(connection, DATASET_SCHEMA_MIGRATIONS)
