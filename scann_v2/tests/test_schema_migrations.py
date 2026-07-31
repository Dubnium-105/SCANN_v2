from __future__ import annotations

import sqlite3

import pytest

from scann.core.dataset_storage import DatasetStorage
from scann.core.dataset_migrate import migrate_dataset_database
from scann.core.schema_migrations import (
    SchemaMigration,
    apply_schema_migrations,
)


def test_dataset_storage_records_schema_baseline(tmp_path):
    storage = DatasetStorage(tmp_path)

    storage.ensure_schema()

    with sqlite3.connect(storage.db_path) as connection:
        row = connection.execute(
            """
            SELECT migration_id, name, checksum, applied_at
            FROM schema_migrations
            """
        ).fetchone()
    assert row is not None
    assert row[0] == 1
    assert row[1] == "dataset_schema_baseline"
    assert len(row[2]) == 64
    assert row[3]


def test_dataset_storage_creates_discovery_lifecycle_tables(tmp_path):
    storage = DatasetStorage(tmp_path)
    storage.ensure_schema()

    with sqlite3.connect(storage.db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                """
            ).fetchall()
        }
        migration_ids = [
            row[0]
            for row in connection.execute(
                """
                SELECT migration_id
                FROM schema_migrations
                ORDER BY migration_id
                """
            ).fetchall()
        ]

    assert migration_ids == [1, 2, 3]
    assert {
        "evaluation_runs",
        "annotation_review_events",
        "active_learning_batches",
        "active_learning_items",
        "model_deployments",
    }.issubset(tables)


def test_schema_migrations_are_idempotent():
    connection = sqlite3.connect(":memory:")
    calls = []

    def apply_example(conn):
        calls.append("called")
        conn.execute("CREATE TABLE example (value TEXT)")

    migration = SchemaMigration(
        migration_id=1,
        name="example",
        checksum_source="create example table v1",
        apply=apply_example,
    )

    assert apply_schema_migrations(connection, [migration]) == [1]
    assert apply_schema_migrations(connection, [migration]) == []
    assert calls == ["called"]


def test_schema_migration_rejects_changed_checksum():
    connection = sqlite3.connect(":memory:")
    original = SchemaMigration(
        migration_id=1,
        name="example",
        checksum_source="v1",
        apply=lambda _connection: None,
    )
    changed = SchemaMigration(
        migration_id=1,
        name="example",
        checksum_source="v2",
        apply=lambda _connection: None,
    )
    apply_schema_migrations(connection, [original])

    with pytest.raises(RuntimeError, match="does not match"):
        apply_schema_migrations(connection, [changed])


def test_failed_schema_migration_rolls_back_and_is_not_recorded():
    connection = sqlite3.connect(":memory:")

    def apply_broken(conn):
        conn.execute("CREATE TABLE should_rollback (value TEXT)")
        raise RuntimeError("migration failed")

    migration = SchemaMigration(
        migration_id=1,
        name="broken",
        checksum_source="broken v1",
        apply=apply_broken,
    )

    with pytest.raises(RuntimeError, match="migration failed"):
        apply_schema_migrations(connection, [migration])

    table_names = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert "should_rollback" not in table_names
    assert connection.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == 0


def test_schema_migration_rejects_unknown_newer_database_migration():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE schema_migrations (
            migration_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            checksum TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        INSERT INTO schema_migrations (migration_id, name, checksum, applied_at)
        VALUES (99, 'future', 'checksum', '2026-01-01T00:00:00+00:00')
        """
    )

    with pytest.raises(RuntimeError, match="newer than this application"):
        apply_schema_migrations(connection, [])


def test_dataset_migration_can_target_an_explicit_database_copy(tmp_path):
    dataset_root = tmp_path / "dataset"
    database_copy = tmp_path / "validation" / "copied.db"

    result = migrate_dataset_database(
        dataset_root,
        db_path=database_copy,
    )

    assert result["database_path"] == str(database_copy.resolve())
    assert result["integrity_check"] == ["ok"]
    assert result["applied_migrations"][0]["migration_id"] == 1
    assert not (dataset_root / "scann_dataset.db").exists()
