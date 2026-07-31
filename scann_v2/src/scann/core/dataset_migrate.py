"""Command-line entry point for SCANN dataset schema migrations."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Iterable

from scann.core.dataset_storage import DatasetStorage
from scann.core.schema_migrations import DATASET_SCHEMA_MIGRATIONS


def migrate_dataset_database(
    dataset_root: Path,
    *,
    db_path: Path | None = None,
) -> dict[str, object]:
    storage = DatasetStorage(dataset_root, db_path=db_path)
    storage.ensure_schema()
    with sqlite3.connect(storage.db_path) as connection:
        rows = connection.execute(
            """
            SELECT migration_id, name, checksum, applied_at
            FROM schema_migrations
            ORDER BY migration_id
            """
        ).fetchall()
        integrity = [
            str(row[0])
            for row in connection.execute("PRAGMA integrity_check").fetchall()
        ]
    return {
        "database_path": str(storage.db_path),
        "latest_application_migration": max(
            (int(item.migration_id) for item in DATASET_SCHEMA_MIGRATIONS),
            default=0,
        ),
        "applied_migrations": [
            {
                "migration_id": int(row[0]),
                "name": str(row[1]),
                "checksum": str(row[2]),
                "applied_at": str(row[3]),
            }
            for row in rows
        ],
        "integrity_check": integrity,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply versioned schema migrations to a SCANN dataset database",
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--db-path", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    result = migrate_dataset_database(
        args.dataset_root,
        db_path=args.db_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["integrity_check"] == ["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
