from __future__ import annotations

import sqlite3

from scann.core.fits_annotation_storage import load_v2_annotation_document


def test_load_v2_annotation_document_preserves_legacy_bbox_without_label(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    legacy_db = dataset_root / "annotations.db"
    connection = sqlite3.connect(legacy_db)
    try:
        connection.execute(
            """
            CREATE TABLE images (
                id TEXT PRIMARY KEY,
                file_name TEXT,
                label TEXT,
                detail_type TEXT,
                ai_suggestion TEXT,
                ai_confidence REAL,
                metadata_json TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE bboxes (
                image_id TEXT,
                box_index INTEGER,
                x INTEGER,
                y INTEGER,
                width INTEGER,
                height INTEGER,
                label TEXT,
                detail_type TEXT,
                confidence REAL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO images (id, file_name, label, detail_type, ai_suggestion, ai_confidence, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("PAIR_A", "PAIR_A.fts", None, None, None, None, None),
        )
        connection.execute(
            """
            INSERT INTO bboxes (image_id, box_index, x, y, width, height, label, detail_type, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("PAIR_A", 0, 10, 12, 8, 8, None, "disappeared_star", 1.0),
        )
        connection.commit()
    finally:
        connection.close()

    annotations_doc = load_v2_annotation_document(dataset_root)
    images = annotations_doc.get("images", [])

    assert len(images) == 1
    assert images[0]["annotations"][0]["label"] is None
    assert images[0]["annotations"][0]["detail_type"] == "disappeared_star"
