"""SQLite database layer for prediction history."""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


class Database:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = sqlite3.connect(str(self.db_path), timeout=10)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    crop TEXT NOT NULL,
                    disease TEXT NOT NULL,
                    condition TEXT,
                    confidence REAL NOT NULL,
                    model TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    top_predictions TEXT
                );

                CREATE TABLE IF NOT EXISTS model_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def insert_prediction(
        self,
        filename: str,
        crop: str,
        disease: str,
        confidence: float,
        model: str,
        mode: str,
        condition: Optional[str] = None,
        top_predictions: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        timestamp = datetime.now(timezone.utc).isoformat()
        top_json = json.dumps(top_predictions) if top_predictions else None
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO prediction_history
                    (timestamp, filename, crop, disease, condition, confidence, model, mode, top_predictions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    filename,
                    crop,
                    disease,
                    condition,
                    float(confidence),
                    model,
                    mode,
                    top_json,
                ),
            )
            return int(cur.lastrowid)

    def list_predictions(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, timestamp, filename, crop, disease, condition, confidence, model, mode, top_predictions
                FROM prediction_history
                ORDER BY id DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
            items: List[Dict[str, Any]] = []
            for row in rows:
                top = None
                if row["top_predictions"]:
                    try:
                        top = json.loads(row["top_predictions"])
                    except json.JSONDecodeError:
                        top = None
                items.append(
                    {
                        "id": row["id"],
                        "timestamp": row["timestamp"],
                        "filename": row["filename"],
                        "crop": row["crop"],
                        "disease": row["disease"],
                        "condition": row["condition"],
                        "confidence": row["confidence"],
                        "model": row["model"],
                        "mode": row["mode"],
                        "top_predictions": top,
                    }
                )
            return items

    def count_predictions(self) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM prediction_history").fetchone()
            return int(row["c"]) if row else 0

    def delete_prediction(self, prediction_id: int) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM prediction_history WHERE id = ?",
                (prediction_id,),
            )
            return int(cur.rowcount)

    def clear_history(self) -> int:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM prediction_history")
            return int(cur.rowcount)

    def set_metadata(self, key: str, value: Dict[str, Any]) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO model_metadata (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (key, json.dumps(value), datetime.now(timezone.utc).isoformat()),
            )

    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM model_metadata WHERE key = ?", (key,)
            ).fetchone()
            if not row:
                return None
            try:
                return json.loads(row["value"])
            except json.JSONDecodeError:
                return None
