"""SQLite-backed persistence for run checkpoints and sessions (stdlib only).

A single connection guarded by a lock; JSON payloads keep the schema simple and
the stored run context fully round-trippable.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from app.config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    state      TEXT NOT NULL,
    payload    TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS episodes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT,
    domain     TEXT,
    role       TEXT,
    task       TEXT,
    outcome    TEXT,
    tools      TEXT,
    summary    TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS knowledge_docs (
    doc_id     TEXT PRIMARY KEY,
    title      TEXT,
    source     TEXT,
    scope      TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS knowledge_chunks (
    chunk_id   TEXT PRIMARY KEY,
    doc_id     TEXT,
    ordinal    INTEGER,
    text       TEXT,
    embedding  TEXT
);
"""


class Database:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
            self._migrate()

    def _migrate(self) -> None:
        """Add columns introduced after a DB was first created."""
        self._ensure_column("knowledge_docs", "scope", "TEXT")
        self._ensure_column("knowledge_chunks", "embedding", "TEXT")

    def _ensure_column(self, table: str, column: str, column_type: str) -> None:
        columns = [row[1] for row in self._conn.execute(f"PRAGMA table_info({table})")]
        if column not in columns:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
            self._conn.commit()

    # ---- runs -----------------------------------------------------------

    def upsert_run(
        self, run_id: str, state: str, payload: dict[str, Any], updated_at: str
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO runs (run_id, state, payload, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET state=excluded.state, "
                "payload=excluded.payload, updated_at=excluded.updated_at",
                (run_id, state, json.dumps(payload, ensure_ascii=False), updated_at),
            )
            self._conn.commit()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            "run_id": row["run_id"],
            "state": row["state"],
            "payload": json.loads(row["payload"]),
            "updated_at": row["updated_at"],
        }

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM runs ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            {
                "run_id": r["run_id"],
                "state": r["state"],
                "payload": json.loads(r["payload"]),
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]

    # ---- sessions -------------------------------------------------------

    def upsert_session(
        self, session_id: str, payload: dict[str, Any], updated_at: str
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions (session_id, payload, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET payload=excluded.payload, "
                "updated_at=excluded.updated_at",
                (session_id, json.dumps(payload, ensure_ascii=False), updated_at),
            )
            self._conn.commit()

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            self._conn.commit()

    def evict_runs(self, limit: int) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM runs WHERE run_id NOT IN "
                "(SELECT run_id FROM runs ORDER BY updated_at DESC LIMIT ?)",
                (limit,),
            )
            self._conn.commit()

    def evict_sessions(self, limit: int) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM sessions WHERE session_id NOT IN "
                "(SELECT session_id FROM sessions ORDER BY updated_at DESC LIMIT ?)",
                (limit,),
            )
            self._conn.commit()

    def execute(self, sql: str, params: tuple = ()) -> None:
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    def query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


_default_db: Database | None = None


def get_default_db() -> Database:
    global _default_db
    if _default_db is None:
        _default_db = Database(settings.db_path)
    return _default_db
