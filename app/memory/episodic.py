"""Episodic memory: distilled experience from past runs, recalled per task."""
from __future__ import annotations

from datetime import datetime, timezone

from app.persistence import Database, get_default_db

from .bm25 import rank


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EpisodicMemory:
    def __init__(self, db: Database | None = None, *, cap: int = 200) -> None:
        self._db = db or get_default_db()
        self._cap = cap

    def record(
        self,
        *,
        run_id: str,
        domain: str,
        role: str,
        task: str,
        outcome: str,
        tools: str,
        summary: str,
    ) -> None:
        self._db.execute(
            "INSERT INTO episodes(run_id, domain, role, task, outcome, tools, summary, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, domain, role, task, outcome, tools, summary, _iso()),
        )
        self._evict(domain, role)

    def _evict(self, domain: str, role: str) -> None:
        self._db.execute(
            "DELETE FROM episodes WHERE domain = ? AND role = ? AND id NOT IN ("
            "  SELECT id FROM episodes WHERE domain = ? AND role = ? "
            "  ORDER BY id DESC LIMIT ?)",
            (domain, role, domain, role, self._cap),
        )

    def recall(self, *, domain: str, role: str, query: str, k: int = 3) -> list[dict]:
        rows = self._db.query(
            "SELECT * FROM episodes WHERE domain = ? AND role = ? ORDER BY id DESC LIMIT ?",
            (domain, role, self._cap),
        )
        if not rows:
            return []
        # BM25 over recent experiences, ranked against the current task.
        texts = [f"{row['task']} {row['summary']}" for row in rows]
        ranked = rank(query, texts, k)
        return [rows[i] for i, _score in ranked]

    def count(self, *, domain: str | None = None, role: str | None = None) -> int:
        clauses, params = [], []
        if domain is not None:
            clauses.append("domain = ?")
            params.append(domain)
        if role is not None:
            clauses.append("role = ?")
            params.append(role)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._db.query(f"SELECT COUNT(*) AS c FROM episodes{where}", tuple(params))
        return rows[0]["c"] if rows else 0

    def reset(self) -> None:
        self._db.execute("DELETE FROM episodes")
