"""Knowledge store interface + SQLite implementation."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from app.persistence import Database

from .embedding import cosine


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@runtime_checkable
class KnowledgeStore(Protocol):
    def upsert_document(
        self,
        doc_id: str,
        title: str,
        source: str,
        scope: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> int: ...

    def list_chunks(self, scope: str | None) -> list[dict]: ...

    def vector_search(
        self, query_vector: list[float], k: int, scope: str | None
    ) -> list[str]: ...

    def stats(self) -> dict: ...

    def reset(self) -> None: ...


class SqliteKnowledgeStore:
    """Default store: embeddings as JSON in SQLite, brute-force cosine."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def upsert_document(
        self, doc_id, title, source, scope, chunks, embeddings
    ) -> int:
        self._db.execute(
            "INSERT OR REPLACE INTO knowledge_docs(doc_id, title, source, scope, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (doc_id, title, source, scope, _iso()),
        )
        self._db.execute("DELETE FROM knowledge_chunks WHERE doc_id = ?", (doc_id,))
        for ordinal, (text, vector) in enumerate(zip(chunks, embeddings)):
            self._db.execute(
                "INSERT INTO knowledge_chunks(chunk_id, doc_id, ordinal, text, embedding) "
                "VALUES (?, ?, ?, ?, ?)",
                (uuid.uuid4().hex, doc_id, ordinal, text, json.dumps(vector)),
            )
        return len(chunks)

    def list_chunks(self, scope: str | None) -> list[dict]:
        sql = (
            "SELECT c.chunk_id, c.doc_id, c.text, c.embedding, "
            "d.title, d.source, d.scope "
            "FROM knowledge_chunks c JOIN knowledge_docs d ON d.doc_id = c.doc_id"
        )
        params: tuple = ()
        if scope is not None:
            sql += " WHERE d.scope = ?"
            params = (scope,)
        return self._db.query(sql, params)

    def vector_search(self, query_vector, k, scope) -> list[str]:
        rows = self.list_chunks(scope)
        sims = [
            cosine(query_vector, json.loads(row["embedding"])) if row["embedding"] else 0.0
            for row in rows
        ]
        order = sorted(range(len(rows)), key=lambda i: sims[i], reverse=True)
        return [rows[i]["chunk_id"] for i in order[:k]]

    def stats(self) -> dict:
        docs = self._db.query("SELECT COUNT(*) AS c FROM knowledge_docs")
        chunks = self._db.query("SELECT COUNT(*) AS c FROM knowledge_chunks")
        return {"docs": docs[0]["c"], "chunks": chunks[0]["c"]}

    def reset(self) -> None:
        self._db.execute("DELETE FROM knowledge_chunks")
        self._db.execute("DELETE FROM knowledge_docs")
