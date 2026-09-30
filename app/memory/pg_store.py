"""PostgreSQL + pgvector knowledge store (real vector database)."""
from __future__ import annotations

import uuid

import psycopg


class PgVectorStore:
    def __init__(
        self,
        dsn: str,
        *,
        dim: int = 768,
        chunks_table: str = "kb_chunks",
        docs_table: str = "kb_docs",
    ) -> None:
        self._dim = dim
        self._chunks = chunks_table
        self._docs = docs_table
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {self._docs}("
                "doc_id text PRIMARY KEY, title text, source text, scope text, "
                "created_at timestamptz DEFAULT now())"
            )
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {self._chunks}("
                "chunk_id text PRIMARY KEY, "
                f"doc_id text REFERENCES {self._docs}(doc_id) ON DELETE CASCADE, "
                f"ordinal int, text text, embedding vector({self._dim}))"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self._chunks}_emb_idx "
                f"ON {self._chunks} USING hnsw (embedding vector_cosine_ops)"
            )

    @staticmethod
    def _vector_literal(vector: list[float]) -> str:
        return "[" + ",".join(f"{x:.7f}" for x in vector) + "]"

    def upsert_document(self, doc_id, title, source, scope, chunks, embeddings) -> int:
        with self._conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {self._docs}(doc_id, title, source, scope) "
                "VALUES (%s, %s, %s, %s) ON CONFLICT(doc_id) DO UPDATE SET "
                "title=excluded.title, source=excluded.source, scope=excluded.scope",
                (doc_id, title, source, scope),
            )
            cur.execute(f"DELETE FROM {self._chunks} WHERE doc_id = %s", (doc_id,))
            for ordinal, (text, vector) in enumerate(zip(chunks, embeddings)):
                cur.execute(
                    f"INSERT INTO {self._chunks}(chunk_id, doc_id, ordinal, text, embedding) "
                    "VALUES (%s, %s, %s, %s, %s::vector)",
                    (uuid.uuid4().hex, doc_id, ordinal, text, self._vector_literal(vector)),
                )
        return len(chunks)

    def list_chunks(self, scope: str | None) -> list[dict]:
        sql = (
            f"SELECT c.chunk_id, c.doc_id, c.text, d.title, d.source, d.scope "
            f"FROM {self._chunks} c JOIN {self._docs} d ON d.doc_id = c.doc_id"
        )
        params: list = []
        if scope is not None:
            sql += " WHERE d.scope = %s"
            params.append(scope)
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
        return [dict(zip(columns, row)) for row in rows]

    def vector_search(self, query_vector, k, scope) -> list[str]:
        sql = (
            f"SELECT c.chunk_id FROM {self._chunks} c "
            f"JOIN {self._docs} d ON d.doc_id = c.doc_id"
        )
        params: list = []
        if scope is not None:
            sql += " WHERE d.scope = %s"
            params.append(scope)
        sql += " ORDER BY c.embedding <=> %s::vector LIMIT %s"
        params.extend([self._vector_literal(query_vector), k])
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return [row[0] for row in cur.fetchall()]

    def stats(self) -> dict:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self._docs}")
            docs = cur.fetchone()[0]
            cur.execute(f"SELECT COUNT(*) FROM {self._chunks}")
            chunks = cur.fetchone()[0]
        return {"docs": docs, "chunks": chunks}

    def reset(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self._docs}")  # chunks cascade
