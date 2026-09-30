"""Semantic / knowledge memory: hybrid (BM25 + vector) RAG over a pluggable store."""
from __future__ import annotations

from pathlib import Path

from app.persistence import Database, get_default_db

from .bm25 import rank
from .chunking import chunk_document
from .embedding import Embedder, HashingEmbedder
from .fusion import rrf_fuse
from .store import KnowledgeStore, SqliteKnowledgeStore


class KnowledgeBase:
    def __init__(
        self,
        db: Database | None = None,
        *,
        store: KnowledgeStore | None = None,
        embedder: Embedder | None = None,
        max_chars: int = 240,
        overlap: int = 1,
    ) -> None:
        self._store = store if store is not None else SqliteKnowledgeStore(db or get_default_db())
        self._embedder = embedder if embedder is not None else HashingEmbedder()
        self._max_chars = max_chars
        self._overlap = overlap

    # ---- ingestion ------------------------------------------------------

    def ingest(
        self, doc_id: str, title: str, text: str, source: str = "", scope: str = "default"
    ) -> int:
        chunks = chunk_document(text, max_chars=self._max_chars, overlap=self._overlap)
        embeddings = self._embedder.embed([f"{title} {c}" for c in chunks])
        return self._store.upsert_document(doc_id, title, source, scope, chunks, embeddings)

    def ingest_file(self, path: str, *, source: str = "", scope: str = "default") -> int:
        file = Path(path)
        return self.ingest(
            doc_id=file.stem,
            title=file.stem,
            text=file.read_text(encoding="utf-8"),
            source=source or str(file),
            scope=scope,
        )

    # ---- retrieval ------------------------------------------------------

    def search(
        self, query: str, k: int = 3, *, scope: str | None = None, use_vector: bool = True
    ) -> list[dict]:
        rows = self._store.list_chunks(scope)
        if not rows:
            return []
        by_id = {row["chunk_id"]: row for row in rows}

        texts = [f"{row['title']} {row['text']}" for row in rows]
        lexical = [rows[i]["chunk_id"] for i, _score in rank(query, texts, len(rows))]
        rankings = [lexical] if lexical else []

        if use_vector:
            query_vector = self._embedder.embed([query])[0]
            vector_ids = self._store.vector_search(query_vector, len(rows), scope)
            if vector_ids:
                rankings.append(vector_ids)

        fused = rrf_fuse(rankings)
        results = []
        for chunk_id, score in fused[: max(1, k)]:
            row = by_id.get(chunk_id)
            if row is None:
                continue
            results.append(
                {
                    "chunk_id": row["chunk_id"],
                    "doc_id": row["doc_id"],
                    "title": row["title"],
                    "source": row["source"],
                    "scope": row["scope"],
                    "text": row["text"],
                    "score": round(score, 6),
                }
            )
        return results

    # ---- admin ----------------------------------------------------------

    def count(self) -> dict:
        return self._store.stats()

    def reset(self) -> None:
        self._store.reset()
