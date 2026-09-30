"""RAG pipeline: optional multi-query rewrite -> hybrid retrieve -> optional rerank."""
from __future__ import annotations

from typing import Any

from .knowledge import KnowledgeBase
from .rerank import LLMReranker
from .rewrite import LLMQueryRewriter


class RagPipeline:
    def __init__(
        self,
        kb: KnowledgeBase,
        *,
        llm=None,
        reranker=None,
        multi_query: bool = False,
        rerank: bool = False,
        rewrites: int = 3,
        max_candidates: int = 8,
    ) -> None:
        self._kb = kb
        self._rewriter = LLMQueryRewriter(llm, n=rewrites) if (llm and multi_query) else None
        if reranker is not None:
            self._reranker = reranker
        elif rerank and llm is not None:
            self._reranker = LLMReranker(llm, max_candidates=max_candidates)
        elif rerank:
            from .factory import build_reranker

            self._reranker = build_reranker(llm=llm)
        else:
            self._reranker = None

    async def aretrieve(
        self, query: str, k: int = 3, *, scope: str | None = None
    ) -> list[dict[str, Any]]:
        queries = [query]
        if self._rewriter is not None:
            queries = await self._rewriter.rewrite(query)

        merged: dict[str, dict[str, Any]] = {}
        for sub_query in queries:
            for hit in self._kb.search(sub_query, k=max(k * 2, k), scope=scope):
                existing = merged.get(hit["chunk_id"])
                if existing is None or hit["score"] > existing["score"]:
                    merged[hit["chunk_id"]] = hit
        candidates = list(merged.values())

        if self._reranker is not None and candidates:
            return await self._reranker.rerank(query, candidates, k)

        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates[:k]
