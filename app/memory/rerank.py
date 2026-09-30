"""Rerankers: identity (rule) and optional LLM reranker for top-k candidates."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from app.llm.base import ChatMessage
from app.llm.structured import parse_model

_RERANK_PROMPT = """你在做检索结果重排。根据用户问题，对候选片段按相关性从高到低排序。

用户问题：{query}

候选片段（编号从 0 开始）：
{candidates}

只输出 JSON：{{"order": [按相关性排序的编号列表]}}"""


class RuleReranker:
    name = "rule"

    async def rerank(
        self, query: str, candidates: list[dict[str, Any]], k: int
    ) -> list[dict[str, Any]]:
        ordered = sorted(candidates, key=lambda c: c.get("score", 0.0), reverse=True)
        return ordered[:k]


class _Order(BaseModel):
    order: list[int] = []


class LLMReranker:
    name = "llm"

    def __init__(self, llm, *, max_candidates: int = 8) -> None:
        self._llm = llm
        self._max_candidates = max_candidates

    async def rerank(
        self, query: str, candidates: list[dict[str, Any]], k: int
    ) -> list[dict[str, Any]]:
        candidates = candidates[: self._max_candidates]
        if len(candidates) <= 1:
            return candidates[:k]

        listing = "\n".join(
            f"[{i}] {c.get('title', '')}：{c.get('text', '')[:120]}"
            for i, c in enumerate(candidates)
        )
        messages = [
            ChatMessage("system", "你是检索重排器，只输出 JSON。"),
            ChatMessage(
                "user", _RERANK_PROMPT.format(query=query, candidates=listing)
            ),
        ]
        try:
            result = await self._llm.complete(messages, temperature=0.0)
            order = parse_model(result.text, _Order).order
        except Exception:  # noqa: BLE001 - rerank must never break retrieval
            return candidates[:k]

        seen: list[int] = []
        for index in order:
            if 0 <= index < len(candidates) and index not in seen:
                seen.append(index)
        for index in range(len(candidates)):
            if index not in seen:
                seen.append(index)
        return [candidates[i] for i in seen][:k]
