"""Optional LLM multi-query rewriting for recall."""
from __future__ import annotations

from pydantic import BaseModel

from app.llm.base import ChatMessage
from app.llm.structured import parse_model

_REWRITE_PROMPT = """把用户问题改写成 {n} 条适合检索的关键词查询，覆盖不同表述，用于召回资料。

用户问题：{query}

只输出 JSON：{{"queries": ["查询1", "查询2"]}}"""


class _Queries(BaseModel):
    queries: list[str] = []


class LLMQueryRewriter:
    def __init__(self, llm, *, n: int = 3) -> None:
        self._llm = llm
        self._n = n

    async def rewrite(self, query: str) -> list[str]:
        messages = [
            ChatMessage("system", "你是检索查询改写器，只输出 JSON。"),
            ChatMessage("user", _REWRITE_PROMPT.format(n=self._n, query=query)),
        ]
        try:
            result = await self._llm.complete(messages, temperature=0.0)
            queries = parse_model(result.text, _Queries).queries
        except Exception:  # noqa: BLE001
            queries = []
        # always include the original query first, dedupe
        ordered = [query, *queries]
        seen: list[str] = []
        for item in ordered:
            item = (item or "").strip()
            if item and item not in seen:
                seen.append(item)
        return seen
