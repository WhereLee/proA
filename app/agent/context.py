"""Context engineering: token-budgeted history compression (checklist P1).

Long tasks must not let the context grow unbounded, yet compression must not
drop key facts (order ids, statuses, amounts, ...). We keep the system prompt and
the most recent turns verbatim and summarize the older turns into a compact
"key information" note produced from the original content (extractive by
default, so facts survive).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from collections.abc import Awaitable, Callable

from app.llm.base import ChatMessage

# Facts worth preserving verbatim through compression.
_KEY_FIELDS = (
    "order_id",
    "status",
    "refunded",
    "amount",
    "latest",
    "role",
    "name",
    "address",
    "tracking_no",
    "user_id",
    "ticket_id",
)

_VALUE_RE = re.compile(r'"([a-zA-Z_]+)"\s*:\s*("(?:[^"\\]|\\.)*"|true|false|-?\d+(?:\.\d+)?)')


def _estimate_text(text: str) -> int:
    """Token estimate: CJK ≈ 1 token/char, ASCII ≈ 1 token / 4 chars."""
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return cjk + (other + 3) // 4  # ceil(other / 4)


def estimate_tokens(messages: list[ChatMessage]) -> int:
    return sum(_estimate_text(m.content) for m in messages)


def local_summarize(messages: list[ChatMessage]) -> str:
    """Extractive summarizer: keep key field=value facts from the dropped turns."""
    facts: list[str] = []
    for message in messages:
        for key, value in _VALUE_RE.findall(message.content):
            if key in _KEY_FIELDS:
                facts.append(f"{key}={value.strip(chr(34))}")
    deduped = list(dict.fromkeys(facts))
    if not deduped:
        return "（较早的对话已省略）"
    return "关键信息：" + "; ".join(deduped)


@dataclass
class ContextManager:
    max_tokens: int = 6000
    keep_recent: int = 4
    summarizer: Callable[[list[ChatMessage]], Awaitable[str]] | None = None

    async def _summarize(self, messages: list[ChatMessage]) -> str:
        if self.summarizer is not None:
            return await self.summarizer(messages)
        return local_summarize(messages)

    async def compress(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        if estimate_tokens(messages) <= self.max_tokens:
            return messages

        system = [m for m in messages if m.role == "system"]
        rest = [m for m in messages if m.role != "system"]
        if len(rest) <= self.keep_recent:
            return messages

        older = rest[: -self.keep_recent]
        recent = rest[-self.keep_recent:]
        summary = await self._summarize(older)
        return [*system, ChatMessage("system", f"【历史摘要】{summary}"), *recent]
