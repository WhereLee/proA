"""Scripted provider: replays canned outputs keyed by user input.

Used to drive the agent loop deterministically in tests and to self-check the
eval harness offline. It is NOT a model; real metrics require a real provider.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from .base import ChatMessage, LLMResult, Usage

_DEFAULT_FINAL = '{"action":"final","answer":"（scripted：无可用脚本）"}'


class ScriptedLLM:
    name = "scripted"

    def __init__(self, script: dict[str, list[str]], *, default: str | None = None) -> None:
        self._script = {key: list(value) for key, value in script.items()}
        self._default = default or _DEFAULT_FINAL

    def _key(self, messages: list[ChatMessage]) -> str:
        user_contents = [m.content for m in messages if m.role == "user"]
        for content in reversed(user_contents):
            if self._script.get(content):
                return content
        return user_contents[0] if user_contents else ""

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> LLMResult:
        key = self._key(messages)
        queue = self._script.get(key)
        text = queue.pop(0) if queue else self._default
        return LLMResult(
            text=text,
            model=model or "scripted",
            provider=self.name,
            latency_ms=0.0,
            usage=Usage(),
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        usage_out: dict | None = None,
    ) -> AsyncIterator[str]:
        result = await self.complete(messages, model=model, temperature=temperature)
        for ch in result.text:
            yield ch
