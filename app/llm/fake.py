"""Deterministic offline provider used for local dev and tests (LLM_PROVIDER=fake)."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from .base import ChatMessage, LLMResult, Usage


class FakeLLM:
    name = "fake"

    def __init__(
        self,
        *,
        prefix: str = "（fake）",
        latency_ms: float = 0.0,
        chunk_delay: float = 0.0,
    ) -> None:
        self._prefix = prefix
        self._latency_ms = latency_ms
        self._chunk_delay = chunk_delay

    def _reply_for(self, messages: list[ChatMessage]) -> str:
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )
        return (
            f"{self._prefix}已收到你的请求：{last_user}。"
            "这是阶段 0 的占位回复，用于验证流式与 trace 链路。"
        )

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> LLMResult:
        if self._latency_ms:
            await asyncio.sleep(self._latency_ms / 1000)
        text = self._reply_for(messages)
        return LLMResult(
            text=text,
            model=model or "fake-model",
            provider=self.name,
            latency_ms=self._latency_ms,
            usage=Usage(
                prompt_tokens=len(" ".join(m.content for m in messages)),
                completion_tokens=len(text),
                total_tokens=len(" ".join(m.content for m in messages)) + len(text),
            ),
            finish_reason="stop",
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        usage_out: dict | None = None,
    ) -> AsyncIterator[str]:
        text = self._reply_for(messages)
        for ch in text:
            if self._chunk_delay:
                await asyncio.sleep(self._chunk_delay)
            yield ch
