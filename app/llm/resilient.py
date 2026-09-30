"""Resilience wrapper: timeout, retry with backoff, and model fallback chain.

Retry/fallback is intentionally kept OUTSIDE the provider so every provider
stays a single-attempt implementation. This is also where the "主备模型降级"
requirement (checklist P0) lives.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable

from .base import (
    ChatMessage,
    LLMError,
    LLMProviderError,
    LLMResult,
    LLMTimeoutError,
)


class ResilientLLM:
    name = "resilient"

    def __init__(
        self,
        inner,
        models: list[str],
        *,
        max_retries: int = 2,
        backoff_base: float = 0.2,
        timeout: float = 30.0,
        sleep: Callable[[float], "asyncio.Future"] | None = None,
    ) -> None:
        self._inner = inner
        self._models = list(models) or ["default"]
        self._max_retries = max(1, max_retries)
        self._backoff_base = backoff_base
        self._timeout = timeout
        self._sleep = sleep or asyncio.sleep

    @property
    def models(self) -> list[str]:
        return list(self._models)

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> LLMResult:
        models = [model] if model else self._models
        last_error: LLMError | None = None
        attempts = 0

        for current_model in models:
            for attempt in range(1, self._max_retries + 1):
                attempts += 1
                try:
                    result = await asyncio.wait_for(
                        self._inner.complete(
                            messages, model=current_model, temperature=temperature
                        ),
                        timeout=self._timeout,
                    )
                    result.attempts = attempts
                    return result
                except LLMError as exc:
                    last_error = exc
                except asyncio.TimeoutError:
                    last_error = LLMTimeoutError(
                        f"timeout after {self._timeout}s for model {current_model}"
                    )
                if attempt < self._max_retries:
                    await self._sleep(self._backoff_base * (2 ** (attempt - 1)))

        raise last_error or LLMProviderError("all models failed")

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        usage_out: dict | None = None,
    ) -> AsyncIterator[str]:
        """Stream tokens. Fallback applies only before the first token is emitted;
        a failure mid-stream is surfaced to the caller (documented limitation).
        """
        models = [model] if model else self._models
        last_error: LLMError | None = None

        for current_model in models:
            for attempt in range(1, self._max_retries + 1):
                agen = self._inner.stream(
                    messages,
                    model=current_model,
                    temperature=temperature,
                    usage_out=usage_out,
                )
                first: str | None = None
                try:
                    first = await asyncio.wait_for(
                        agen.__anext__(), timeout=self._timeout
                    )
                except StopAsyncIteration:
                    return
                except asyncio.TimeoutError:
                    await agen.aclose()
                    last_error = LLMTimeoutError(
                        f"timeout after {self._timeout}s for model {current_model}"
                    )
                except LLMError as exc:
                    await agen.aclose()
                    last_error = exc

                if first is not None:
                    yield first
                    async for chunk in agen:
                        yield chunk
                    return
                if attempt < self._max_retries:
                    await self._sleep(self._backoff_base * (2 ** (attempt - 1)))

        raise last_error or LLMProviderError("all models failed")


__all__ = ["ResilientLLM"]
