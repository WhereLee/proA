"""Concurrency gate and token-bucket rate limiter (checklist P0: 限流与并发闸门)."""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator


class ConcurrencyGate:
    """Bounds the number of concurrent agent runs."""

    def __init__(self, limit: int) -> None:
        self._limit = max(1, limit)
        self._sem = asyncio.Semaphore(self._limit)
        self._active = 0
        self._max_active = 0

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def max_active(self) -> int:
        return self._max_active

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[None]:
        await self._sem.acquire()
        self._active += 1
        self._max_active = max(self._max_active, self._active)
        try:
            yield
        finally:
            self._active -= 1
            self._sem.release()


class TokenBucketLimiter:
    """Simple async token bucket, e.g. model-side RPM throttling."""

    def __init__(self, rate_per_min: float, burst: int | None = None) -> None:
        self._rate_per_sec = rate_per_min / 60.0
        self._capacity = float(burst if burst is not None else max(1, int(rate_per_min)))
        self._tokens = self._capacity
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()
        self.throttled_events = 0

    async def acquire(self) -> float:
        """Wait until a token is available; return the wait time in ms."""
        async with self._lock:
            now = time.monotonic()
            self._tokens = min(
                self._capacity, self._tokens + (now - self._updated) * self._rate_per_sec
            )
            self._updated = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return 0.0
            wait = (1.0 - self._tokens) / self._rate_per_sec
            self._tokens = 0.0
            self.throttled_events += 1
        await asyncio.sleep(wait)
        return wait * 1000.0


class ThrottledLLM:
    """Wraps an LLM provider with a token-bucket rate limiter (model RPM)."""

    def __init__(self, inner, limiter: TokenBucketLimiter) -> None:
        self._inner = inner
        self._limiter = limiter

    @property
    def name(self) -> str:
        return getattr(self._inner, "name", "throttled")

    async def complete(self, messages, *, model=None, temperature=None):
        await self._limiter.acquire()
        return await self._inner.complete(messages, model=model, temperature=temperature)

    async def stream(self, messages, *, model=None, temperature=None, usage_out=None):
        await self._limiter.acquire()
        async for chunk in self._inner.stream(
            messages, model=model, temperature=temperature, usage_out=usage_out
        ):
            yield chunk
