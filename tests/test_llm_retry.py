import asyncio

import pytest

from app.llm.base import ChatMessage, LLMProviderError, LLMResult
from app.llm.resilient import ResilientLLM


class FlakyLLM:
    """Test double: fails N times, and/or fails for specific model names."""

    name = "flaky"

    def __init__(self, *, fail_times: int = 0, fail_models: tuple[str, ...] = ()) -> None:
        self.fail_times = fail_times
        self.fail_models = set(fail_models)
        self.calls = 0

    def _maybe_fail(self, model: str | None) -> None:
        self.calls += 1
        if model in self.fail_models:
            raise LLMProviderError(f"model {model} down", model=model)
        if self.calls <= self.fail_times:
            raise LLMProviderError("transient error", model=model)

    async def complete(self, messages, *, model=None, temperature=None):
        self._maybe_fail(model)
        return LLMResult(text="ok", model=model or "m", provider=self.name, latency_ms=0.0)

    async def stream(self, messages, *, model=None, temperature=None, usage_out=None):
        self._maybe_fail(model)
        yield "o"
        yield "k"


def test_retry_then_success():
    inner = FlakyLLM(fail_times=2)
    llm = ResilientLLM(inner, ["m1"], max_retries=3, backoff_base=0.0)
    result = asyncio.run(llm.complete([ChatMessage("user", "hi")]))
    assert result.text == "ok"
    assert result.attempts == 3
    assert inner.calls == 3


def test_exhausts_retries_then_raises():
    inner = FlakyLLM(fail_times=99)
    llm = ResilientLLM(inner, ["m1"], max_retries=2, backoff_base=0.0)
    with pytest.raises(LLMProviderError):
        asyncio.run(llm.complete([ChatMessage("user", "hi")]))
    assert inner.calls == 2


def test_fallback_to_next_model():
    inner = FlakyLLM(fail_models=("bad",))
    llm = ResilientLLM(inner, ["bad", "good"], max_retries=2, backoff_base=0.0)
    result = asyncio.run(llm.complete([ChatMessage("user", "hi")]))
    assert result.model == "good"
    assert result.attempts == 3


def test_stream_retries_then_success():
    inner = FlakyLLM(fail_times=1)
    llm = ResilientLLM(inner, ["m1"], max_retries=2, backoff_base=0.0)

    async def collect():
        return [chunk async for chunk in llm.stream([ChatMessage("user", "hi")])]

    assert asyncio.run(collect()) == ["o", "k"]


def test_stream_falls_back_to_next_model():
    inner = FlakyLLM(fail_models=("bad",))
    llm = ResilientLLM(inner, ["bad", "good"], max_retries=1, backoff_base=0.0)

    async def collect():
        return [chunk async for chunk in llm.stream([ChatMessage("user", "hi")])]

    assert asyncio.run(collect()) == ["o", "k"]
