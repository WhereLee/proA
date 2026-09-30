"""Core LLM types shared by all providers."""
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class LLMError(Exception):
    """Base class for all LLM failures."""


class LLMTimeoutError(LLMError):
    """Raised when a call exceeds the configured timeout."""


class LLMProviderError(LLMError):
    """Raised when a provider returns an error (HTTP error, bad response, ...)."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.model = model


@dataclass
class ChatMessage:
    role: str  # system | user | assistant | tool
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class LLMResult:
    text: str
    model: str
    provider: str
    latency_ms: float
    attempts: int = 1
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "provider": self.provider,
            "latency_ms": round(self.latency_ms, 2),
            "attempts": self.attempts,
            "usage": self.usage.to_dict(),
            "finish_reason": self.finish_reason,
        }


@runtime_checkable
class LLMProvider(Protocol):
    """A single-attempt provider. Retry/fallback is applied by ResilientLLM."""

    name: str

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> LLMResult: ...

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]: ...
