"""OpenAI-compatible chat provider (works with OpenAI, DeepSeek, Qwen, vLLM, ...)."""
from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

import httpx

from .base import (
    ChatMessage,
    LLMProviderError,
    LLMResult,
    LLMTimeoutError,
    Usage,
)


class OpenAICompatibleLLM:
    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout: float = 30.0,
        auth_header: str = "Authorization",
        auth_prefix: str = "Bearer ",
        extra_body: dict | None = None,
        default_temperature: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._auth_header = auth_header
        self._auth_prefix = auth_prefix
        self._extra_body = extra_body or {}
        self._default_temperature = default_temperature
        self._transport = transport

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._auth_header.lower() == "api-key":
            # e.g. Xiaomi MiMo also accepts `api-key: <key>`
            headers["api-key"] = self._api_key
        else:
            headers[self._auth_header] = f"{self._auth_prefix}{self._api_key}"
        return headers

    def _payload(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float | None,
        stream: bool,
    ) -> dict:
        payload: dict = {
            "model": model,
            "messages": [m.to_dict() for m in messages],
            "stream": stream,
        }
        # Explicit per-call temperature wins; otherwise fall back to the
        # configured default. Never send null.
        effective = temperature if temperature is not None else self._default_temperature
        if effective is not None:
            payload["temperature"] = effective
        if self._extra_body:
            payload.update(self._extra_body)
        return payload

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> LLMResult:
        model = model or "gpt-4o-mini"
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, transport=self._transport
            ) as client:
                resp = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers=self._headers(),
                    json=self._payload(messages, model, temperature, False),
                )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise LLMProviderError(str(exc), model=model) from exc

        if resp.status_code >= 400:
            raise LLMProviderError(
                f"HTTP {resp.status_code}: {resp.text[:200]}",
                status_code=resp.status_code,
                model=model,
            )

        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        usage = data.get("usage") or {}
        return LLMResult(
            text=((choice.get("message") or {}).get("content")) or "",
            model=data.get("model", model),
            provider=self.name,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            usage=Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
            finish_reason=choice.get("finish_reason"),
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        usage_out: dict | None = None,
    ) -> AsyncIterator[str]:
        model = model or "gpt-4o-mini"
        payload = self._payload(messages, model, temperature, True)
        payload["stream_options"] = {"include_usage": True}
        usage: dict | None = None
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, transport=self._transport
            ) as client:
                async with client.stream(
                    "POST",
                    f"{self._base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                ) as resp:
                    if resp.status_code >= 400:
                        body = await resp.aread()
                        raise LLMProviderError(
                            f"HTTP {resp.status_code}: {body[:200]!r}",
                            status_code=resp.status_code,
                            model=model,
                        )
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        chunk = line[len("data:") :].strip()
                        if chunk == "[DONE]":
                            break
                        try:
                            obj = json.loads(chunk)
                        except json.JSONDecodeError:
                            continue
                        if obj.get("usage"):
                            usage = obj["usage"]
                        delta = ((obj.get("choices") or [{}])[0].get("delta") or {}).get(
                            "content"
                        )
                        if delta:
                            yield delta
                    if usage_out is not None and usage is not None:
                        usage_out.update(usage)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise LLMProviderError(str(exc), model=model) from exc
