"""HTTP client for the mock business systems.

`transport` is injectable so tests can talk to the mock ASGI app in-process.
"""
from __future__ import annotations

from typing import Any

import httpx

from .registry import ToolExecutionError


class MockClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 5.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._transport = transport

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict:
        url = f"{self._base_url}{path}"
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, transport=self._transport
            ) as client:
                resp = await client.request(method, url, **kwargs)
        except httpx.TimeoutException as exc:
            raise ToolExecutionError(f"backend timeout: {path}") from exc
        except httpx.HTTPError as exc:
            raise ToolExecutionError(f"backend error: {path}: {exc}") from exc

        if resp.status_code >= 400:
            detail: Any = resp.text[:200]
            try:
                detail = resp.json().get("detail", detail)
            except Exception:  # noqa: BLE001 - body may not be JSON
                pass
            raise ToolExecutionError(
                f"backend {resp.status_code}: {detail}", status_code=resp.status_code
            )
        return resp.json()

    async def get(self, path: str) -> dict:
        return await self._request("GET", path)

    async def post(
        self, path: str, json: dict | None = None, headers: dict | None = None
    ) -> dict:
        return await self._request("POST", path, json=json, headers=headers)
