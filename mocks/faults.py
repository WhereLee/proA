"""Fault injection for the mock business systems (checklist P0: failure paths).

Modes:
- error   : return 5xx before doing anything
- timeout : sleep then return 504 (simulates a slow/timing-out backend)
- partial : perform the side effect, THEN return 5xx (the hardest case: the
            caller sees a failure but the effect already happened -> drives
            idempotency design later)
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel

Mode = Literal["none", "error", "timeout", "partial"]
_VALID_MODES = ("none", "error", "timeout", "partial")


@dataclass
class FaultState:
    mode: Mode = "none"
    remaining: int = 0            # 0 = unlimited, else auto-disables when exhausted
    target: str = "*"             # service name ("order"/"ticket"/"notify"/"user") or "*"
    timeout_seconds: float = 1.0
    status_code: int = 500
    message: str = "injected fault"

    def applies(self, service: str) -> bool:
        if self.mode == "none":
            return False
        if self.target not in ("*", service):
            return False
        return self.remaining != 0

    def consume(self) -> None:
        if self.remaining > 0:
            self.remaining -= 1
            if self.remaining == 0:
                self.mode = "none"


class FaultRequest(BaseModel):
    mode: Mode = "none"
    remaining: int = 0
    target: str = "*"
    timeout_seconds: float = 1.0
    status_code: int = 500
    message: str = "injected fault"


fault = FaultState()


def configure(request: FaultRequest) -> FaultState:
    if request.mode not in _VALID_MODES:
        raise ValueError(f"unsupported fault mode: {request.mode}")
    fault.mode = request.mode
    fault.remaining = request.remaining
    fault.target = request.target
    fault.timeout_seconds = request.timeout_seconds
    fault.status_code = request.status_code
    fault.message = request.message
    return fault


def reset() -> FaultState:
    fault.mode = "none"
    fault.remaining = 0
    fault.target = "*"
    fault.timeout_seconds = 1.0
    fault.status_code = 500
    fault.message = "injected fault"
    return fault


def snapshot() -> dict:
    return asdict(fault)


async def preflight(service: str) -> Mode:
    """Apply timeout/error before the operation; expose 'partial' to the caller."""
    if not fault.applies(service):
        return "none"
    mode = fault.mode
    fault.consume()
    if mode == "timeout":
        await asyncio.sleep(fault.timeout_seconds)
        raise HTTPException(status_code=504, detail=f"[fault] timeout: {fault.message}")
    if mode == "error":
        raise HTTPException(
            status_code=fault.status_code, detail=f"[fault] error: {fault.message}"
        )
    return mode


def postflight(mode: Mode) -> None:
    if mode == "partial":
        raise HTTPException(
            status_code=fault.status_code,
            detail=f"[fault] partial: side effect applied but {fault.message}",
        )
