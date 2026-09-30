"""Lightweight self-hosted tracer (checklist P0: full-chain trace).

Every run creates a Trace with named Spans. Traces are kept in memory and can be
fetched via the API, so "any run can be replayed to locate the failing step".
Swap this for Langfuse later without touching call sites.
"""
from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Iterator

from pydantic import BaseModel, Field

from app.config import settings
from app.security.redact import redact_value


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Span(BaseModel):
    span_id: str
    name: str
    start_ms: float
    end_ms: float | None = None
    duration_ms: float | None = None
    status: str = "ok"
    attributes: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class Trace(BaseModel):
    trace_id: str
    name: str
    started_at: str
    finished_at: str | None = None
    total_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    spans: list[Span] = Field(default_factory=list)


class Tracer:
    def __init__(self, *, store_limit: int = 500) -> None:
        self._store: dict[str, Trace] = {}
        self._limit = store_limit
        self._lock = Lock()

    def new_trace(self, name: str, **metadata: Any) -> Trace:
        return Trace(
            trace_id=uuid.uuid4().hex,
            name=name,
            started_at=_iso(),
            metadata=redact_value(metadata),
        )

    def finish(self, trace: Trace) -> Trace:
        trace.finished_at = _iso()
        starts = [s.start_ms for s in trace.spans]
        ends = [s.end_ms for s in trace.spans if s.end_ms is not None]
        if starts and ends:
            trace.total_ms = round(max(ends) - min(starts), 2)
        with self._lock:
            self._store[trace.trace_id] = trace
            if len(self._store) > self._limit:
                for key in list(self._store)[: len(self._store) - self._limit]:
                    self._store.pop(key, None)
        return trace

    @contextmanager
    def span(self, trace: Trace, name: str, **attributes: Any) -> Iterator[Span]:
        span = Span(
            span_id=uuid.uuid4().hex[:12],
            name=name,
            start_ms=_now_ms(),
            attributes=redact_value(dict(attributes)),
        )
        trace.spans.append(span)
        try:
            yield span
        except Exception as exc:  # noqa: BLE001 - record any failure
            span.status = "error"
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.end_ms = _now_ms()
            span.duration_ms = round(span.end_ms - span.start_ms, 2)

    def get(self, trace_id: str) -> Trace | None:
        with self._lock:
            return self._store.get(trace_id)

    def list(self, limit: int = 20) -> list[Trace]:
        with self._lock:
            items = list(self._store.values())
        return items[-limit:][::-1]


tracer = Tracer(store_limit=settings.trace_store_limit)
