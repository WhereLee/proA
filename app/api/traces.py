"""Trace inspection API: any run can be replayed to locate the failing step."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.obs import tracer

router = APIRouter()


@router.get("/traces")
def list_traces(limit: int = 20) -> list[dict]:
    return [t.model_dump() for t in tracer.list(limit)]


@router.get("/traces/{trace_id}")
def get_trace(trace_id: str) -> dict:
    trace = tracer.get(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="trace not found")
    return trace.model_dump()
