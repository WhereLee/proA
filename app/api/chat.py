"""Chat API: SSE streaming endpoint + a non-streaming convenience endpoint.

DEPRECATED: superseded by `/agent/*` (which has tools, guardrails and cost
tracking). Kept only for the raw token-stream demo, so its streaming path also
captures usage for cost accounting.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.api.sse import sse_event
from app.config import settings
from app.llm import ChatMessage, get_llm
from app.obs import tracer

router = APIRouter()

SYSTEM_PROMPT = (
    "你是智能工单处理助手。当前处于阶段 0，仅用于验证链路，请用中文简洁回答用户。"
)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    temperature: float | None = None


_sse = sse_event


def _messages(req: ChatRequest) -> list[ChatMessage]:
    return [ChatMessage("system", SYSTEM_PROMPT), ChatMessage("user", req.message)]


@router.post("/chat")
async def chat(req: ChatRequest) -> dict:
    llm = get_llm()
    trace = tracer.new_trace("chat", session_id=req.session_id, message=req.message)
    try:
        with tracer.span(
            trace, "llm.complete", provider=llm.name, models=settings.llm.models
        ):
            result = await llm.complete(_messages(req), temperature=req.temperature)
    finally:
        tracer.finish(trace)
    return {
        "reply": result.text,
        "model": result.model,
        "attempts": result.attempts,
        "usage": result.usage.to_dict(),
        "trace_id": trace.trace_id,
    }


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    llm = get_llm()
    trace = tracer.new_trace(
        "chat.stream", session_id=req.session_id, message=req.message
    )

    async def event_gen() -> AsyncIterator[str]:
        collected: list[str] = []
        usage: dict = {}
        error: Exception | None = None
        try:
            try:
                with tracer.span(
                    trace, "llm.stream", provider=llm.name, models=settings.llm.models
                ):
                    async for chunk in llm.stream(
                        _messages(req), temperature=req.temperature, usage_out=usage
                    ):
                        collected.append(chunk)
                        yield _sse("token", {"text": chunk})
            except Exception as exc:  # noqa: BLE001 - surfaced to the client
                error = exc
        finally:
            tracer.finish(trace)

        if error is not None:
            yield _sse("error", {"message": f"{type(error).__name__}: {error}"})
        yield _sse(
            "done",
            {
                "trace_id": trace.trace_id,
                "chunks": len(collected),
                "chars": len("".join(collected)),
                "total_ms": trace.total_ms,
                "usage": usage,
            },
        )

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
