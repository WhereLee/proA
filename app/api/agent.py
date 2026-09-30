"""Agent API: tool loop, human confirmation, run inspection, audit."""
from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agent.audit import audit
from app.agent.context import ContextManager
from app.agent.loop import AgentRunner
from app.agent.session import sessions
from app.agent.state import runs
from app.agent.statemachine import TaskState
from app.api.sse import sse_event
from app.config import settings
from app.domains import get_domain
from app.llm import get_llm
from app.memory import MemoryManager
from app.obs.ratelimit import ConcurrencyGate

router = APIRouter()

# Bounds concurrent agent runs (checklist P0: 限流与并发闸门).
gate = ConcurrencyGate(settings.agent.max_concurrency)


class AgentRequest(BaseModel):
    message: str
    session_id: str | None = None
    role: str = "agent"
    reset: bool = False


class ConfirmRequest(BaseModel):
    run_id: str
    approved: bool


class RunIdRequest(BaseModel):
    run_id: str


def _build_runner() -> AgentRunner:
    domain = get_domain("tickets")
    return AgentRunner(
        get_llm(),
        domain.build_registry(),  # includes search_knowledge (RAG)
        max_steps=settings.agent.max_steps,
        max_repeated_tool_calls=settings.agent.max_repeated_tool_calls,
        tool_timeout=settings.agent.tool_timeout,
        tool_attempts=settings.agent.tool_attempts,
        tool_backoff=settings.agent.tool_backoff,
        max_cost_usd=settings.agent.max_cost_usd,
        price_prompt_per_1k=settings.agent.price_prompt_per_1k,
        price_completion_per_1k=settings.agent.price_completion_per_1k,
        system_prompt_template=domain.system_prompt_template,
        context_manager=ContextManager(max_tokens=settings.agent.context_max_tokens),
        memory=MemoryManager(),
        memory_domain="tickets",
    )


def _prepare_session(req: AgentRequest):
    session = sessions.get_or_create(req.session_id)
    if req.reset:
        sessions.reset(session.session_id)
        session = sessions.get_or_create(session.session_id)
    return session


async def _consume(agen: AsyncIterator[dict]) -> dict | None:
    last: dict | None = None
    async for event in agen:
        if event["type"] == "run":
            last = event
    return last


def _record_if_completed(run_event: dict | None) -> None:
    if not run_event or run_event.get("status") != TaskState.COMPLETED:
        return
    if run_event.get("answer") and run_event.get("session_id"):
        sessions.record_turn(
            run_event["session_id"], run_event.get("input", ""), run_event["answer"]
        )


@router.post("/agent/run")
async def agent_run(req: AgentRequest) -> dict:
    session = _prepare_session(req)
    runner = _build_runner()
    async with gate.acquire():
        run_event = await _consume(
            runner.start(
                req.message,
                history=sessions.history(session.session_id),
                session_id=session.session_id,
                role=req.role,
            )
        )
    _record_if_completed(run_event)
    return run_event or {"error": "no result"}


@router.post("/agent/run/stream")
async def agent_run_stream(req: AgentRequest) -> StreamingResponse:
    session = _prepare_session(req)
    runner = _build_runner()
    sid = session.session_id
    history = sessions.history(sid)
    role = req.role
    message = req.message

    async def event_gen() -> AsyncIterator[str]:
        async for event in runner.start(
            message, history=history, session_id=sid, role=role
        ):
            if event["type"] == "step":
                yield sse_event("step", event)
            elif event["type"] == "confirmation_required":
                yield sse_event("confirmation_required", event)
            elif event["type"] == "run":
                _record_if_completed(event)
                yield sse_event(
                    "final",
                    {
                        "run_id": event.get("run_id"),
                        "answer": event.get("answer"),
                        "status": event.get("status"),
                        "stopped_reason": event.get("stopped_reason"),
                        "session_id": sid,
                    },
                )
                yield sse_event(
                    "done",
                    {"trace_id": event.get("trace_id"), "run_id": event.get("run_id")},
                )

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/agent/confirm")
async def agent_confirm(req: ConfirmRequest) -> dict:
    runner = _build_runner()
    async with gate.acquire():
        run_event = await _consume(runner.resume(req.run_id, approved=req.approved))
    _record_if_completed(run_event)
    return run_event or {"error": "no result"}


@router.post("/agent/cancel")
async def agent_cancel(req: RunIdRequest) -> dict:
    runner = _build_runner()
    state = await runner.cancel(req.run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="run not found")
    return {"run_id": req.run_id, "status": state}


@router.get("/agent/runs")
def list_runs(limit: int = 50) -> list[dict]:
    return [ctx.to_run().to_dict() for ctx in runs.list()[:limit]]


@router.get("/agent/run/{run_id}")
def get_run(run_id: str) -> dict:
    ctx = runs.get(run_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail="run not found")
    return ctx.to_run().to_dict()


@router.get("/audit/confirmations")
def list_confirmations(limit: int = 50) -> list[dict]:
    return audit.list(limit)
