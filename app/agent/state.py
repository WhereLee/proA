"""Run state + persistent checkpoint store (checklist P0: checkpoint / 断点续跑).

RunContext is fully serializable; RunStore writes a checkpoint to SQLite (when a
Database is provided) after every transition, so a run can be resumed after a
process restart.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.config import settings
from app.llm.base import ChatMessage
from app.obs.tracer import Trace
from app.persistence import Database, get_default_db
from app.security.redact import redact, redact_value

from .statemachine import TaskState


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AgentStep:
    step: int
    action: str  # tool | final | invalid_output | confirm_required
    tool: str | None = None
    args: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    text: str | None = None
    latency_ms: float = 0.0
    attempts: int = 1
    injection_flags: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentRun:
    answer: str | None
    steps: list[AgentStep]
    stopped_reason: str
    trace_id: str
    session_id: str | None = None
    run_id: str | None = None
    status: str = TaskState.PENDING
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "steps": [s.to_dict() for s in self.steps],
            "stopped_reason": self.stopped_reason,
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "status": self.status,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class PendingTool:
    tool: str
    args: dict[str, Any]
    idempotency_key: str
    approved: bool | None = None  # None = awaiting decision


@dataclass
class RunContext:
    run_id: str
    user_input: str
    session_id: str | None
    role: str
    trace: Trace
    messages: list[ChatMessage]
    steps: list[AgentStep] = field(default_factory=list)
    call_counts: dict[str, int] = field(default_factory=dict)
    turns: int = 0
    pending: PendingTool | None = None
    state: str = TaskState.PENDING
    stopped_reason: str = "max_steps"
    answer: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    def to_run(self) -> AgentRun:
        return AgentRun(
            answer=self.answer,
            steps=self.steps,
            stopped_reason=self.stopped_reason,
            trace_id=self.trace.trace_id,
            session_id=self.session_id,
            run_id=self.run_id,
            status=self.state,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            cost_usd=self.cost_usd,
        )


def serialize_context(ctx: RunContext) -> dict[str, Any]:
    return redact_value({
        "run_id": ctx.run_id,
        "user_input": ctx.user_input,
        "session_id": ctx.session_id,
        "role": ctx.role,
        "state": ctx.state,
        "stopped_reason": ctx.stopped_reason,
        "answer": ctx.answer,
        "turns": ctx.turns,
        "call_counts": ctx.call_counts,
        # redact secrets before persisting prompts/tool output
        "messages": [{"role": m.role, "content": redact(m.content)} for m in ctx.messages],
        "steps": [asdict(s) for s in ctx.steps],
        "pending": asdict(ctx.pending) if ctx.pending else None,
        "trace": ctx.trace.model_dump(),
        "prompt_tokens": ctx.prompt_tokens,
        "completion_tokens": ctx.completion_tokens,
        "cost_usd": ctx.cost_usd,
    })


def deserialize_context(data: dict[str, Any]) -> RunContext:
    return RunContext(
        run_id=data["run_id"],
        user_input=data["user_input"],
        session_id=data.get("session_id"),
        role=data.get("role", "agent"),
        trace=Trace.model_validate(data["trace"]),
        messages=[ChatMessage(m["role"], m["content"]) for m in data["messages"]],
        steps=[AgentStep(**s) for s in data["steps"]],
        call_counts=dict(data.get("call_counts", {})),
        turns=data.get("turns", 0),
        pending=PendingTool(**data["pending"]) if data.get("pending") else None,
        state=data.get("state", TaskState.PENDING),
        stopped_reason=data.get("stopped_reason", "max_steps"),
        answer=data.get("answer"),
        prompt_tokens=data.get("prompt_tokens", 0),
        completion_tokens=data.get("completion_tokens", 0),
        cost_usd=data.get("cost_usd", 0.0),
    )


class RunStore:
    def __init__(self, db: Database | None = None, *, cap: int = 500) -> None:
        self._cache: dict[str, RunContext] = {}
        self._db = db
        self._cap = cap

    def put(self, ctx: RunContext) -> None:
        self._cache[ctx.run_id] = ctx
        if len(self._cache) > self._cap:
            for key in list(self._cache)[: len(self._cache) - self._cap]:
                self._cache.pop(key, None)
        if self._db is not None:
            self._db.upsert_run(
                ctx.run_id, ctx.state, serialize_context(ctx), _iso()
            )
            self._db.evict_runs(self._cap)

    def get(self, run_id: str) -> RunContext | None:
        if run_id in self._cache:
            return self._cache[run_id]
        if self._db is not None:
            record = self._db.get_run(run_id)
            if record is not None:
                ctx = deserialize_context(record["payload"])
                self._cache[run_id] = ctx
                return ctx
        return None

    def list(self) -> list[RunContext]:
        if self._db is not None:
            return [deserialize_context(r["payload"]) for r in self._db.list_runs()]
        return list(self._cache.values())


runs = RunStore(get_default_db(), cap=settings.run_store_limit)
