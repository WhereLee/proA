"""Research API: background queue for long-running research runs.

submit -> returns run_id immediately; status -> poll; cancel; resume (continue a
run from its persisted checkpoint after an interruption).
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.agent.context import ContextManager
from app.agent.loop import AgentRunner
from app.agent.state import runs
from app.config import settings
from app.domains import get_domain
from app.llm import get_llm
from app.memory import MemoryManager

router = APIRouter()
_tasks: dict[str, asyncio.Task] = {}


def _build_runner() -> AgentRunner:
    domain = get_domain("research")
    return AgentRunner(
        get_llm(),
        domain.build_registry(),
        max_steps=domain.max_steps or settings.agent.max_steps,
        tool_timeout=settings.agent.tool_timeout,
        system_prompt_template=domain.system_prompt_template,
        context_manager=ContextManager(max_tokens=settings.agent.context_max_tokens),
        memory=MemoryManager(),
        memory_domain="research",
        run_store=runs,
    )


class SubmitRequest(BaseModel):
    topic: str
    session_id: str | None = None


@router.post("/research/submit")
async def submit(req: SubmitRequest) -> dict:
    runner = _build_runner()
    ctx = runner.create_run(req.topic, session_id=req.session_id)
    _tasks[ctx.run_id] = asyncio.create_task(runner.execute(ctx))
    return {"run_id": ctx.run_id, "status": ctx.state}


@router.get("/research/{run_id}")
def status(run_id: str) -> dict:
    ctx = runs.get(run_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail="run not found")
    return ctx.to_run().to_dict()


@router.post("/research/{run_id}/cancel")
async def cancel(run_id: str) -> dict:
    task = _tasks.pop(run_id, None)
    if task is not None:
        task.cancel()
    state = await _build_runner().cancel(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="run not found")
    return {"run_id": run_id, "status": state}


@router.post("/research/{run_id}/resume")
async def resume(run_id: str) -> dict:
    runner = _build_runner()
    if runs.get(run_id) is None:
        raise HTTPException(status_code=404, detail="run not found")

    async def _worker() -> None:
        async for _ in runner.resume_run(run_id):
            pass

    _tasks[run_id] = asyncio.create_task(_worker())
    return {"run_id": run_id, "status": "resuming"}
