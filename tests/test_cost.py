import asyncio

from app.agent.loop import AgentRunner
from app.agent.state import RunStore
from app.agent.statemachine import TaskState
from app.llm.base import LLMResult, Usage


class CostlyLLM:
    name = "costly"

    async def complete(self, messages, *, model=None, temperature=None):
        return LLMResult(
            text='{"action":"final","answer":"ok"}',
            model="m",
            provider=self.name,
            latency_ms=0.0,
            usage=Usage(prompt_tokens=1000, completion_tokens=1000, total_tokens=2000),
        )

    async def stream(self, messages, *, model=None, temperature=None):
        yield "ok"


def test_run_cost_is_accumulated(full_registry):
    runner = AgentRunner(CostlyLLM(), full_registry, run_store=RunStore())
    run = asyncio.run(runner.run("q"))
    # 1k prompt * 0.005/1k + 1k completion * 0.015/1k = 0.02
    assert run.prompt_tokens == 1000
    assert run.completion_tokens == 1000
    assert round(run.cost_usd, 6) == 0.02


def test_budget_breaker_trips(full_registry):
    runner = AgentRunner(
        CostlyLLM(),
        full_registry,
        run_store=RunStore(),
        max_cost_usd=0.001,  # below one call's cost
        price_prompt_per_1k=0.005,
        price_completion_per_1k=0.015,
    )
    events = []

    async def scenario():
        async for event in runner.start("q"):
            events.append(event)

    asyncio.run(scenario())
    assert any(e["type"] == "budget_exceeded" for e in events)

    run = next(e for e in events if e["type"] == "run")
    assert run["status"] == TaskState.FAILED
    assert run["stopped_reason"] == "budget_exceeded"
