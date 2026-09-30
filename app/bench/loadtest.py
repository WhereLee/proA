"""Agent load test (checklist P1: 并发下的吞吐/延迟/成本；P0: 并发闸门).

Measures latency P50/P95/P99 and cost under bounded concurrency, and can wrap
the LLM with a token-bucket limiter to observe model-side throttling.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from app.agent.loop import AgentRunner
from app.agent.state import RunStore
from app.obs.ratelimit import ConcurrencyGate, ThrottledLLM, TokenBucketLimiter


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo), 2)


async def run_loadtest(
    inputs: list[str],
    registry,
    llm,
    *,
    concurrency: int = 4,
    max_steps: int = 6,
    rpm: float = 0.0,
) -> dict[str, Any]:
    gate = ConcurrencyGate(concurrency)
    limiter = TokenBucketLimiter(rpm) if rpm > 0 else None
    active_llm = ThrottledLLM(llm, limiter) if limiter else llm
    runner = AgentRunner(
        active_llm, registry, max_steps=max_steps, run_store=RunStore()
    )

    latencies: list[float] = []
    costs: list[float] = []
    states: dict[str, int] = {}

    async def one(user_input: str) -> None:
        async with gate.acquire():
            start = time.perf_counter()
            run = await runner.run(user_input)
            latencies.append((time.perf_counter() - start) * 1000.0)
            costs.append(run.cost_usd)
            states[run.status] = states.get(run.status, 0) + 1

    wall_start = time.perf_counter()
    await asyncio.gather(*(one(i) for i in inputs))
    wall_ms = (time.perf_counter() - wall_start) * 1000.0

    completed = states.get("completed", 0)
    return {
        "runs": len(inputs),
        "completed": completed,
        "failures": len(inputs) - completed,
        "concurrency": concurrency,
        "max_concurrent": gate.max_active,
        "wall_ms": round(wall_ms, 2),
        "throughput_rps": round(len(inputs) / (wall_ms / 1000.0), 2) if wall_ms else 0.0,
        "latency_ms": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": round(max(latencies), 2) if latencies else 0.0,
        },
        "cost_usd": {
            "total": round(sum(costs), 6),
            "avg": round(sum(costs) / len(costs), 6) if costs else 0.0,
        },
        "throttled_events": limiter.throttled_events if limiter else 0,
        "states": states,
    }
