"""Compare the self-built harness vs the LangGraph harness on the same dataset.

    python -m app.harness_langgraph --limit 20

Both harnesses run the deterministic oracle LLM, so this compares the *control
flow implementations*, not model quality. Requires the optional langgraph dep.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

import httpx

from app.agent.loop import AgentRunner
from app.agent.state import RunStore
from app.eval.graders import grade_task
from app.eval.oracle import build_oracle_llm
from app.eval.runner import load_dataset
from app.tools import MockClient, build_readonly_registry

from .runner import LangGraphRunner


def _registry():
    from mocks.app import app as mock_app

    client = MockClient(
        "http://mock", timeout=5.0, transport=httpx.ASGITransport(app=mock_app)
    )
    return build_readonly_registry(client)


async def _score(runner, tasks: list[dict]) -> tuple[int, float]:
    passed = 0
    start = time.perf_counter()
    for task in tasks:
        run = await runner.run(task["input"])
        if grade_task(task, run).passed:
            passed += 1
    return passed, (time.perf_counter() - start) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-built vs LangGraph harness")
    parser.add_argument("--dataset", default="eval/datasets/readonly_v1.jsonl")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    tasks = load_dataset(args.dataset)
    if args.limit:
        tasks = tasks[: args.limit]

    registry = _registry()
    self_passed, self_ms = asyncio.run(
        _score(AgentRunner(build_oracle_llm(tasks), registry, run_store=RunStore()), tasks)
    )
    lg_passed, lg_ms = asyncio.run(
        _score(LangGraphRunner(build_oracle_llm(tasks), registry), tasks)
    )

    report = {
        "tasks": len(tasks),
        "self_built": {
            "passed": self_passed,
            "pass_rate": round(self_passed / len(tasks), 4),
            "elapsed_ms": round(self_ms, 2),
        },
        "langgraph": {
            "passed": lg_passed,
            "pass_rate": round(lg_passed / len(tasks), 4),
            "elapsed_ms": round(lg_ms, 2),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
