import asyncio

import httpx
import pytest

pytest.importorskip("langgraph")

from app.agent.loop import AgentRunner  # noqa: E402
from app.agent.state import RunStore  # noqa: E402
from app.eval.graders import grade_task  # noqa: E402
from app.eval.oracle import build_oracle_llm  # noqa: E402
from app.eval.runner import load_dataset  # noqa: E402
from app.harness_langgraph import LangGraphRunner  # noqa: E402
from app.tools import MockClient, build_readonly_registry  # noqa: E402

DATASET = "eval/datasets/readonly_v1.jsonl"


def _registry(mock_app):
    client = MockClient(
        "http://mock", timeout=2.0, transport=httpx.ASGITransport(app=mock_app)
    )
    return build_readonly_registry(client)


def _score(runner, tasks):
    async def run():
        passed = 0
        for task in tasks:
            result = await runner.run(task["input"])
            if grade_task(task, result).passed:
                passed += 1
        return passed

    return asyncio.run(run())


def test_langgraph_harness_matches_self_built(mock_app):
    tasks = load_dataset(DATASET)[:6]
    registry = _registry(mock_app)

    self_passed = _score(
        AgentRunner(build_oracle_llm(tasks), registry, run_store=RunStore()), tasks
    )
    lg_passed = _score(LangGraphRunner(build_oracle_llm(tasks), registry), tasks)

    assert self_passed == len(tasks)
    assert lg_passed == len(tasks)
