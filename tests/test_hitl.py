import asyncio

from app.agent.loop import AgentRunner
from app.agent.state import RunStore
from app.llm.scripted import ScriptedLLM
from mocks import data

_REFUND_THEN_FINAL = {
    "帮我退款 O1001": [
        '{"action":"tool","tool":"refund","args":{"order_id":"O1001"}}',
        '{"action":"final","answer":"退款已处理"}',
    ]
}


def _runner(registry, script):
    return AgentRunner(ScriptedLLM(script), registry, run_store=RunStore())


def test_write_suspends_until_confirmed(full_registry):
    runner = _runner(full_registry, _REFUND_THEN_FINAL)
    run = asyncio.run(runner.run("帮我退款 O1001"))
    assert run.status == "awaiting_confirmation"
    assert run.stopped_reason == "awaiting_confirmation"
    assert data.ORDERS["O1001"]["refunded"] is False  # not executed yet
    assert run.run_id


def test_approve_executes_the_write(full_registry):
    runner = _runner(full_registry, _REFUND_THEN_FINAL)
    run = asyncio.run(runner.run("帮我退款 O1001"))

    async def resume():
        final = None
        async for event in runner.resume(run.run_id, approved=True):
            if event["type"] == "run":
                final = event
        return final

    final = asyncio.run(resume())
    assert final["status"] == "completed"
    assert final["answer"] == "退款已处理"
    assert data.ORDERS["O1001"]["refunded"] is True


def test_reject_blocks_the_write(full_registry):
    runner = _runner(full_registry, _REFUND_THEN_FINAL)
    run = asyncio.run(runner.run("帮我退款 O1001"))

    async def resume():
        final = None
        async for event in runner.resume(run.run_id, approved=False):
            if event["type"] == "run":
                final = event
        return final

    final = asyncio.run(resume())
    assert final["status"] == "completed"
    assert data.ORDERS["O1001"]["refunded"] is False
