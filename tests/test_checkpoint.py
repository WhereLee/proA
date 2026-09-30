import asyncio

import httpx

from app.agent.loop import AgentRunner
from app.agent.state import RunStore
from app.agent.statemachine import TaskState
from app.llm.scripted import ScriptedLLM
from app.persistence import Database
from app.tools import MockClient, build_registry
from mocks import data
from mocks.faults import FaultRequest, configure

_REFUND_THEN_FINAL = {
    "帮我退款 O1001": [
        '{"action":"tool","tool":"refund","args":{"order_id":"O1001"}}',
        '{"action":"final","answer":"(unused)"}',
    ]
}


def _registry(mock_app):
    client = MockClient(
        "http://mock", timeout=2.0, transport=httpx.ASGITransport(app=mock_app)
    )
    return build_registry(client, include_writes=True)


def test_checkpoint_persists_state(tmp_path, mock_app):
    db_path = tmp_path / "agent.db"
    store = RunStore(Database(db_path))
    runner = AgentRunner(
        ScriptedLLM(_REFUND_THEN_FINAL), _registry(mock_app), run_store=store
    )
    run = asyncio.run(runner.run("帮我退款 O1001"))
    assert run.status == TaskState.AWAITING_CONFIRMATION

    record = Database(db_path).get_run(run.run_id)
    assert record is not None
    assert record["state"] == TaskState.AWAITING_CONFIRMATION
    assert record["payload"]["pending"]["tool"] == "refund"


def test_resume_after_restart_executes_write_exactly_once(tmp_path, mock_app):
    db_path = tmp_path / "agent.db"
    registry = _registry(mock_app)

    store1 = RunStore(Database(db_path))
    runner1 = AgentRunner(ScriptedLLM(_REFUND_THEN_FINAL), registry, run_store=store1)
    run = asyncio.run(runner1.run("帮我退款 O1001"))
    assert run.status == TaskState.AWAITING_CONFIRMATION
    run_id = run.run_id

    # Simulate a process restart: fresh DB connection, fresh store, fresh LLM.
    # Also inject an ambiguous failure: the refund side effect commits but the
    # response is lost (partial fault).
    configure(FaultRequest(mode="partial", target="order", remaining=1))
    script = {"帮我退款 O1001": ['{"action":"final","answer":"已退款"}']}
    store2 = RunStore(Database(db_path))
    runner2 = AgentRunner(ScriptedLLM(script), registry, run_store=store2, tool_attempts=2)

    assert store2.get(run_id) is not None  # rehydrated from disk

    async def resume():
        final = None
        async for event in runner2.resume(run_id, approved=True):
            if event["type"] == "run":
                final = event
        return final

    final = asyncio.run(resume())
    assert final["status"] == TaskState.COMPLETED
    assert final["answer"] == "已退款"

    refund_step = next(
        s for s in final["steps"] if s["action"] == "tool" and s["tool"] == "refund"
    )
    assert refund_step["attempts"] == 2  # first attempt failed, retried
    assert refund_step["result"].get("idempotent_replay") is True
    assert data.ORDERS["O1001"]["refunded"] is True


def test_read_timeout_is_retried_then_reported(mock_app):
    registry = _registry(mock_app)
    configure(
        FaultRequest(
            mode="timeout", target="order", remaining=5, timeout_seconds=0.02
        )
    )
    script = {
        "查 O1001": [
            '{"action":"tool","tool":"get_order","args":{"order_id":"O1001"}}',
            '{"action":"final","answer":"暂时查不到"}',
        ]
    }
    runner = AgentRunner(
        ScriptedLLM(script),
        registry,
        run_store=RunStore(),
        tool_attempts=2,
        tool_backoff=0.0,
    )
    run = asyncio.run(runner.run("查 O1001"))
    tool_step = next(s for s in run.steps if s.action == "tool")
    assert tool_step.attempts == 2
    assert "504" in (tool_step.error or "")
    assert run.status == TaskState.COMPLETED  # model still finalizes


def test_cancel_run(mock_app):
    registry = _registry(mock_app)
    runner = AgentRunner(
        ScriptedLLM(_REFUND_THEN_FINAL), registry, run_store=RunStore()
    )
    run = asyncio.run(runner.run("帮我退款 O1001"))
    assert run.status == TaskState.AWAITING_CONFIRMATION
    state = asyncio.run(runner.cancel(run.run_id))
    assert state == TaskState.CANCELLED
