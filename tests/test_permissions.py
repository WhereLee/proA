import asyncio

from app.agent.loop import AgentRunner
from app.agent.state import RunStore
from app.llm.scripted import ScriptedLLM
from mocks import data


def test_customer_role_cannot_call_write_tool(full_registry):
    script = {
        "帮我退款 O1001": [
            '{"action":"tool","tool":"refund","args":{"order_id":"O1001"}}',
            '{"action":"final","answer":"无法完成"}',
        ]
    }
    runner = AgentRunner(ScriptedLLM(script), full_registry, run_store=RunStore())
    run = asyncio.run(runner.run("帮我退款 O1001", role="customer"))

    assert any(s.error and "permission denied" in s.error for s in run.steps)
    assert run.status == "completed"
    assert data.ORDERS["O1001"]["refunded"] is False
