import asyncio

from app.agent.context import ContextManager, estimate_tokens, local_summarize
from app.agent.loop import AgentRunner
from app.agent.state import RunStore
from app.llm.base import ChatMessage
from app.llm.scripted import ScriptedLLM


def test_no_compression_under_budget():
    messages = [ChatMessage("system", "sys"), ChatMessage("user", "hi")]
    manager = ContextManager(max_tokens=1000)
    assert asyncio.run(manager.compress(messages)) is messages


def test_local_summarize_extracts_key_facts():
    summary = local_summarize(
        [ChatMessage("user", '{"order_id": "O1009", "status": "paid"}')]
    )
    assert "order_id=O1009" in summary
    assert "status=paid" in summary


def test_compression_preserves_key_facts():
    messages = [ChatMessage("system", "sys")]
    messages.append(ChatMessage("assistant", '{"action":"tool"}'))
    messages.append(
        ChatMessage(
            "user",
            '工具 get_order 返回：<tool_result>{"order_id": "O1001", '
            '"status": "delivered", "amount": 199.0}</tool_result>',
        )
    )
    for i in range(40):
        messages.append(ChatMessage("user", f"填充消息 {i} " + "x" * 60))

    before = estimate_tokens(messages)
    manager = ContextManager(max_tokens=50, keep_recent=2)
    compressed = asyncio.run(manager.compress(messages))
    after = estimate_tokens(compressed)

    assert after < before
    assert compressed[0].role == "system"
    joined = " ".join(m.content for m in compressed)
    assert "O1001" in joined
    assert "delivered" in joined


def test_runner_completes_with_tiny_budget(readonly_registry):
    script = {
        "q": [
            '{"action":"tool","tool":"get_order","args":{"order_id":"O1001"}}',
            '{"action":"final","answer":"ok"}',
        ]
    }
    manager = ContextManager(max_tokens=1, keep_recent=2)
    runner = AgentRunner(
        ScriptedLLM(script), readonly_registry, run_store=RunStore(), context_manager=manager
    )
    run = asyncio.run(runner.run("q"))
    assert run.stopped_reason == "final"
