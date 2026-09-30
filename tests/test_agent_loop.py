import asyncio

from app.agent.loop import AgentRunner
from app.llm.scripted import ScriptedLLM


def _run(script, registry, **kwargs):
    runner = AgentRunner(ScriptedLLM(script), registry, **kwargs)
    return asyncio.run(runner.run(next(iter(script))))


def test_single_tool_then_final(readonly_registry):
    script = {
        "帮我查一下订单 O1001": [
            '{"action":"tool","tool":"get_order","args":{"order_id":"O1001"}}',
            '{"action":"final","answer":"订单 O1001 已签收"}',
        ]
    }
    run = _run(script, readonly_registry)
    assert run.stopped_reason == "final"
    assert run.answer == "订单 O1001 已签收"
    assert run.steps[0].action == "tool"
    assert run.steps[0].result["status"] == "delivered"


def test_unknown_tool_recovers(readonly_registry):
    script = {
        "q": [
            '{"action":"tool","tool":"nope","args":{}}',
            '{"action":"final","answer":"done"}',
        ]
    }
    run = _run(script, readonly_registry)
    assert "unknown tool" in run.steps[0].error
    assert run.stopped_reason == "final"


def test_invalid_args_recovers(readonly_registry):
    script = {
        "q": [
            '{"action":"tool","tool":"get_order","args":{"wrong":"x"}}',
            '{"action":"final","answer":"done"}',
        ]
    }
    run = _run(script, readonly_registry)
    assert "invalid args" in run.steps[0].error
    assert run.stopped_reason == "final"


def test_invalid_output_recovers(readonly_registry):
    script = {
        "q": [
            "抱歉，先让我想想。",
            '{"action":"final","answer":"ok"}',
        ]
    }
    run = _run(script, readonly_registry)
    assert run.steps[0].action == "invalid_output"
    assert run.stopped_reason == "final"


def test_loop_detected(readonly_registry):
    call = '{"action":"tool","tool":"get_order","args":{"order_id":"O1001"}}'
    script = {"q": [call, call, call, call]}
    run = _run(script, readonly_registry, max_repeated_tool_calls=2)
    assert run.stopped_reason == "loop_detected"


def test_max_steps(readonly_registry):
    script = {
        "q": [
            '{"action":"tool","tool":"get_order","args":{"order_id":"O1001"}}',
            '{"action":"tool","tool":"get_order","args":{"order_id":"O1002"}}',
        ]
    }
    run = _run(script, readonly_registry, max_steps=2)
    assert run.stopped_reason == "max_steps"
    assert len(run.steps) == 2
