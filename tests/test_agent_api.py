import asyncio

import httpx

from app.agent.loop import AgentRunner
from app.api import agent as agent_api
from app.llm.scripted import ScriptedLLM
from app.main import create_app
from mocks import data


def _client(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


def _patch_runner(monkeypatch, registry, script, **kwargs):
    def fake_build():
        return AgentRunner(ScriptedLLM(script), registry, **kwargs)

    monkeypatch.setattr(agent_api, "_build_runner", fake_build)


def test_agent_run_endpoint(monkeypatch, readonly_registry):
    script = {
        "帮我查一下订单 O1001": [
            '{"action":"tool","tool":"get_order","args":{"order_id":"O1001"}}',
            '{"action":"final","answer":"订单 O1001 已签收"}',
        ]
    }
    _patch_runner(monkeypatch, readonly_registry, script)
    app = create_app()

    async def scenario():
        async with _client(app) as c:
            resp = await c.post("/agent/run", json={"message": "帮我查一下订单 O1001"})
            assert resp.status_code == 200
            body = resp.json()
            assert body["stopped_reason"] == "final"
            assert body["answer"] == "订单 O1001 已签收"
            assert body["session_id"]
            trace = await c.get(f"/traces/{body['trace_id']}")
            assert trace.status_code == 200
            assert any(s["name"].startswith("llm.step") for s in trace.json()["spans"])

    asyncio.run(scenario())


def test_agent_stream_endpoint(monkeypatch, readonly_registry):
    script = {
        "帮我查一下订单 O1001": [
            '{"action":"tool","tool":"get_order","args":{"order_id":"O1001"}}',
            '{"action":"final","answer":"done"}',
        ]
    }
    _patch_runner(monkeypatch, readonly_registry, script)
    app = create_app()

    async def scenario():
        events = []
        async with _client(app) as c:
            async with c.stream(
                "POST", "/agent/run/stream", json={"message": "帮我查一下订单 O1001"}
            ) as resp:
                assert resp.status_code == 200
                async for line in resp.aiter_lines():
                    if line.startswith("event:"):
                        events.append(line.split(":", 1)[1].strip())
        return events

    events = asyncio.run(scenario())
    assert "step" in events
    assert "final" in events
    assert events[-1] == "done"


def test_hitl_confirm_endpoint(monkeypatch, full_registry):
    llm = ScriptedLLM(
        {
            "帮我退款 O1001": [
                '{"action":"tool","tool":"refund","args":{"order_id":"O1001"}}',
                '{"action":"final","answer":"已退款"}',
            ]
        }
    )

    def fake_build():
        return AgentRunner(llm, full_registry, max_steps=6)

    monkeypatch.setattr(agent_api, "_build_runner", fake_build)
    app = create_app()

    async def scenario():
        async with _client(app) as c:
            started = await c.post("/agent/run", json={"message": "帮我退款 O1001"})
            body = started.json()
            assert body["status"] == "awaiting_confirmation"
            assert data.ORDERS["O1001"]["refunded"] is False

            confirmed = await c.post(
                "/agent/confirm", json={"run_id": body["run_id"], "approved": True}
            )
            result = confirmed.json()
            assert result["status"] == "completed"
            assert result["answer"] == "已退款"

            audit = await c.get("/audit/confirmations")
            assert audit.status_code == 200
            assert any(r["run_id"] == body["run_id"] for r in audit.json())

    asyncio.run(scenario())
    assert data.ORDERS["O1001"]["refunded"] is True
