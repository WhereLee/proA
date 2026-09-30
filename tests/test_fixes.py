import asyncio
import json

import httpx

from app.agent.loop import AgentRunner
from app.agent.state import RunStore
from app.llm.base import ChatMessage
from app.llm.provider import OpenAICompatibleLLM
from app.llm.scripted import ScriptedLLM
from app.main import create_app
from app.persistence import Database
from app.tools.registry import ToolRegistry


# --- P0-1: API path is wired to the domain registry (RAG tool) ------------


def test_agent_run_can_use_knowledge_tool(monkeypatch):
    from app.api import agent as agent_api

    script = {
        "退款规则是什么？": [
            '{"action":"tool","tool":"search_knowledge","args":{"query":"退款规则","k":2}}',
            '{"action":"final","answer":"退款规则见知识库"}',
        ]
    }
    monkeypatch.setattr(agent_api, "get_llm", lambda: ScriptedLLM(script))
    app = create_app()

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            resp = await c.post("/agent/run", json={"message": "退款规则是什么？"})
            return resp.json()

    body = asyncio.run(scenario())
    assert body["stopped_reason"] == "final"
    assert any(
        s["action"] == "tool" and s["tool"] == "search_knowledge" and not s["error"]
        for s in body["steps"]
    )


# --- P0-2: persisted run payload is redacted ------------------------------


def test_run_payload_is_redacted(tmp_path):
    store = RunStore(Database(tmp_path / "runs.db"))
    secret = "sk-abcdef1234567890"
    llm = ScriptedLLM({f"我的密钥是 {secret}": ['{"action":"final","answer":"ok"}']})
    runner = AgentRunner(llm, ToolRegistry(), run_store=store)
    run = asyncio.run(runner.run(f"我的密钥是 {secret}"))

    record = Database(tmp_path / "runs.db").get_run(run.run_id)
    payload = json.dumps(record["payload"], ensure_ascii=False)
    assert secret not in payload
    assert "REDACTED" in payload


# --- P1-3: temperature default is applied ---------------------------------


def test_provider_default_temperature_and_override():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "model": "m",
            },
        )

    provider = OpenAICompatibleLLM(
        api_key="k",
        base_url="https://x/v1",
        default_temperature=0.2,
        transport=httpx.MockTransport(handler),
    )
    asyncio.run(provider.complete([ChatMessage("user", "hi")]))
    assert captured["body"]["temperature"] == 0.2
    asyncio.run(provider.complete([ChatMessage("user", "hi")], temperature=0.0))
    assert captured["body"]["temperature"] == 0.0


# --- P1-4: streaming usage is captured ------------------------------------


def test_provider_stream_usage():
    sse = (
        'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        'data: {"choices":[{"delta":{}}],"usage":{"prompt_tokens":3,'
        '"completion_tokens":1,"total_tokens":4}}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, content=sse.encode(), headers={"content-type": "text/event-stream"}
        )

    captured = {}
    provider = OpenAICompatibleLLM(
        api_key="k", base_url="https://x/v1", transport=httpx.MockTransport(handler)
    )
    usage: dict = {}

    async def collect():
        return [
            chunk
            async for chunk in provider.stream([ChatMessage("user", "hi")], usage_out=usage)
        ]

    tokens = asyncio.run(collect())
    assert "".join(tokens) == "hi"
    assert usage.get("total_tokens") == 4
    assert captured["body"]["stream_options"]["include_usage"] is True


# --- P1-5: confirmation on the last step still allows a final turn --------


def test_confirmation_on_last_step_still_finalizes(full_registry):
    script = {
        "帮我退款 O1001": [
            '{"action":"tool","tool":"refund","args":{"order_id":"O1001"}}',
            '{"action":"final","answer":"已退款"}',
        ]
    }
    runner = AgentRunner(ScriptedLLM(script), full_registry, max_steps=1, run_store=RunStore())
    run = asyncio.run(runner.run("帮我退款 O1001"))
    assert run.status == "awaiting_confirmation"

    async def resume():
        final = None
        async for event in runner.resume(run.run_id, approved=True):
            if event["type"] == "run":
                final = event
        return final

    final = asyncio.run(resume())
    assert final["status"] == "completed"
    assert final["answer"] == "已退款"


# --- P1-6: run / session stores are bounded -------------------------------


def test_run_store_eviction(tmp_path):
    db = Database(tmp_path / "runs.db")
    store = RunStore(db, cap=2)
    for i in range(5):
        runner = AgentRunner(
            ScriptedLLM({f"q{i}": ['{"action":"final","answer":"ok"}']}),
            ToolRegistry(),
            run_store=store,
        )
        asyncio.run(runner.run(f"q{i}"))
    assert len(db.list_runs(100)) == 2


def test_session_store_eviction(tmp_path):
    from app.agent.session import SessionStore

    db = Database(tmp_path / "sessions.db")
    store = SessionStore(db, cap=2)
    for i in range(5):
        store.record_turn(f"s{i}", "hi", "yo")
    assert db.query("SELECT COUNT(*) AS c FROM sessions")[0]["c"] == 2


# --- P2-8: token estimate is CJK-aware ------------------------------------


def test_estimate_tokens_is_cjk_aware():
    from app.agent.context import estimate_tokens

    assert estimate_tokens([ChatMessage("user", "退款规则")]) == 4
