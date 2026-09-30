import asyncio

import httpx

from app.llm.fake import FakeLLM
from app.llm.resilient import ResilientLLM
from app.llm.runtime import set_llm
from app.main import create_app


def _fake_llm() -> ResilientLLM:
    return ResilientLLM(FakeLLM(), ["fake-model"], max_retries=1, backoff_base=0.0)


def _client(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


def test_health():
    async def scenario():
        app = create_app()
        async with _client(app) as c:
            resp = await c.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

    asyncio.run(scenario())


def test_chat_records_trace():
    async def scenario():
        set_llm(_fake_llm())
        try:
            app = create_app()
            async with _client(app) as c:
                resp = await c.post("/chat", json={"message": "查一下 O1001"})
                assert resp.status_code == 200
                body = resp.json()
                assert body["reply"]
                trace = await c.get(f"/traces/{body['trace_id']}")
                assert trace.status_code == 200
                assert trace.json()["spans"][0]["name"] == "llm.complete"
                return body
        finally:
            set_llm(None)

    body = asyncio.run(scenario())
    assert body["attempts"] >= 1


def test_chat_stream_emits_sse_events():
    async def scenario():
        set_llm(_fake_llm())
        events: list[str] = []
        try:
            app = create_app()
            async with _client(app) as c:
                async with c.stream(
                    "POST", "/chat/stream", json={"message": "你好"}
                ) as resp:
                    assert resp.status_code == 200
                    assert resp.headers["content-type"].startswith("text/event-stream")
                    async for line in resp.aiter_lines():
                        if line.startswith("event:"):
                            events.append(line.split(":", 1)[1].strip())
        finally:
            set_llm(None)
        return events

    events = asyncio.run(scenario())
    assert events.count("token") > 1
    assert events[-1] == "done"
