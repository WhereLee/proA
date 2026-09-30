import asyncio

import httpx

from app.agent.loop import AgentRunner
from app.agent.state import RunStore, runs
from app.api import research as research_api
from app.domains import get_domain
from app.domains import research_corpus
from app.eval.runner import evaluate, load_dataset
from app.eval.oracle import build_oracle_llm
from app.llm.scripted import ScriptedLLM
from app.main import create_app

RESEARCH = "eval/datasets/research_v1.jsonl"


def _registry():
    return get_domain("research").build_registry()


def test_search_returns_relevant_docs():
    results = research_corpus.search("企业客服 大模型", 3)
    assert results
    assert any("客服" in r["title"] for r in results)
    assert all(r["url"].startswith("http") for r in results)


def test_fetch_existing_and_missing():
    registry = _registry()
    params = registry.validate_args(
        "fetch", {"url": "https://example.com/rag-knowledge-base"}
    )
    result = asyncio.run(registry.get("fetch").handler(params))
    assert "RAG" in result["text"]

    from app.tools.registry import ToolExecutionError

    bad = registry.validate_args("fetch", {"url": "https://example.com/nope"})
    try:
        asyncio.run(registry.get("fetch").handler(bad))
        raise AssertionError("expected ToolExecutionError")
    except ToolExecutionError as exc:
        assert exc.status_code == 404


def test_oracle_passes_research_dataset():
    tasks = load_dataset(RESEARCH)
    report = asyncio.run(
        evaluate(RESEARCH, llm=build_oracle_llm(tasks), registry=_registry())
    )
    assert report.pass_rate == 1.0, report.failure_summary


def test_resume_run_continues_running_context():
    script = {"t": ['{"action":"final","answer":"done http://x"}' ]}
    runner = AgentRunner(ScriptedLLM(script), _registry(), run_store=runs)
    ctx = runner.create_run("t")
    # simulate a process restart: state left as running after one turn
    ctx.state = "running"
    ctx.turns = 1
    runs.put(ctx)

    async def go():
        final = None
        async for event in runner.resume_run(ctx.run_id):
            if event["type"] == "run":
                final = event
        return final

    final = asyncio.run(go())
    assert final["status"] == "completed"
    assert "done" in final["answer"]


def test_resume_terminal_run_is_rejected():
    runner = AgentRunner(
        ScriptedLLM({"t": ['{"action":"final","answer":"x"}']}),
        _registry(),
        run_store=runs,
    )
    ctx = runner.create_run("t")
    ctx.state = "completed"
    runs.put(ctx)

    async def go():
        return [event async for event in runner.resume_run(ctx.run_id)]

    events = asyncio.run(go())
    assert events and events[0]["type"] == "error"


def test_research_queue_submit_and_status(monkeypatch):
    script = {
        "调研X": [
            '{"action":"tool","tool":"search","args":{"query":"X","k":2}}',
            '{"action":"tool","tool":"fetch","args":{"url":"https://example.com/rag-knowledge-base"}}',
            '{"action":"final","answer":"报告见 https://example.com/rag-knowledge-base 内容摘要"}',
        ]
    }
    llm = ScriptedLLM(script)

    def fake_build():
        domain = get_domain("research")
        return AgentRunner(
            llm, domain.build_registry(), system_prompt_template=domain.system_prompt_template
        )

    monkeypatch.setattr(research_api, "_build_runner", fake_build)
    app = create_app()

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            resp = await c.post("/research/submit", json={"topic": "调研X"})
            assert resp.status_code == 200
            run_id = resp.json()["run_id"]
            data = {}
            for _ in range(100):
                status = await c.get(f"/research/{run_id}")
                data = status.json()
                if data["status"] in ("completed", "failed"):
                    break
                await asyncio.sleep(0.02)
            return data

    result = asyncio.run(scenario())
    assert result["status"] == "completed"
    assert "http" in result["answer"]
