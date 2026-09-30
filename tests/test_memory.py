import asyncio

import httpx

from app.agent.loop import AgentRunner
from app.agent.state import RunStore
from app.domains import get_domain
from app.llm.base import LLMResult
from app.llm.scripted import ScriptedLLM
from app.memory import BM25, EpisodicMemory, KnowledgeBase, MemoryManager, rank, tokenize
from app.memory.seed import SOP_DOCS
from app.persistence import Database

# ---------------------------------------------------------------- bm25 / rag


def test_tokenize_mixed_language():
    tokens = tokenize("退款 refund O1001")
    assert "refund" in tokens
    assert "退" in tokens and "退款" in tokens


def test_bm25_ranks_relevant_first():
    docs = ["退款规则说明", "改地址规则", "工单处理流程"]
    ranked = rank("退款", docs, 1)
    assert ranked and ranked[0][0] == 0


def test_knowledge_ingest_and_search(tmp_path):
    kb = KnowledgeBase(Database(tmp_path / "m.db"))
    for doc in SOP_DOCS:
        kb.ingest(doc["doc_id"], doc["title"], doc["text"], source="sop")
    assert kb.count()["docs"] == len(SOP_DOCS)
    results = kb.search("退款 期限", k=2)
    assert results
    assert any("退款" in r["title"] for r in results)
    assert all(r["source"] == "sop" for r in results)


# ---------------------------------------------------------------- episodic


def test_episodic_record_recall_and_isolation(tmp_path):
    memory = EpisodicMemory(Database(tmp_path / "m.db"))
    memory.record(
        run_id="r1", domain="tickets", role="agent",
        task="帮我退款 O1001", outcome="success", tools="refund",
        summary="任务「退款 O1001」→ success",
    )
    memory.record(
        run_id="r2", domain="tickets", role="customer",
        task="帮我退款 O1001", outcome="failed", tools="refund",
        summary="权限不足",
    )
    recalled = memory.recall(domain="tickets", role="agent", query="退款 O1001")
    assert len(recalled) == 1
    assert recalled[0]["outcome"] == "success"
    # isolation by role
    assert memory.count(domain="tickets", role="customer") == 1
    assert memory.count(domain="tickets", role="agent") == 1


def test_episodic_eviction(tmp_path):
    memory = EpisodicMemory(Database(tmp_path / "m.db"), cap=3)
    for i in range(6):
        memory.record(
            run_id=f"r{i}", domain="tickets", role="agent",
            task=f"任务{i}", outcome="success", tools="", summary=f"s{i}",
        )
    assert memory.count(domain="tickets", role="agent") == 3


# ---------------------------------------------------------------- manager + runner


class SpyLLM:
    name = "spy"

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls: list[list] = []

    async def complete(self, messages, *, model=None, temperature=None):
        self.calls.append(list(messages))
        return LLMResult(text=self._reply, model="spy", provider="spy", latency_ms=0.0)

    async def stream(self, messages, *, model=None, temperature=None):
        yield self._reply


def test_memory_manager_block_and_record(tmp_path):
    memory = MemoryManager(episodic=EpisodicMemory(Database(tmp_path / "m.db")))
    assert memory.recall_block(domain="tickets", role="agent", query="退款") is None

    from app.agent.state import AgentRun, AgentStep

    run = AgentRun(
        answer="已退款", steps=[AgentStep(1, "tool", tool="refund")],
        stopped_reason="final", trace_id="t", run_id="r", status="completed",
    )
    memory.record_run(run, domain="tickets", role="agent", task="帮我退款 O1001")
    block = memory.recall_block(domain="tickets", role="agent", query="退款 O1001")
    assert block is not None and "历史经验" in block


def test_runner_injects_and_records_memory(tmp_path):
    memory = MemoryManager(episodic=EpisodicMemory(Database(tmp_path / "m.db")))
    memory.record_run(
        _completed_run("r0"), domain="tickets", role="agent", task="退款 O1001 处理经验"
    )
    spy = SpyLLM('{"action":"final","answer":"完成"}')
    runner = AgentRunner(
        spy, _readonly(), run_store=RunStore(), memory=memory, memory_domain="tickets"
    )
    run = asyncio.run(runner.run("退款 O1001"))
    # the recall block was injected into the first prompt
    assert any(
        m.role == "system" and "历史经验" in m.content for m in spy.calls[0]
    )
    # and this run was recorded as a new episode
    assert memory._episodic.count(domain="tickets", role="agent") == 2


def _completed_run(run_id: str):
    from app.agent.state import AgentRun, AgentStep

    return AgentRun(
        answer="已处理退款", steps=[AgentStep(1, "tool", tool="refund")],
        stopped_reason="final", trace_id="t", run_id=run_id, status="completed",
    )


def _readonly():
    from app.tools import build_readonly_registry
    from mocks.app import app as mock_app

    return build_readonly_registry(
        _client(mock_app)
    )


def _client(mock_app):
    from app.tools import MockClient

    return MockClient(
        "http://mock", timeout=2.0, transport=httpx.ASGITransport(app=mock_app)
    )


def test_search_knowledge_tool_via_tickets_domain(mock_app):
    registry = get_domain("tickets").build_registry(
        transport=httpx.ASGITransport(app=mock_app)
    )
    tool = registry.get("search_knowledge")
    params = registry.validate_args("search_knowledge", {"query": "退款 规则"})
    result = asyncio.run(tool.handler(params))
    assert result["results"]
    assert any("退款" in r["title"] for r in result["results"])
