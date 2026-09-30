import asyncio

import httpx

from app.eval.retrieval import evaluate_retrieval, load_dataset
from app.llm.base import LLMResult
from app.main import create_app
from app.memory import (
    HashingEmbedder,
    KnowledgeBase,
    LLMQueryRewriter,
    LLMReranker,
    RagPipeline,
    chunk_document,
    cosine,
    rrf_fuse,
)
from app.memory.seed import SOP_DOCS
from app.persistence import Database


class JSONLLM:
    def __init__(self, text: str) -> None:
        self._text = text

    async def complete(self, *args, **kwargs):
        return LLMResult(text=self._text, model="j", provider="j", latency_ms=0.0)

    async def stream(self, *args, **kwargs):
        yield self._text


def _kb(tmp_path) -> KnowledgeBase:
    kb = KnowledgeBase(Database(tmp_path / "kb.db"))
    for doc in SOP_DOCS:
        kb.ingest(doc["doc_id"], doc["title"], doc["text"], source="sop", scope="public")
    return kb


def test_chunk_document_sentence_packing():
    text = "退款仅限已签收订单。已经退款的不能重复退。未签收不能直接退。"
    chunks = chunk_document(text, max_chars=16, overlap=0)
    assert len(chunks) >= 2
    assert all(len(c) <= 16 for c in chunks)
    assert "退款仅限已签收订单。" in "".join(chunks)


def test_hashing_embedder_similarity():
    embedder = HashingEmbedder()
    v = embedder.embed(["退款规则说明", "退款规则与期限", "今天天气真不错"])
    assert cosine(v[0], v[1]) > cosine(v[0], v[2])


def test_rrf_fuse_combines_rankings():
    fused = rrf_fuse([[0, 1, 2], [2, 0, 1]])
    assert len(fused) == 3
    assert fused[0][0] in (0, 2)


def test_hybrid_search_finds_expected_doc(tmp_path):
    kb = _kb(tmp_path)
    results = kb.search("如何申请发票", k=3, scope="public")
    assert any(r["doc_id"] == "sop-invoice" for r in results)


def test_scope_filter(tmp_path):
    kb = KnowledgeBase(Database(tmp_path / "kb.db"))
    kb.ingest("a", "退款", "退款相关内容", scope="s1")
    kb.ingest("b", "发票", "发票相关内容", scope="s2")
    assert all(r["scope"] == "s1" for r in kb.search("退款 发票", k=5, scope="s1"))
    assert all(r["scope"] == "s2" for r in kb.search("退款 发票", k=5, scope="s2"))


def test_llm_reranker_reorders():
    candidates = [
        {"title": "A", "text": "a", "score": 0.9},
        {"title": "B", "text": "b", "score": 0.1},
    ]
    reranked = asyncio.run(
        LLMReranker(JSONLLM('{"order":[1,0]}')).rerank("q", candidates, 2)
    )
    assert reranked[0]["title"] == "B"


def test_query_rewriter_keeps_original():
    queries = asyncio.run(
        LLMQueryRewriter(JSONLLM('{"queries":["退款","退钱"]}')).rewrite("怎么退款")
    )
    assert queries[0] == "怎么退款"
    assert "退款" in queries


def test_rag_pipeline_retrieve(tmp_path):
    kb = _kb(tmp_path)
    results = asyncio.run(RagPipeline(kb).aretrieve("怎么申请发票", k=2, scope="public"))
    assert results
    assert len(results) <= 2


def test_rag_pipeline_with_rerank_llm(tmp_path):
    kb = _kb(tmp_path)
    pipeline = RagPipeline(kb, llm=JSONLLM('{"order":[0,1]}'), rerank=True)
    results = asyncio.run(pipeline.aretrieve("怎么申请发票", k=2, scope="public"))
    assert len(results) <= 2


def test_retrieval_eval_metrics(tmp_path):
    kb = _kb(tmp_path)
    report = evaluate_retrieval(kb, load_dataset("eval/datasets/rag_v1.jsonl"))
    assert report["total"] >= 10
    assert report["recall"][5] == 1.0
    assert report["mrr"] >= 0.5


def test_knowledge_api():
    app = create_app()

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            stats = await c.get("/knowledge/stats")
            assert stats.json()["docs"] >= len(SOP_DOCS)

            search = await c.get("/knowledge/search", params={"q": "退款规则", "k": 3})
            assert search.status_code == 200
            assert any(r["doc_id"] == "sop-refund" for r in search.json()["results"])

            ingest = await c.post(
                "/knowledge/ingest",
                json={
                    "doc_id": "tmp-doc",
                    "title": "测试文档",
                    "text": "这是一段测试内容，关于退款的补充说明。",
                    "source": "test",
                    "scope": "public",
                },
            )
            assert ingest.status_code == 200
            assert ingest.json()["chunks"] >= 1

    asyncio.run(scenario())
