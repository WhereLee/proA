"""Knowledge base API: ingest, hybrid search, stats."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.llm import get_llm
from app.memory import RagPipeline, default_knowledge

router = APIRouter()


class IngestRequest(BaseModel):
    doc_id: str
    title: str
    text: str
    source: str = ""
    scope: str = "default"


@router.post("/knowledge/ingest")
def ingest(req: IngestRequest) -> dict:
    kb = default_knowledge()
    chunks = kb.ingest(req.doc_id, req.title, req.text, req.source, req.scope)
    return {"doc_id": req.doc_id, "chunks": chunks, **kb.count()}


@router.get("/knowledge/search")
async def search(
    q: str, k: int = 3, scope: str | None = None, rerank: bool = False
) -> dict:
    kb = default_knowledge()
    if rerank:
        pipeline = RagPipeline(kb, llm=get_llm(), rerank=True)
        results = await pipeline.aretrieve(q, k, scope=scope)
    else:
        results = kb.search(q, k, scope=scope)
    return {"query": q, "results": results}


@router.get("/knowledge/stats")
def stats() -> dict:
    return default_knowledge().count()
