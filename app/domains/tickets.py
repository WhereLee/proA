"""Ticket domain (project A): read-only + write tools over the mock systems,
plus a knowledge-base (RAG) tool for SOPs."""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.agent.loop import SYSTEM_PROMPT_TEMPLATE
from app.config import settings
from app.memory import default_knowledge
from app.tools import MockClient, Tool, build_registry
from app.tools.registry import ToolRegistry

from .base import Domain


class SearchKnowledgeParams(BaseModel):
    query: str = Field(..., description="检索关键词", min_length=1)
    k: int = Field(default=3, description="返回条数", ge=1, le=10)


async def _search_knowledge(params: SearchKnowledgeParams, *, idempotency_key: str | None = None) -> dict:
    return {"results": default_knowledge().search(params.query, params.k)}


def build_tickets_registry(*, transport=None) -> ToolRegistry:
    client = MockClient(
        settings.mock_base_url,
        timeout=settings.agent.tool_timeout,
        transport=transport,
    )
    registry = build_registry(client, include_writes=True)
    registry.register(
        Tool(
            "search_knowledge",
            "检索内部知识库/SOP（退款规则、改地址规则、权限与确认、处理流程）。",
            SearchKnowledgeParams,
            _search_knowledge,
            read_only=True,
        )
    )
    return registry


TICKETS = Domain("tickets", SYSTEM_PROMPT_TEMPLATE, build_tickets_registry)
