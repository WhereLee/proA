"""Research domain (project C): plan -> search -> fetch -> cited report.

Long-running by nature; runs are submitted to a background queue (see
app/api/research.py) and can be resumed from a checkpoint after a restart.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.tools import Tool, ToolExecutionError, ToolRegistry, ToolValidationError

from . import research_corpus
from .base import Domain

RESEARCH_PROMPT = """你是行业调研助手，可以使用下面这些工具：

{tools}

规则：
1. 每次只输出一个 JSON 对象，不要输出多余文字或 Markdown。
2. 先检索：{{"action":"tool","tool":"search","args":{{"query":"关键词","k":3}}}}
3. 需要详情时：{{"action":"tool","tool":"fetch","args":{{"url":"链接"}}}}
4. 调研完成后输出：{{"action":"final","answer":"调研报告"}}
5. 报告最后必须有一节「来源」，逐条列出你实际检索/抓取过的链接（以 http 开头）；不得编造信息或不存在的链接。
6. 先检索多组关键词、必要时抓取详情，再综合成结构化的结论。"""


class SearchParams(BaseModel):
    query: str = Field(..., description="检索关键词", min_length=1)
    k: int = Field(default=3, description="返回结果条数", ge=1, le=10)


class FetchParams(BaseModel):
    url: str = Field(..., description="要抓取的链接（来自 search 结果）", min_length=1)


def register_research_tools(registry: ToolRegistry) -> None:
    async def search(params: SearchParams, *, idempotency_key: str | None = None) -> dict:
        return {"results": research_corpus.search(params.query, params.k)}

    async def fetch(params: FetchParams, *, idempotency_key: str | None = None) -> dict:
        doc = research_corpus.get(params.url)
        if doc is None:
            raise ToolExecutionError(f"not found: {params.url}", status_code=404)
        return {"url": doc["url"], "title": doc["title"], "text": doc["text"]}

    registry.register(
        Tool("search", "按关键词检索资料，返回标题、链接与摘要。", SearchParams, search)
    )
    registry.register(
        Tool("fetch", "按链接抓取文档全文。", FetchParams, fetch)
    )


def build_research_registry(*, transport=None) -> ToolRegistry:
    registry = ToolRegistry()
    register_research_tools(registry)
    return registry


# Research is planning-heavy: allow more steps than the global default.
RESEARCH = Domain("research", RESEARCH_PROMPT, build_research_registry, max_steps=12)
