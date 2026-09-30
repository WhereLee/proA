"""A small static corpus used as the research agent's mock web (offline-safe)."""
from __future__ import annotations

DOCS: list[dict] = [
    {
        "url": "https://example.com/llm-customer-service",
        "title": "国产大模型在企业客服场景的落地",
        "text": (
            "国产大模型在企业客服场景已广泛用于智能问答、工单分流与坐席辅助。"
            "落地收益包括降低人工坐席压力、缩短首响时间；主要挑战是幻觉控制、"
            "私有化部署成本与数据安全合规。建议结合检索增强（RAG）与人工审核。"
        ),
    },
    {
        "url": "https://example.com/rag-knowledge-base",
        "title": "RAG 在企业知识库中的落地要点",
        "text": (
            "检索增强生成（RAG）是让大模型基于企业知识库作答的主流方案。"
            "落地要点：文档解析与分块、混合检索加重排、引用溯源、以及效果评估。"
            "常见坑包括召回不准、上下文过长、知识更新滞后。"
        ),
    },
    {
        "url": "https://example.com/agent-workflow",
        "title": "AI Agent 与企业流程自动化",
        "text": (
            "AI Agent 通过工具调用把大模型接入企业系统，实现流程自动化。"
            "关键工程点：权限控制、人工确认、失败重试与幂等、状态与断点续跑、"
            "可观测与成本治理。适用于工单处理、数据查询、报告生成等场景。"
        ),
    },
    {
        "url": "https://example.com/llm-cost",
        "title": "大模型推理成本治理",
        "text": (
            "企业使用大模型的成本治理手段：token 统计与预算熔断、模型分级调用、"
            "语义缓存、上下文压缩，以及按需检索与重排以降低输入长度。"
        ),
    },
    {
        "url": "https://example.com/agent-eval",
        "title": "Agent 应用的效果评估",
        "text": (
            "Agent 评估应采用任务级数据集与规则评分优先，主观质量用 LLM-as-judge "
            "并用人工抽检校准；同时接入回归门禁，防止改动导致效果倒退。"
        ),
    },
    {
        "url": "https://example.com/prompt-injection",
        "title": "提示注入与安全防护",
        "text": (
            "提示注入常出现在工具返回值与外部文档中。防护要点：把外部数据视为不可信、"
            "最小权限、危险操作强制人工确认，并在代码层而非仅靠提示词实施授权。"
        ),
    },
]


def all_docs() -> list[dict]:
    return DOCS


def get(url: str) -> dict | None:
    return next((doc for doc in DOCS if doc["url"] == url), None)


def _score(query: str, doc: dict) -> int:
    q = set(ch for ch in query if not ch.isspace())
    text = doc["title"] + doc["text"]
    base = sum(1 for ch in q if ch in text)
    title_bonus = sum(2 for ch in q if ch in doc["title"])
    return base + title_bonus


def search(query: str, k: int = 3) -> list[dict]:
    ranked = sorted(DOCS, key=lambda doc: _score(query, doc), reverse=True)
    results = []
    for doc in ranked[: max(1, k)]:
        results.append(
            {
                "title": doc["title"],
                "url": doc["url"],
                "snippet": doc["text"][:60],
            }
        )
    return results
