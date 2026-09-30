"""Data-analysis domain (project B): text-to-SQL over a read-only analytical DB.

Safety: only single read-only SELECT/WITH statements are accepted, the SQLite
connection runs with `PRAGMA query_only=ON`, results are row-capped, and SQL
errors are classified as business errors (400) so they are not blindly retried.
"""
from __future__ import annotations

import re
import sqlite3

from pydantic import BaseModel, Field

from app.tools import Tool, ToolExecutionError, ToolRegistry, ToolValidationError

from . import sales_data
from .base import Domain

DATA_PROMPT = """你是数据分析助手，只能使用下面这些工具：

{tools}

规则：
1. 每次只输出一个 JSON 对象，不要输出多余文字或 Markdown。
2. 调用工具时输出：{{"action":"tool","tool":"工具名","args":{{...}}}}
3. 信息足够时输出：{{"action":"final","answer":"给用户的最终回答，包含关键数字/结论"}}
4. 先用 list_tables 了解表结构，再用 run_sql 查询；run_sql 只允许 SELECT。
5. 结论必须来自查询结果，不要编造数据。"""

_MAX_ROWS = 200
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|"
    r"pragma|vacuum|reindex|grant|revoke)\b",
    re.IGNORECASE,
)


def _guard(sql: str) -> str:
    text = (sql or "").strip().rstrip(";").strip()
    if not text:
        raise ToolValidationError("empty SQL")
    if ";" in text:
        raise ToolValidationError("only a single SQL statement is allowed")
    if _FORBIDDEN.search(text):
        raise ToolValidationError("only read-only SELECT queries are allowed")
    if not text.lower().startswith(("select", "with")):
        raise ToolValidationError("only SELECT / WITH queries are allowed")
    return text


class ListTablesParams(BaseModel):
    pass


class RunSqlParams(BaseModel):
    sql: str = Field(..., description="只读 SQL 查询（SELECT）", min_length=1)


def register_data_tools(registry: ToolRegistry) -> None:
    async def list_tables(params: ListTablesParams, *, idempotency_key: str | None = None) -> dict:
        return {"tables": sales_data.schema()}

    async def run_sql(params: RunSqlParams, *, idempotency_key: str | None = None) -> dict:
        sql = _guard(params.sql)
        try:
            cursor = sales_data.get_connection().execute(sql)
            columns = [desc[0] for desc in (cursor.description or [])]
            rows = [list(row) for row in cursor.fetchmany(_MAX_ROWS)]
        except sqlite3.Error as exc:
            # business error -> status 400 so it is not retried
            raise ToolExecutionError(f"SQL error: {exc}", status_code=400) from exc
        return {"columns": columns, "rows": rows, "row_count": len(rows)}

    registry.register(
        Tool("list_tables", "列出数据库中的表及其字段。", ListTablesParams, list_tables)
    )
    registry.register(
        Tool(
            "run_sql",
            "执行只读 SQL（仅 SELECT / WITH）并返回列与行。",
            RunSqlParams,
            run_sql,
        )
    )


def build_data_registry(*, transport=None) -> ToolRegistry:
    registry = ToolRegistry()
    register_data_tools(registry)
    return registry


DATA_ANALYSIS = Domain("data", DATA_PROMPT, build_data_registry)
