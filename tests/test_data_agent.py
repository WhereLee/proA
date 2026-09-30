import asyncio

import pytest

from app.agent.loop import AgentRunner
from app.agent.state import RunStore
from app.domains import DATA_ANALYSIS, get_domain
from app.domains.data_analysis import _guard
from app.eval.oracle import build_oracle_llm
from app.eval.runner import evaluate, load_dataset
from app.llm.scripted import ScriptedLLM
from app.tools.registry import ToolExecutionError, ToolValidationError

DATA = "eval/datasets/data_v1.jsonl"


def _registry():
    return get_domain("data").build_registry()


def test_domain_has_data_prompt():
    assert "数据分析助手" in DATA_ANALYSIS.system_prompt_template


def test_sql_guard_accepts_select_and_rejects_dangerous():
    assert _guard("SELECT 1") == "SELECT 1"
    assert _guard(" select * from orders ; ") == "select * from orders"
    for bad in ["DROP TABLE orders", "SELECT 1; SELECT 2", "UPDATE orders SET amount=0", ""]:
        with pytest.raises(ToolValidationError):
            _guard(bad)


def test_run_sql_rejects_write():
    registry = _registry()
    params = registry.validate_args("run_sql", {"sql": "DELETE FROM orders"})
    with pytest.raises(ToolValidationError):
        asyncio.run(registry.get("run_sql").handler(params))


def test_run_sql_returns_rows():
    registry = _registry()
    params = registry.validate_args("run_sql", {"sql": "SELECT COUNT(*) AS c FROM orders"})
    result = asyncio.run(registry.get("run_sql").handler(params))
    assert result["columns"] == ["c"]
    assert result["rows"][0][0] == 12


def test_run_sql_bad_sql_is_non_retryable_business_error():
    registry = _registry()
    params = registry.validate_args("run_sql", {"sql": "SELECT nope FROM orders"})
    with pytest.raises(ToolExecutionError) as excinfo:
        asyncio.run(registry.get("run_sql").handler(params))
    assert excinfo.value.status_code == 400


def test_agent_list_tables_then_query():
    script = {
        "一共有多少张订单？": [
            '{"action":"tool","tool":"list_tables","args":{}}',
            '{"action":"tool","tool":"run_sql","args":{"sql":"SELECT COUNT(*) AS c FROM orders"}}',
            '{"action":"final","answer":"共 12 张订单"}',
        ]
    }
    runner = AgentRunner(ScriptedLLM(script), _registry(), run_store=RunStore())
    run = asyncio.run(runner.run("一共有多少张订单？"))
    assert run.stopped_reason == "final"
    assert any(
        s.tool == "run_sql" and s.result and s.result["rows"][0][0] == 12
        for s in run.steps
    )


def test_oracle_passes_data_dataset():
    tasks = load_dataset(DATA)
    report = asyncio.run(
        evaluate(DATA, llm=build_oracle_llm(tasks), registry=_registry())
    )
    assert report.pass_rate == 1.0, report.failure_summary
