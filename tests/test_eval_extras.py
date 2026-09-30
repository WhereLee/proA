import asyncio
import json

from app.agent.loop import AgentRunner
from app.agent.state import RunStore
from app.eval.analysis import classify_failure, summarize_failures
from app.eval.gate import check_gate
from app.eval.judge import judge_run
from app.eval.runner import EvalReport, evaluate
from app.llm.base import LLMResult, Usage
from app.llm.scripted import ScriptedLLM

DATASET = "eval/datasets/readonly_v1.jsonl"


def _run(registry, script, user_input):
    runner = AgentRunner(ScriptedLLM(script), registry, run_store=RunStore())
    return asyncio.run(runner.run(user_input))


def test_classify_tool_missing(readonly_registry):
    task = {"expected_tools": ["get_logistics"], "expected_observations": []}
    run = _run(readonly_registry, {"q": ['{"action":"final","answer":"ok"}']}, "q")
    assert classify_failure(task, run) == "tool_missing"


def test_classify_observation_mismatch(readonly_registry):
    task = {
        "expected_tools": ["get_order"],
        "expected_observations": [
            {"tool": "get_order", "args": {"order_id": "O1001"}, "path": "status", "equals": "X"}
        ],
    }
    script = {
        "q": [
            '{"action":"tool","tool":"get_order","args":{"order_id":"O1001"}}',
            '{"action":"final","answer":"ok"}',
        ]
    }
    run = _run(readonly_registry, script, "q")
    assert classify_failure(task, run) == "observation_mismatch"


def test_gate_fails_below_threshold():
    report = EvalReport(total=10, passed=5, pass_rate=0.5)
    assert check_gate(report, min_pass_rate=0.9).passed is False
    assert check_gate(report, min_pass_rate=0.4).passed is True


def test_gate_detects_regression():
    report = EvalReport(total=10, passed=8, pass_rate=0.8)
    result = check_gate(report, min_pass_rate=0.5, baseline_pass_rate=0.9, max_regression=0.05)
    assert result.passed is False
    assert "regression" in result.reason


class _JudgeLLM:
    """Returns a fixed verdict (double-eval agrees)."""

    name = "judge"

    def __init__(self, passed: bool):
        self._passed = passed

    async def complete(self, messages, *, model=None, temperature=None):
        return LLMResult(
            text=json.dumps({"passed": self._passed, "reason": "x"}),
            model="judge",
            provider=self.name,
            latency_ms=0.0,
            usage=Usage(),
        )

    async def stream(self, messages, *, model=None, temperature=None):
        yield ""


def test_judge_double_eval_agrees(readonly_registry):
    run = _run(readonly_registry, {"q": ['{"action":"final","answer":"ok"}']}, "q")
    verdict = asyncio.run(judge_run(_JudgeLLM(True), {"input": "q"}, run))
    assert verdict["passed"] is True
    assert verdict["disagreement"] is False
    assert len(verdict["verdicts"]) == 2


def test_evaluate_with_oracle_pipeline(readonly_registry):
    from app.eval.oracle import build_oracle_llm
    from app.eval.runner import load_dataset

    tasks = load_dataset(DATASET)[:5]
    report = asyncio.run(
        evaluate(
            DATASET,
            llm=build_oracle_llm(tasks),
            registry=readonly_registry,
            limit=5,
        )
    )
    assert report.pass_rate == 1.0
    assert report.failure_summary == {}
