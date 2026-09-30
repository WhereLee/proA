import asyncio
import json

from app.agent.loop import AgentRunner
from app.eval.graders import grade_task
from app.eval.runner import evaluate, load_dataset
from app.llm.scripted import ScriptedLLM

DATASET = "eval/datasets/readonly_v1.jsonl"


def test_dataset_has_enough_tasks():
    tasks = load_dataset(DATASET)
    assert len(tasks) >= 50


def test_eval_passes_with_scripted_oracle(readonly_registry):
    tasks = load_dataset(DATASET)[:4]
    script: dict[str, list[str]] = {}
    for task in tasks:
        calls = [
            json.dumps(
                {"action": "tool", "tool": obs["tool"], "args": obs["args"]},
                ensure_ascii=False,
            )
            for obs in task.get("expected_observations", [])
        ]
        calls.append(json.dumps({"action": "final", "answer": "完成"}, ensure_ascii=False))
        script[task["input"]] = calls

    report = asyncio.run(
        evaluate(DATASET, llm=ScriptedLLM(script), registry=readonly_registry, limit=4)
    )
    assert report.total == 4
    assert report.pass_rate == 1.0


def test_grader_flags_wrong_observation(readonly_registry):
    task = {
        "task_id": "x",
        "input": "q",
        "expected_tools": ["get_order"],
        "expected_observations": [
            {
                "tool": "get_order",
                "args": {"order_id": "O1001"},
                "path": "status",
                "equals": "WRONG",
            }
        ],
    }
    script = {
        "q": [
            '{"action":"tool","tool":"get_order","args":{"order_id":"O1001"}}',
            '{"action":"final","answer":"ok"}',
        ]
    }
    run = asyncio.run(AgentRunner(ScriptedLLM(script), readonly_registry).run("q"))
    assert grade_task(task, run).passed is False


def test_grader_flags_missing_tool(readonly_registry):
    task = {
        "task_id": "y",
        "input": "q",
        "expected_tools": ["get_logistics"],
        "expected_observations": [],
    }
    script = {"q": ['{"action":"final","answer":"ok"}']}
    run = asyncio.run(AgentRunner(ScriptedLLM(script), readonly_registry).run("q"))
    assert grade_task(task, run).passed is False
