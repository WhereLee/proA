"""Eval runner: run the agent over a task-level dataset and aggregate metrics.

Supports the v2 (hard) dataset schema:
- role: principal role for the run (default "agent")
- fault: mock fault-injection config applied before the task
- confirm: whether to auto-approve a pending confirmation (default True)
- reset_state: reset mock data + faults before each task
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.agent.loop import AgentRunner
from app.tools.registry import ToolRegistry

from .analysis import classify_failure, summarize_failures
from .graders import grade_task
from .judge import judge_run


@dataclass
class EvalReport:
    total: int
    passed: int
    pass_rate: float
    failure_summary: dict[str, int] = field(default_factory=dict)
    results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            tasks.append(json.loads(line))
    return tasks


def _reset_mock_state(task: dict[str, Any]) -> None:
    from mocks import data
    from mocks.faults import FaultRequest, configure, reset

    reset()
    data.reset()
    if task.get("fault"):
        configure(FaultRequest(**task["fault"]))


async def _run_with_confirmation(runner: AgentRunner, task: dict[str, Any]):
    run = await runner.run(task["input"], role=task.get("role", "agent"))
    approved = task.get("confirm", True)
    guard = 0
    while run.status == "awaiting_confirmation" and guard < 3:
        async for _ in runner.resume(run.run_id, approved=approved):
            pass
        run = runner.get_run(run.run_id)
        guard += 1
    return run


async def evaluate(
    dataset_path: str | Path,
    *,
    llm,
    registry: ToolRegistry,
    max_steps: int = 6,
    limit: int = 0,
    judge_llm=None,
    judge_passes: int = 2,
    reset_state: bool = False,
) -> EvalReport:
    tasks = load_dataset(dataset_path)
    if limit:
        tasks = tasks[:limit]

    runner = AgentRunner(llm, registry, max_steps=max_steps)
    results: list[dict[str, Any]] = []

    for task in tasks:
        if reset_state:
            _reset_mock_state(task)

        run = await _run_with_confirmation(runner, task)
        grade = grade_task(task, run)
        category = None if grade.passed else classify_failure(task, run)
        judge = None
        if judge_llm is not None:
            judge = await judge_run(judge_llm, task, run, passes=judge_passes)
        results.append(
            {
                "task_id": task["task_id"],
                "input": task["input"],
                "passed": grade.passed,
                "category": category,
                "answer": run.answer,
                "status": run.status,
                "cost_usd": round(run.cost_usd, 6),
                "details": grade.details,
                "judge": judge,
            }
        )

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    return EvalReport(
        total=total,
        passed=passed,
        pass_rate=(passed / total) if total else 0.0,
        failure_summary=summarize_failures(results),
        results=results,
    )
