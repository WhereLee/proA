"""Deterministic oracle LLM for CI/harness self-checks.

Replays the *expected* tool calls from the dataset so the eval pipeline
(graders/gate/reporting) can be validated without a real model or API key.
This is NOT a model and its pass rate is NOT a quality metric.
"""
from __future__ import annotations

import json
import re

from app.llm.scripted import ScriptedLLM


def _extract(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text or "")
    return match.group(0) if match else None


def _order_id(task: dict) -> str | None:
    for obs in task.get("expected_observations", []):
        args = obs.get("args") or {}
        if "order_id" in args:
            return args["order_id"]
    return _extract(task.get("input", ""), r"O\d+")


def _user_id(task: dict) -> str | None:
    for obs in task.get("expected_observations", []):
        args = obs.get("args") or {}
        if "user_id" in args:
            return args["user_id"]
    return _extract(task.get("input", ""), r"\bU\d+\b")


def _args_for(tool: str, task: dict) -> dict:
    for obs in task.get("expected_observations", []):
        if obs["tool"] == tool and obs.get("args"):
            return obs["args"]
    order_id = _order_id(task)
    if tool in ("get_order", "get_logistics", "refund") and order_id:
        return {"order_id": order_id}
    if tool == "update_address" and order_id:
        return {"order_id": order_id, "address": "上海市浦东新区新地址 99 号"}
    if tool == "get_user":
        return {"user_id": _user_id(task)}
    return {}


def build_oracle_llm(tasks: list[dict]) -> ScriptedLLM:
    script: dict[str, list[str]] = {}
    for task in tasks:
        if task.get("oracle"):
            # explicit scripted actions (e.g. arbitrary SQL)
            outputs = [json.dumps(action, ensure_ascii=False) for action in task["oracle"]]
        else:
            outputs = [
                json.dumps(
                    {"action": "tool", "tool": tool, "args": _args_for(tool, task)},
                    ensure_ascii=False,
                )
                for tool in task.get("expected_tools", [])
            ]
        outputs.append(json.dumps({"action": "final", "answer": "完成"}, ensure_ascii=False))
        script[task["input"]] = outputs
    return ScriptedLLM(script)
