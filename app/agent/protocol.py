"""Agent output protocol: a single JSON object per model turn.

Kept provider-agnostic (JSON-in-text) instead of native function-calling so the
loop is easy to test offline and works with any OpenAI-compatible model.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from app.llm.structured import StructuredOutputError, extract_json


class ToolCallAction(BaseModel):
    action: Literal["tool"]
    tool: str
    args: dict[str, Any] = {}
    thought: str | None = None


class FinalAction(BaseModel):
    action: Literal["final"]
    answer: str
    thought: str | None = None


AgentAction = ToolCallAction | FinalAction


class ActionParseError(StructuredOutputError):
    pass


def parse_action(text: str | None) -> AgentAction:
    data = extract_json(text)
    if not isinstance(data, dict):
        raise ActionParseError("model output must be a JSON object")

    action = data.get("action")
    try:
        if action == "tool":
            return ToolCallAction.model_validate(data)
        if action == "final":
            return FinalAction.model_validate(data)
    except ValidationError as exc:
        raise ActionParseError(str(exc)) from exc
    raise ActionParseError(f"unknown action: {action!r}")
