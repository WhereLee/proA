"""Robust JSON extraction + repair for structured LLM output (checklist P0).

Handles the common failure modes of real models:
- markdown code fences
- prose wrapped around the JSON
- trailing commas
"""
from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError


class StructuredOutputError(Exception):
    """Raised when the model output cannot be parsed/validated as JSON."""


_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _strip_fences(text: str) -> str:
    match = _FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _extract_balanced(text: str) -> str | None:
    """Return the first balanced {...} or [...] block, ignoring braces in strings."""
    start = next((i for i, ch in enumerate(text) if ch in "{["), None)
    if start is None:
        return None

    pairs = {"}": "{", "]": "["}
    stack: list[str] = []
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack or stack[-1] != pairs[ch]:
                return None
            stack.pop()
            if not stack:
                return text[start : i + 1]
    return None


def _remove_trailing_commas(text: str) -> str:
    return _TRAILING_COMMA_RE.sub(r"\1", text)


def extract_json(text: str | None) -> Any:
    """Best-effort extraction of a JSON value from raw model output."""
    if text is None:
        raise StructuredOutputError("empty output")

    stripped = text.strip()
    candidates = [stripped, _strip_fences(stripped)]
    balanced = _extract_balanced(stripped)
    if balanced:
        candidates.append(balanced)

    for candidate in candidates:
        for attempt in (candidate, _remove_trailing_commas(candidate)):
            try:
                return json.loads(attempt)
            except (json.JSONDecodeError, TypeError):
                continue

    raise StructuredOutputError(f"unable to parse JSON from output: {stripped[:200]!r}")


T = TypeVar("T", bound=BaseModel)


def parse_model(text: str | None, model_cls: type[T]) -> T:
    """Extract JSON and validate it against a pydantic model."""
    data = extract_json(text)
    try:
        return model_cls.model_validate(data)
    except ValidationError as exc:
        raise StructuredOutputError(str(exc)) from exc
