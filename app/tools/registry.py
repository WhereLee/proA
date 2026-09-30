"""Tool registry with schema-level argument validation (checklist P0).

Model-produced arguments are NEVER trusted or passed through raw: they are
parsed against the tool's pydantic params model first.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError


class ToolError(Exception):
    """Base class for tool-layer failures."""


class UnknownToolError(ToolError):
    pass


class ToolValidationError(ToolError):
    pass


class ToolExecutionError(ToolError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class Tool:
    name: str
    description: str
    params_model: type[BaseModel]
    handler: Callable[..., Awaitable[dict]]
    read_only: bool = True
    idempotent: bool = True
    requires_confirmation: bool = False
    required_role: str | None = None

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.params_model.model_json_schema(),
            "read_only": self.read_only,
            "requires_confirmation": self.requires_confirmation,
            "required_role": self.required_role,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None:
            raise UnknownToolError(f"unknown tool: {name}")
        return tool

    def names(self) -> list[str]:
        return list(self._tools)

    def describe(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def validate_args(self, name: str, args: dict[str, Any] | None) -> BaseModel:
        tool = self.get(name)
        try:
            return tool.params_model.model_validate(args or {})
        except ValidationError as exc:
            raise ToolValidationError(f"invalid args for tool '{name}': {exc}") from exc
