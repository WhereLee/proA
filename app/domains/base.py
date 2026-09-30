"""Domain pack: a task domain = a tool registry + a system prompt template.

The Agent loop is domain-agnostic; a Domain binds it to a concrete task family
(tickets / data analysis / research) by supplying tools and a persona prompt.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.tools.registry import ToolRegistry


@dataclass(frozen=True)
class Domain:
    name: str
    system_prompt_template: str
    registry_factory: Callable[..., ToolRegistry]
    max_steps: int | None = None  # overrides the global step budget if set

    def build_registry(self, **kwargs) -> ToolRegistry:
        return self.registry_factory(**kwargs)
