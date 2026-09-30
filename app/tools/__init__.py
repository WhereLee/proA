from .mock_client import MockClient
from .readonly import build_readonly_registry, register_readonly
from .registry import (
    Tool,
    ToolError,
    ToolExecutionError,
    ToolRegistry,
    ToolValidationError,
    UnknownToolError,
)
from .writes import build_write_registry, register_writes


def build_registry(client: MockClient, *, include_writes: bool = True) -> ToolRegistry:
    """Full tool catalog: read-only tools plus (optionally) write tools."""
    registry = ToolRegistry()
    register_readonly(registry, client)
    if include_writes:
        register_writes(registry, client)
    return registry


__all__ = [
    "MockClient",
    "Tool",
    "ToolError",
    "ToolExecutionError",
    "ToolRegistry",
    "ToolValidationError",
    "UnknownToolError",
    "build_registry",
    "build_readonly_registry",
    "build_write_registry",
    "register_readonly",
    "register_writes",
]
