"""Expose the agent's tools as an MCP server (checklist P1: MCP).

    python -m app.mcp            # serves over stdio

The tool catalog is reused as-is; MCP is just an adapter over the ToolRegistry.
Requires the optional `mcp` dependency.
"""
from __future__ import annotations

import json

from mcp import types
from mcp.server.lowlevel import Server

from app.domains import get_domain
from app.tools.registry import ToolError, ToolRegistry


def default_registry() -> ToolRegistry:
    # reuse the domain catalog so MCP exposes the same tools (incl. search_knowledge)
    return get_domain("tickets").build_registry()


async def list_tools(registry: ToolRegistry) -> list[types.Tool]:
    return [
        types.Tool(
            name=tool["name"],
            description=tool["description"],
            input_schema=tool["parameters"],
        )
        for tool in registry.describe()
    ]


async def call_tool(
    registry: ToolRegistry, name: str, arguments: dict | None
) -> types.CallToolResult:
    try:
        validated = registry.validate_args(name, arguments or {})
        tool = registry.get(name)
        observed = await tool.handler(validated, idempotency_key=None)
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text", text=json.dumps(observed, ensure_ascii=False)
                )
            ]
        )
    except ToolError as exc:
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=str(exc))], is_error=True
        )


def build_server(registry: ToolRegistry | None = None) -> Server:
    registry = registry or default_registry()

    async def on_list_tools(context, params):  # noqa: ANN001
        return types.ListToolsResult(tools=await list_tools(registry))

    async def on_call_tool(context, params):  # noqa: ANN001
        return await call_tool(registry, params.name, params.arguments)

    return Server(
        "agent-ticket",
        version="0.1.0",
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


async def run_stdio() -> None:
    from mcp.server.stdio import stdio_server

    server = build_server()
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())
