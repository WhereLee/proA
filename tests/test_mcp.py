import asyncio

import pytest

pytest.importorskip("mcp")

from app.mcp import server as mcp_server  # noqa: E402


def test_list_tools_exposes_catalog(full_registry):
    tools = asyncio.run(mcp_server.list_tools(full_registry))
    names = {t.name for t in tools}
    assert {"get_order", "get_logistics", "get_user", "refund"} <= names
    assert all(t.input_schema for t in tools)


def test_call_tool_ok(full_registry):
    result = asyncio.run(
        mcp_server.call_tool(full_registry, "get_order", {"order_id": "O1001"})
    )
    assert result.is_error is not True
    assert "delivered" in result.content[0].text


def test_call_tool_invalid_args_sets_error(full_registry):
    result = asyncio.run(mcp_server.call_tool(full_registry, "get_order", {}))
    assert result.is_error is True


def test_build_server(full_registry):
    server = mcp_server.build_server(full_registry)
    assert server is not None
