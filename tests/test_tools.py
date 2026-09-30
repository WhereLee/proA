import asyncio

import pytest

from app.tools.registry import ToolValidationError, UnknownToolError


def test_registry_lists_readonly_tools(readonly_registry):
    assert set(readonly_registry.names()) == {
        "get_order",
        "get_logistics",
        "get_user",
        "get_ticket",
    }
    assert all(tool["parameters"] for tool in readonly_registry.describe())


def test_validate_args_rejects_bad_params(readonly_registry):
    with pytest.raises(ToolValidationError):
        readonly_registry.validate_args("get_order", {"wrong": "x"})


def test_unknown_tool_raises(readonly_registry):
    with pytest.raises(UnknownToolError):
        readonly_registry.get("nope")


def test_execute_get_order(readonly_registry):
    tool = readonly_registry.get("get_order")
    params = readonly_registry.validate_args("get_order", {"order_id": "O1001"})
    result = asyncio.run(tool.handler(params))
    assert result["status"] == "delivered"


def test_execute_get_logistics(readonly_registry):
    tool = readonly_registry.get("get_logistics")
    params = readonly_registry.validate_args("get_logistics", {"order_id": "O1002"})
    result = asyncio.run(tool.handler(params))
    assert result["latest"] == "包裹运输中，预计 1-2 天送达"
