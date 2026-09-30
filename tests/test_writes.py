import asyncio

import httpx

from app.tools import MockClient, build_write_registry


def _client(mock_app):
    return MockClient(
        "http://mock", timeout=2.0, transport=httpx.ASGITransport(app=mock_app)
    )


def test_write_tools_are_confirm_gated(mock_app):
    registry = build_write_registry(_client(mock_app))
    refund = registry.get("refund")
    assert refund.read_only is False
    assert refund.requires_confirmation is True
    assert refund.required_role == "agent"
    assert refund.idempotent is True


def test_refund_is_idempotent(mock_app):
    registry = build_write_registry(_client(mock_app))
    handler = registry.get("refund").handler
    params = registry.validate_args("refund", {"order_id": "O1001"})

    first = asyncio.run(handler(params, idempotency_key="key-1"))
    assert first["refunded"] is True

    second = asyncio.run(handler(params, idempotency_key="key-1"))
    assert second.get("idempotent_replay") is True


def test_refund_without_key_duplicates_is_rejected(mock_app):
    registry = build_write_registry(_client(mock_app))
    handler = registry.get("refund").handler
    params = registry.validate_args("refund", {"order_id": "O1001"})

    asyncio.run(handler(params))
    # No idempotency key -> a genuine duplicate is rejected by business rules.
    try:
        asyncio.run(handler(params))
        raise AssertionError("expected ToolExecutionError")
    except Exception as exc:  # noqa: BLE001
        assert "已退款" in str(exc)
