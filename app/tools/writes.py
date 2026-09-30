"""Write tool definitions (phase 2): side effects, idempotent, confirm-gated.

Write tools carry an Idempotency-Key derived from (run_id, tool, args) so that a
retry after an ambiguous failure (side effect applied but response lost) does not
double-apply. They also require human confirmation and the 'agent' role.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from .mock_client import MockClient
from .registry import Tool, ToolRegistry


def _idem_headers(key: str | None) -> dict | None:
    return {"Idempotency-Key": key} if key else None


class UpdateAddressParams(BaseModel):
    order_id: str = Field(..., description="订单号，例如 O1001", min_length=1)
    address: str = Field(..., description="新的收货地址", min_length=1)


class RefundParams(BaseModel):
    order_id: str = Field(..., description="订单号，例如 O1001", min_length=1)
    reason: str | None = Field(default=None, description="退款原因（可选）")


def register_writes(registry: ToolRegistry, client: MockClient) -> None:
    async def update_address(
        params: UpdateAddressParams, *, idempotency_key: str | None = None
    ) -> dict:
        return await client.post(
            f"/orders/{params.order_id}/address",
            json={"address": params.address},
            headers=_idem_headers(idempotency_key),
        )

    async def refund(params: RefundParams, *, idempotency_key: str | None = None) -> dict:
        return await client.post(
            f"/orders/{params.order_id}/refund",
            json={"reason": params.reason},
            headers=_idem_headers(idempotency_key),
        )

    registry.register(
        Tool(
            name="update_address",
            description="修改订单收货地址（仅未发货订单可改）。",
            params_model=UpdateAddressParams,
            handler=update_address,
            read_only=False,
            idempotent=True,
            requires_confirmation=True,
            required_role="agent",
        )
    )
    registry.register(
        Tool(
            name="refund",
            description="对已签收订单发起退款。",
            params_model=RefundParams,
            handler=refund,
            read_only=False,
            idempotent=True,
            requires_confirmation=True,
            required_role="agent",
        )
    )


def build_write_registry(client: MockClient) -> ToolRegistry:
    registry = ToolRegistry()
    register_writes(registry, client)
    return registry
