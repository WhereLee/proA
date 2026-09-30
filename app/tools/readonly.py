"""Read-only tool definitions (phase 1)."""
from __future__ import annotations

from pydantic import BaseModel, Field

from .mock_client import MockClient
from .registry import Tool, ToolRegistry


class GetOrderParams(BaseModel):
    order_id: str = Field(..., description="订单号，例如 O1001", min_length=1)


class GetLogisticsParams(BaseModel):
    order_id: str = Field(..., description="订单号，例如 O1001", min_length=1)


class GetUserParams(BaseModel):
    user_id: str = Field(..., description="用户 ID，例如 U1", min_length=1)


class GetTicketParams(BaseModel):
    ticket_id: str = Field(..., description="工单号，例如 T1a2b3c4d", min_length=1)


def register_readonly(registry: ToolRegistry, client: MockClient) -> None:
    async def get_order(params: GetOrderParams, *, idempotency_key: str | None = None) -> dict:
        return await client.get(f"/orders/{params.order_id}")

    async def get_logistics(
        params: GetLogisticsParams, *, idempotency_key: str | None = None
    ) -> dict:
        return await client.get(f"/orders/{params.order_id}/logistics")

    async def get_user(params: GetUserParams, *, idempotency_key: str | None = None) -> dict:
        return await client.get(f"/users/{params.user_id}")

    async def get_ticket(
        params: GetTicketParams, *, idempotency_key: str | None = None
    ) -> dict:
        return await client.get(f"/tickets/{params.ticket_id}")

    registry.register(
        Tool(
            name="get_order",
            description="查询订单详情：状态、金额、收货地址、是否已退款。",
            params_model=GetOrderParams,
            handler=get_order,
        )
    )
    registry.register(
        Tool(
            name="get_logistics",
            description="查询订单物流信息：承运商、运单号、最新物流状态。",
            params_model=GetLogisticsParams,
            handler=get_logistics,
        )
    )
    registry.register(
        Tool(
            name="get_user",
            description="查询用户信息：姓名、角色（customer/agent）。",
            params_model=GetUserParams,
            handler=get_user,
        )
    )
    registry.register(
        Tool(
            name="get_ticket",
            description="按工单号查询工单详情与状态。",
            params_model=GetTicketParams,
            handler=get_ticket,
        )
    )


def build_readonly_registry(client: MockClient) -> ToolRegistry:
    registry = ToolRegistry()
    register_readonly(registry, client)
    return registry
