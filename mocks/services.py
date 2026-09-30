"""Mock business services with real-ish rules + fault injection hooks."""
from __future__ import annotations

from datetime import date
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from . import data
from .faults import postflight, preflight

order_router = APIRouter()
ticket_router = APIRouter()
notify_router = APIRouter()
user_router = APIRouter()


def _today() -> str:
    return date.today().isoformat()


def _get_order(order_id: str) -> dict:
    order = data.ORDERS.get(order_id)
    if order is None:
        raise HTTPException(status_code=404, detail="order not found")
    return order


class AddressUpdate(BaseModel):
    address: str


class RefundRequest(BaseModel):
    reason: str | None = None


_LOGISTICS_TEXT = {
    "paid": "商家已接单，等待发货",
    "shipped": "包裹运输中，预计 1-2 天送达",
    "delivered": "已签收",
}


@order_router.get("/{order_id}")
async def get_order(order_id: str) -> dict:
    await preflight("order")  # error/timeout fault; 'partial' has no effect on reads
    return _get_order(order_id)


@order_router.get("/{order_id}/logistics")
async def get_logistics(order_id: str) -> dict:
    await preflight("order")
    order = _get_order(order_id)
    return {
        "order_id": order_id,
        "carrier": "顺丰速运",
        "tracking_no": f"SF{order_id.lstrip('O')}",
        "latest": _LOGISTICS_TEXT.get(order["status"], "未知"),
    }


@order_router.post("/{order_id}/address")
async def update_address(
    order_id: str, body: AddressUpdate, request: Request
) -> dict:
    key = request.headers.get("Idempotency-Key")
    if key and key in data.IDEMPOTENCY:
        return {**data.IDEMPOTENCY[key], "idempotent_replay": True}

    mode = await preflight("order")
    order = _get_order(order_id)
    if order["status"] != "paid":
        raise HTTPException(
            status_code=409, detail="订单已发货或已签收，无法修改收货地址"
        )
    order["address"] = body.address
    result = dict(order)
    if key:
        data.IDEMPOTENCY[key] = result
    postflight(mode)  # 'partial' raises AFTER the side effect is committed
    return result


@order_router.post("/{order_id}/refund")
async def refund(order_id: str, body: RefundRequest, request: Request) -> dict:
    key = request.headers.get("Idempotency-Key")
    if key and key in data.IDEMPOTENCY:
        return {**data.IDEMPOTENCY[key], "idempotent_replay": True}

    mode = await preflight("order")
    order = _get_order(order_id)
    if order["refunded"]:
        raise HTTPException(status_code=409, detail="订单已退款，请勿重复申请")
    if order["status"] != "delivered":
        raise HTTPException(status_code=400, detail="仅已签收订单可申请退款")
    if order["refund_deadline"] < _today():
        raise HTTPException(status_code=400, detail="已超过退款期限")
    order["refunded"] = True
    result = {"order_id": order_id, "refunded": True, "amount": order["amount"]}
    if key:
        data.IDEMPOTENCY[key] = result
    postflight(mode)
    return result


class TicketCreate(BaseModel):
    user_id: str
    subject: str
    body: str = ""


@ticket_router.post("")
async def create_ticket(body: TicketCreate) -> dict:
    mode = await preflight("ticket")
    ticket_id = f"T{uuid4().hex[:8]}"
    ticket = {
        "ticket_id": ticket_id,
        "user_id": body.user_id,
        "subject": body.subject,
        "body": body.body,
        "status": "open",
    }
    data.TICKETS[ticket_id] = ticket
    postflight(mode)
    return dict(ticket)


@ticket_router.get("/{ticket_id}")
async def get_ticket(ticket_id: str) -> dict:
    await preflight("ticket")
    ticket = data.TICKETS.get(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="ticket not found")
    return ticket


class NotifyRequest(BaseModel):
    user_id: str
    channel: str = "sms"
    content: str


@notify_router.post("")
async def notify(body: NotifyRequest) -> dict:
    mode = await preflight("notify")
    record = {
        "user_id": body.user_id,
        "channel": body.channel,
        "content": body.content,
    }
    data.NOTIFY_LOG.append(record)
    postflight(mode)
    return {"sent": True, **record}


@user_router.get("/{user_id}")
async def get_user(user_id: str) -> dict:
    await preflight("user")
    user = data.USERS.get(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="user not found")
    return user
