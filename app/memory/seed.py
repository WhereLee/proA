"""Seed SOP knowledge for the ticket domain.

`default_knowledge()` builds the store/embedder from settings (SQLite+hashing by
default; pgvector+ONNX when configured) and seeds the SOP docs.
"""
from __future__ import annotations

from .knowledge import KnowledgeBase

SOP_DOCS = [
    {
        "doc_id": "sop-refund",
        "title": "退款规则",
        "text": (
            "仅已签收（delivered）且在退款期限内的订单可申请退款；已经退款的订单不可重复退款；"
            "未签收（paid/shipped）的订单不能直接退款。退款属于写操作，需要客服角色发起并经过人工确认。"
        ),
    },
    {
        "doc_id": "sop-address",
        "title": "修改收货地址规则",
        "text": (
            "只有未发货（paid）的订单可以修改收货地址；已发货（shipped）或已签收（delivered）的订单"
            "不允许修改地址，应引导用户联系快递或走售后流程。修改地址属于写操作，需要人工确认。"
        ),
    },
    {
        "doc_id": "sop-permission",
        "title": "权限与人工确认",
        "text": (
            "写操作（退款、改地址）仅允许 agent 角色调用，普通用户请求需要转人工客服处理。"
            "所有写操作在真正执行前都必须经过人工确认（HITL），确认记录会写入审计日志。"
        ),
    },
    {
        "doc_id": "sop-workflow",
        "title": "工单处理流程",
        "text": (
            "标准流程：理解用户诉求 → 查询订单/物流/用户信息 → 校验业务规则 → 危险操作请求人工确认 → "
            "执行并通知用户。查不到信息时应如实说明，不要编造。"
        ),
    },
    {
        "doc_id": "sop-logistics",
        "title": "物流查询与催发货",
        "text": (
            "查询物流需使用订单号获取承运商、运单号与最新状态。状态为 paid 表示商家已接单待发货，"
            "shipped 表示运输中，delivered 表示已签收。用户催发货时先确认订单状态再决定是否转人工。"
        ),
    },
    {
        "doc_id": "sop-invoice",
        "title": "发票申请",
        "text": (
            "用户可申请电子发票，需提供订单号与抬头信息。发票开具后不可修改抬头，如需重开应先作废再重开。"
            "发票类请求通常需要转人工财务处理。"
        ),
    },
    {
        "doc_id": "sop-cancel",
        "title": "取消订单",
        "text": (
            "仅未发货（paid）的订单可以取消；已发货或已签收的订单无法取消，只能走退货退款流程。"
            "取消订单属于写操作，需要人工确认。"
        ),
    },
    {
        "doc_id": "sop-escalation",
        "title": "转人工与升级",
        "text": (
            "当用户诉求超出规则范围、涉及投诉、或需要权限之外的操作时，应创建工单并转人工客服处理，"
            "同时向用户说明预计的处理方式。"
        ),
    },
]


_default: KnowledgeBase | None = None


def _seed(kb: KnowledgeBase) -> None:
    if kb.count()["docs"] < len(SOP_DOCS):
        for doc in SOP_DOCS:
            kb.ingest(
                doc["doc_id"], doc["title"], doc["text"], source="sop", scope="public"
            )


def default_knowledge() -> KnowledgeBase:
    global _default
    if _default is None:
        from .factory import build_knowledge

        _default = build_knowledge()
    _seed(_default)
    return _default


def set_knowledge(kb: KnowledgeBase | None) -> None:
    """Override the default knowledge base (used by tests)."""
    global _default
    _default = kb
