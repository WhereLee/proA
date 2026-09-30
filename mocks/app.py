"""Standalone mock business-system gateway (run on its own port, e.g. 8100)."""
from __future__ import annotations

from fastapi import FastAPI

from . import data
from .faults import FaultRequest, configure, reset as reset_fault, snapshot
from .services import notify_router, order_router, ticket_router, user_router


def create_app() -> FastAPI:
    app = FastAPI(title="agent-ticket mocks", version="0.1.0")
    app.include_router(order_router, prefix="/orders", tags=["order"])
    app.include_router(ticket_router, prefix="/tickets", tags=["ticket"])
    app.include_router(notify_router, prefix="/notify", tags=["notify"])
    app.include_router(user_router, prefix="/users", tags=["user"])

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": "mocks"}

    @app.get("/_fault")
    def get_fault() -> dict:
        return snapshot()

    @app.post("/_fault")
    def set_fault(payload: FaultRequest) -> dict:
        configure(payload)
        return snapshot()

    @app.post("/_reset")
    def do_reset() -> dict:
        reset_fault()
        data.reset()
        return {"status": "reset"}

    return app


app = create_app()
