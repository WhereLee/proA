import asyncio

import httpx

from mocks.app import create_app


def _run(coro):
    return asyncio.run(coro)


def _client(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://mock"
    )


def test_business_read_ok():
    async def scenario():
        app = create_app()
        async with _client(app) as c:
            await c.post("/_reset")
            resp = await c.get("/orders/O1001")
            assert resp.status_code == 200
            assert resp.json()["status"] == "delivered"

    _run(scenario())


def test_address_change_rejected_when_shipped():
    async def scenario():
        app = create_app()
        async with _client(app) as c:
            await c.post("/_reset")
            resp = await c.post("/orders/O1002/address", json={"address": "新地址"})
            assert resp.status_code == 409

    _run(scenario())


def test_refund_rules():
    async def scenario():
        app = create_app()
        async with _client(app) as c:
            await c.post("/_reset")
            assert (await c.post("/orders/O1001/refund", json={})).status_code == 200
            assert (await c.post("/orders/O1001/refund", json={})).status_code == 409
            assert (await c.post("/orders/O1004/refund", json={})).status_code == 400
            assert (await c.post("/orders/O1002/refund", json={})).status_code == 400

    _run(scenario())


def test_fault_error_is_bounded_by_remaining():
    async def scenario():
        app = create_app()
        async with _client(app) as c:
            await c.post("/_reset")
            await c.post(
                "/_fault",
                json={"mode": "error", "target": "order", "remaining": 1, "status_code": 503},
            )
            assert (await c.get("/orders/O1001")).status_code == 503
            assert (await c.get("/orders/O1001")).status_code == 200

    _run(scenario())


def test_fault_timeout():
    async def scenario():
        app = create_app()
        async with _client(app) as c:
            await c.post("/_reset")
            await c.post(
                "/_fault",
                json={
                    "mode": "timeout",
                    "target": "order",
                    "remaining": 1,
                    "timeout_seconds": 0.02,
                },
            )
            assert (await c.get("/orders/O1001")).status_code == 504

    _run(scenario())


def test_fault_partial_applies_side_effect_then_fails():
    async def scenario():
        app = create_app()
        async with _client(app) as c:
            await c.post("/_reset")
            await c.post(
                "/_fault", json={"mode": "partial", "target": "order", "remaining": 1}
            )
            resp = await c.post("/orders/O1001/refund", json={})
            assert resp.status_code == 500
            # The side effect WAS applied: a second refund is rejected as duplicate.
            second = await c.post("/orders/O1001/refund", json={})
            assert second.status_code == 409

    _run(scenario())
