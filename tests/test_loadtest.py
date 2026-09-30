import asyncio
import json

import httpx

from app.bench.loadtest import percentile, run_loadtest
from app.llm.scripted import ScriptedLLM
from app.tools import MockClient, build_readonly_registry

_ORDERS = ["O1001", "O1002", "O1003", "O1004", "O1005"]


def test_percentile():
    assert percentile([], 0.5) == 0.0
    assert percentile([10.0], 0.95) == 10.0
    assert percentile([1, 2, 3, 4], 0.5) == 2.5


def test_loadtest_bounds_concurrency(mock_app):
    client = MockClient(
        "http://mock", timeout=2.0, transport=httpx.ASGITransport(app=mock_app)
    )
    registry = build_readonly_registry(client)

    inputs: list[str] = []
    script: dict[str, list[str]] = {}
    for i in range(12):
        order_id = _ORDERS[i % len(_ORDERS)]
        user_input = f"查询订单 {order_id} #{i}"
        inputs.append(user_input)
        script[user_input] = [
            json.dumps(
                {"action": "tool", "tool": "get_order", "args": {"order_id": order_id}},
                ensure_ascii=False,
            ),
            json.dumps({"action": "final", "answer": "完成"}, ensure_ascii=False),
        ]

    report = asyncio.run(
        run_loadtest(inputs, registry, ScriptedLLM(script), concurrency=3)
    )
    assert report["runs"] == 12
    assert report["completed"] == 12
    assert report["max_concurrent"] <= 3
    assert report["latency_ms"]["p95"] <= report["latency_ms"]["max"]
    assert "cost_usd" in report
