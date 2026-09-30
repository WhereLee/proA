"""Load-test CLI:

    python -m app.bench --runs 40 --concurrency 8 [--rpm 120]
"""
from __future__ import annotations

import argparse
import asyncio
import json

import httpx

from app.config import settings
from app.llm.scripted import ScriptedLLM
from app.tools import MockClient, build_readonly_registry

from .loadtest import run_loadtest

_ORDERS = ["O1001", "O1002", "O1003", "O1004", "O1005"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent load test")
    parser.add_argument("--runs", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--rpm", type=float, default=0.0, help="simulated model RPM (0=off)")
    args = parser.parse_args()

    from mocks.app import app as mock_app

    client = MockClient(
        "http://mock",
        timeout=settings.agent.tool_timeout,
        transport=httpx.ASGITransport(app=mock_app),
    )
    registry = build_readonly_registry(client)

    inputs: list[str] = []
    script: dict[str, list[str]] = {}
    for i in range(args.runs):
        order_id = _ORDERS[i % len(_ORDERS)]
        user_input = f"查询订单 {order_id} 的详情 #{i}"
        inputs.append(user_input)
        script[user_input] = [
            json.dumps(
                {"action": "tool", "tool": "get_order", "args": {"order_id": order_id}},
                ensure_ascii=False,
            ),
            json.dumps({"action": "final", "answer": "完成"}, ensure_ascii=False),
        ]

    report = asyncio.run(
        run_loadtest(
            inputs,
            registry,
            ScriptedLLM(script),
            concurrency=args.concurrency,
            rpm=args.rpm,
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
