"""Generate the phase-1 read-only task dataset from the mock seed data.

    python eval/generate_dataset.py

Output: eval/datasets/readonly_v1.jsonl  (task-level, one JSON per line)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mocks.data import _seed_orders, _seed_users  # noqa: E402
from mocks.services import _LOGISTICS_TEXT  # noqa: E402

OUT = Path(__file__).resolve().parent / "datasets" / "readonly_v1.jsonl"

ORDER_PHRASINGS = [
    "帮我查一下订单 {oid}",
    "订单 {oid} 现在是什么状态？",
    "查订单 {oid} 的详情",
    "我想看看订单 {oid} 的信息",
]
LOGISTICS_PHRASINGS = [
    "订单 {oid} 的物流到哪了？",
    "帮我查一下 {oid} 的快递",
    "订单 {oid} 发货了吗？",
]
USER_PHRASINGS = [
    "查询用户 {uid} 的信息",
    "{uid} 是什么角色？",
    "帮我看看用户 {uid}",
]
COMBO_PHRASINGS = ["帮我查订单 {oid} 和它的物流信息"]


def build() -> list[dict]:
    tasks: list[dict] = []
    counter = 0

    for oid, order in _seed_orders().items():
        for tpl in ORDER_PHRASINGS:
            counter += 1
            tasks.append(
                {
                    "task_id": f"order-get-{counter:03d}",
                    "input": tpl.format(oid=oid),
                    "expected_tools": ["get_order"],
                    "expected_observations": [
                        {
                            "tool": "get_order",
                            "args": {"order_id": oid},
                            "path": "status",
                            "equals": order["status"],
                        }
                    ],
                }
            )
        for tpl in LOGISTICS_PHRASINGS:
            counter += 1
            tasks.append(
                {
                    "task_id": f"order-logistics-{counter:03d}",
                    "input": tpl.format(oid=oid),
                    "expected_tools": ["get_logistics"],
                    "expected_observations": [
                        {
                            "tool": "get_logistics",
                            "args": {"order_id": oid},
                            "path": "latest",
                            "equals": _LOGISTICS_TEXT.get(order["status"], "未知"),
                        }
                    ],
                }
            )
        for tpl in COMBO_PHRASINGS:
            counter += 1
            tasks.append(
                {
                    "task_id": f"order-combo-{counter:03d}",
                    "input": tpl.format(oid=oid),
                    "expected_tools": ["get_order", "get_logistics"],
                    "expected_observations": [
                        {
                            "tool": "get_order",
                            "args": {"order_id": oid},
                            "path": "status",
                            "equals": order["status"],
                        }
                    ],
                }
            )

    for uid, user in _seed_users().items():
        for tpl in USER_PHRASINGS:
            counter += 1
            tasks.append(
                {
                    "task_id": f"user-get-{counter:03d}",
                    "input": tpl.format(uid=uid),
                    "expected_tools": ["get_user"],
                    "expected_observations": [
                        {
                            "tool": "get_user",
                            "args": {"user_id": uid},
                            "path": "role",
                            "equals": user["role"],
                        }
                    ],
                }
            )

    return tasks


def main() -> None:
    tasks = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in tasks) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(tasks)} tasks to {OUT}")


if __name__ == "__main__":
    main()
