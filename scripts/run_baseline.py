"""Run a full agent eval baseline against the configured real LLM.

    python scripts/run_baseline.py                 # all tasks, rule graders
    python scripts/run_baseline.py --limit 8
    python scripts/run_baseline.py --judge         # + LLM-judge (slower)

Writes baseline_report.json (UTF-8) and prints a summary.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from app.config import settings  # noqa: E402
from app.eval.runner import evaluate  # noqa: E402
from app.llm.runtime import get_llm  # noqa: E402
from app.tools import MockClient, build_registry  # noqa: E402


def _registry(domain: str = "tickets"):
    if domain != "tickets":
        from app.domains import get_domain

        return get_domain(domain).build_registry()

    from mocks.app import app as mock_app

    client = MockClient(
        "http://mock",
        timeout=settings.agent.tool_timeout,
        transport=httpx.ASGITransport(app=mock_app),
    )
    return build_registry(client, include_writes=True)


async def _run(args) -> dict:
    domain = None
    if args.domain != "tickets":
        from app.domains import get_domain

        domain = get_domain(args.domain)
    max_steps = domain.max_steps if domain and domain.max_steps else settings.agent.max_steps
    report = await evaluate(
        args.dataset,
        llm=get_llm(),
        registry=_registry(args.domain),
        max_steps=max_steps,
        limit=args.limit,
        judge_llm=get_llm() if args.judge else None,
        reset_state=args.reset_state,
    )
    return report.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-model agent eval baseline")
    parser.add_argument("--dataset", default="eval/datasets/readonly_v1.jsonl")
    parser.add_argument(
        "--domain", choices=["tickets", "data", "research"], default="tickets"
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument(
        "--reset-state", action="store_true", help="reset mock data/faults per task (v2 dataset)"
    )
    parser.add_argument("--out", default="baseline_report.json")
    args = parser.parse_args()

    report = asyncio.run(_run(args))
    Path(args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"model={settings.llm.model} total={report['total']} "
        f"passed={report['passed']} pass_rate={report['pass_rate']:.1%}"
    )
    print("failure_summary=" + json.dumps(report["failure_summary"], ensure_ascii=False))
    for result in report["results"]:
        if not result["passed"]:
            print(
                f"FAIL [{result['category']}] {result['task_id']} | "
                f"answer={result['answer']!r} | judge={result.get('judge')}"
            )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
