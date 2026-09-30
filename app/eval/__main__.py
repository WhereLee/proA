"""Eval CLI + regression gate.

    python -m app.eval --limit 10                 # run against configured LLM
    python -m app.eval --oracle --in-process-mock --min-pass-rate 0.9
    python -m app.eval --json

--oracle uses a deterministic replay LLM to self-check the harness (no API key,
no running mock server needed with --in-process-mock). Its pass rate is a
harness sanity check, NOT a model-quality metric.
"""
from __future__ import annotations

import argparse
import asyncio
import json

import httpx

from app.config import settings
from app.eval.gate import check_gate
from app.eval.oracle import build_oracle_llm
from app.eval.runner import evaluate, load_dataset
from app.llm.runtime import get_llm
from app.tools import MockClient, build_registry


def _build_registry(in_process: bool):
    if in_process:
        from mocks.app import app as mock_app

        client = MockClient(
            "http://mock",
            timeout=settings.agent.tool_timeout,
            transport=httpx.ASGITransport(app=mock_app),
        )
    else:
        client = MockClient(settings.mock_base_url, timeout=settings.agent.tool_timeout)
    return build_registry(client, include_writes=True)


def _resolve_registry(domain: str, in_process: bool):
    if domain != "tickets":
        from app.domains import get_domain

        return get_domain(domain).build_registry()
    return _build_registry(in_process)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agent eval harness")
    parser.add_argument("--dataset", default="eval/datasets/readonly_v1.jsonl")
    parser.add_argument(
        "--domain", choices=["tickets", "data", "research"], default="tickets"
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="print full JSON report")
    parser.add_argument("--oracle", action="store_true", help="use the deterministic oracle LLM")
    parser.add_argument(
        "--in-process-mock", action="store_true", help="call mocks in-process (no server)"
    )
    parser.add_argument("--judge", action="store_true", help="also run the LLM-judge grader")
    parser.add_argument(
        "--reset-state", action="store_true", help="reset mock data/faults per task (v2 dataset)"
    )
    parser.add_argument("--min-pass-rate", type=float, default=None)
    parser.add_argument("--baseline-pass-rate", type=float, default=None)
    parser.add_argument("--max-regression", type=float, default=0.0)
    args = parser.parse_args()

    registry = _resolve_registry(args.domain, args.in_process_mock)
    domain = None
    if args.domain != "tickets":
        from app.domains import get_domain

        domain = get_domain(args.domain)
    max_steps = domain.max_steps if domain and domain.max_steps else settings.agent.max_steps

    if args.oracle:
        tasks = load_dataset(args.dataset)
        if args.limit:
            tasks = tasks[: args.limit]
        llm = build_oracle_llm(tasks)
    else:
        llm = get_llm()

    judge_llm = get_llm() if args.judge else None

    report = asyncio.run(
        evaluate(
            args.dataset,
            llm=llm,
            registry=registry,
            max_steps=max_steps,
            limit=args.limit,
            judge_llm=judge_llm,
            reset_state=args.reset_state,
        )
    )

    provider = "oracle" if args.oracle else settings.llm.provider

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(
            f"provider={provider} total={report.total} passed={report.passed} "
            f"pass_rate={report.pass_rate:.1%}"
        )
        if report.failure_summary:
            print(f"failures={json.dumps(report.failure_summary, ensure_ascii=False)}")
        for result in report.results:
            flag = "PASS" if result["passed"] else "FAIL"
            extra = f" [{result['category']}]" if result["category"] else ""
            print(f"[{flag}]{extra} {result['task_id']}: {result['input']}")

    if args.min_pass_rate is not None:
        gate = check_gate(
            report,
            min_pass_rate=args.min_pass_rate,
            baseline_pass_rate=args.baseline_pass_rate,
            max_regression=args.max_regression,
        )
        print(f"gate={'PASS' if gate.passed else 'FAIL'} ({gate.reason})")
        if not gate.passed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
