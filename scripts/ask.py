"""Ask the agent a single question from the CLI (demo / smoke utility).

    python scripts/ask.py "退款需要满足什么条件？"
    python scripts/ask.py --domain data "一共有多少张订单？"
    python scripts/ask.py --role customer "帮我退款 O1002"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from app.agent.context import ContextManager  # noqa: E402
from app.agent.loop import AgentRunner  # noqa: E402
from app.config import settings  # noqa: E402
from app.domains import get_domain  # noqa: E402
from app.llm.runtime import get_llm  # noqa: E402
from app.memory import MemoryManager  # noqa: E402


def _build_registry(domain: str):
    if domain == "tickets":
        from mocks.app import app as mock_app

        return get_domain("tickets").build_registry(
            transport=httpx.ASGITransport(app=mock_app)
        )
    return get_domain(domain).build_registry()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the agent one question")
    parser.add_argument("question")
    parser.add_argument("--domain", default="tickets")
    parser.add_argument("--role", default="agent")
    args = parser.parse_args()

    domain = get_domain(args.domain)
    runner = AgentRunner(
        get_llm(),
        _build_registry(args.domain),
        system_prompt_template=domain.system_prompt_template,
        max_steps=domain.max_steps or settings.agent.max_steps,
        context_manager=ContextManager(max_tokens=settings.agent.context_max_tokens),
        memory=MemoryManager(),
        memory_domain=args.domain,
    )
    run = asyncio.run(runner.run(args.question, role=args.role))

    print(f"status={run.status} stopped={run.stopped_reason}")
    print(f"answer: {run.answer}")
    for step in run.steps:
        if step.action == "tool":
            print(f"  - {step.tool}({json.dumps(step.args, ensure_ascii=False)})"
                  f" -> {'ERROR: ' + (step.error or '') if step.error else 'ok'}")


if __name__ == "__main__":
    main()
