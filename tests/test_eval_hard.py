import asyncio

import httpx

from app.eval.oracle import build_oracle_llm
from app.eval.runner import evaluate, load_dataset
from app.tools import MockClient, build_registry

HARD = "eval/datasets/hard_v2.jsonl"


def _registry(mock_app):
    client = MockClient(
        "http://mock", timeout=2.0, transport=httpx.ASGITransport(app=mock_app)
    )
    return build_registry(client, include_writes=True)


def test_hard_dataset_shape():
    tasks = load_dataset(HARD)
    assert len(tasks) >= 18
    ids = {t["task_id"] for t in tasks}
    assert {"hitl-01", "perm-01", "inj-01", "fault-01", "chain-01"} <= ids


def test_oracle_passes_hard_dataset(mock_app):
    tasks = load_dataset(HARD)
    registry = _registry(mock_app)
    report = asyncio.run(
        evaluate(
            HARD,
            llm=build_oracle_llm(tasks),
            registry=registry,
            reset_state=True,
        )
    )
    failures = [
        (r["task_id"], r["category"], r["details"]) for r in report.results if not r["passed"]
    ]
    assert report.pass_rate == 1.0, failures
