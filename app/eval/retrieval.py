"""Retrieval evaluation: Recall@k and MRR for the RAG knowledge base.

    python -m app.eval.retrieval --dataset eval/datasets/rag_v1.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.memory import default_knowledge


def load_dataset(path: str | Path) -> list[dict]:
    tasks = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            tasks.append(json.loads(line))
    return tasks


def evaluate_retrieval(
    kb, tasks: list[dict], *, ks: tuple[int, ...] = (1, 3, 5)
) -> dict:
    recalls = {k: 0 for k in ks}
    reciprocal_rank = 0.0
    total = len(tasks)
    for task in tasks:
        hits = kb.search(task["query"], k=max(ks), scope=task.get("scope"))
        docs = [hit["doc_id"] for hit in hits]
        expected = task["expected_doc"]
        for k in ks:
            if expected in docs[:k]:
                recalls[k] += 1
        if expected in docs:
            reciprocal_rank += 1.0 / (docs.index(expected) + 1)
    if not total:
        return {"total": 0, "recall": {}, "mrr": 0.0}
    return {
        "total": total,
        "recall": {k: round(recalls[k] / total, 4) for k in ks},
        "mrr": round(reciprocal_rank / total, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG retrieval eval (Recall@k / MRR)")
    parser.add_argument("--dataset", default="eval/datasets/rag_v1.jsonl")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = evaluate_retrieval(default_knowledge(), load_dataset(args.dataset))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        recall = " ".join(f"R@{k}={v:.0%}" for k, v in report["recall"].items())
        print(f"total={report['total']} {recall} MRR={report['mrr']:.3f}")


if __name__ == "__main__":
    main()
