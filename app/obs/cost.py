"""Per-run cost accounting and budget circuit breaker (checklist P0: 成本治理)."""
from __future__ import annotations

from dataclasses import dataclass

from app.llm.base import Usage


@dataclass
class CostTracker:
    prompt_per_1k: float = 0.005
    completion_per_1k: float = 0.015
    budget_usd: float = 0.0  # 0 = unlimited
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, usage: Usage) -> float:
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.cost_usd += (
            usage.prompt_tokens / 1000.0 * self.prompt_per_1k
            + usage.completion_tokens / 1000.0 * self.completion_per_1k
        )
        return self.cost_usd

    def exceeded(self) -> bool:
        return self.budget_usd > 0 and self.cost_usd > self.budget_usd

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "budget_usd": self.budget_usd,
        }
