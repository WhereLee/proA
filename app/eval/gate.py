"""Regression gate (checklist P0: 回归门禁——指标过阈值才可合并)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GateResult:
    passed: bool
    pass_rate: float
    threshold: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "pass_rate": self.pass_rate,
            "threshold": self.threshold,
            "reason": self.reason,
        }


def check_gate(
    report,
    *,
    min_pass_rate: float,
    baseline_pass_rate: float | None = None,
    max_regression: float = 0.0,
) -> GateResult:
    if report.pass_rate < min_pass_rate:
        return GateResult(
            passed=False,
            pass_rate=report.pass_rate,
            threshold=min_pass_rate,
            reason=(
                f"pass_rate {report.pass_rate:.3f} < min {min_pass_rate:.3f}"
            ),
        )
    if baseline_pass_rate is not None and report.pass_rate < (
        baseline_pass_rate - max_regression
    ):
        return GateResult(
            passed=False,
            pass_rate=report.pass_rate,
            threshold=min_pass_rate,
            reason=(
                f"regression: {report.pass_rate:.3f} < baseline "
                f"{baseline_pass_rate:.3f} - {max_regression:.3f}"
            ),
        )
    return GateResult(
        passed=True,
        pass_rate=report.pass_rate,
        threshold=min_pass_rate,
        reason="ok",
    )
