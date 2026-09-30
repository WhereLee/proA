"""Audit log for human confirmations (checklist P0: 确认记录持久化可审计)."""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ConfirmationRecord:
    id: str
    run_id: str
    tool: str
    args: dict[str, Any]
    created_at: str
    decision: str | None = None  # approved | rejected
    decided_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AuditLog:
    def __init__(self) -> None:
        self._records: list[ConfirmationRecord] = []

    def request(self, run_id: str, tool: str, args: dict[str, Any]) -> ConfirmationRecord:
        record = ConfirmationRecord(
            id=uuid.uuid4().hex,
            run_id=run_id,
            tool=tool,
            args=args,
            created_at=_iso(),
        )
        self._records.append(record)
        return record

    def decide(self, run_id: str, approved: bool) -> None:
        decision = "approved" if approved else "rejected"
        for record in reversed(self._records):
            if record.run_id == run_id and record.decision is None:
                record.decision = decision
                record.decided_at = _iso()
                return

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self._records[-limit:][::-1]]


audit = AuditLog()
