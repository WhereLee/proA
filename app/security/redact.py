"""Secret redaction for logs and traces (checklist P0: trace 脱敏)."""
from __future__ import annotations

import re
from typing import Any

_REDACTIONS = [
    (re.compile(r"sk-[A-Za-z0-9_\-]{8,}"), "sk-***REDACTED***"),
    (re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)\S+"), r"\1***REDACTED***"),
    (re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)\S+"), r"\1***REDACTED***"),
    (re.compile(r"(?i)(token\s*[:=]\s*)\S+"), r"\1***REDACTED***"),
]


def redact(text: str) -> str:
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def redact_value(value: Any) -> Any:
    """Recursively redact strings inside dict/list structures."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: redact_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(redact_value(v) for v in value)
    return value
