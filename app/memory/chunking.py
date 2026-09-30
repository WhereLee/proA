"""Sentence/structure-aware chunking (replaces fixed-window splitting)."""
from __future__ import annotations

import re

_SENT_SPLIT = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.findall(text or "") if s.strip()]


def chunk_document(text: str, *, max_chars: int = 240, overlap: int = 1) -> list[str]:
    """Pack sentences into chunks of <= max_chars, overlapping by N sentences."""
    sentences = split_sentences(text)
    if not sentences:
        return [text.strip()] if text and text.strip() else []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        if current and current_len + len(sentence) > max_chars:
            chunks.append("".join(current))
            current = current[-overlap:] if overlap > 0 else []
            current_len = sum(len(s) for s in current)
        current.append(sentence)
        current_len += len(sentence)
    if current:
        chunks.append("".join(current))
    return chunks


# Backwards-compatible alias
chunk_text = chunk_document
