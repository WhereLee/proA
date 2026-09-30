"""Pluggable embeddings for hybrid retrieval.

Default is a deterministic, offline hashing embedder (char n-grams -> fixed
dimensional vector). Swap in a real embedding model in production by
implementing the same `embed()` interface (e.g. an OpenAI-compatible endpoint).
"""
from __future__ import annotations

import hashlib
import math
from typing import Protocol, runtime_checkable

from .bm25 import tokenize


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _hash_index(token: str, dim: int) -> int:
    digest = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % dim


class HashingEmbedder:
    """Offline bag-of-ngrams vectorizer with L2 normalization."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in tokenize(text):
                vec[_hash_index(token, self.dim)] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
