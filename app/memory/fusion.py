"""Reciprocal Rank Fusion for combining multiple retrieval rankings."""
from __future__ import annotations

from typing import Hashable


def rrf_fuse(
    rankings: list[list[Hashable]], *, k: int = 60
) -> list[tuple[Hashable, float]]:
    """Fuse ranked key lists (indexes or ids) into [(key, score)], best first."""
    scores: dict[Hashable, float] = {}
    for ranking in rankings:
        for position, key in enumerate(ranking):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + position + 1)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
