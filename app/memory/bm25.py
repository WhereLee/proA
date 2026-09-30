"""Tiny pure-Python BM25 for offline lexical retrieval (Chinese + ASCII).

No external embedding dependency: Chinese is tokenized into unigrams + bigrams,
ASCII into lowercased words. Swap for dense vectors (pgvector/Milvus) in prod.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_ASCII_WORD = re.compile(r"[a-zA-Z0-9_]+")
_CJK = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    text = (text or "").lower()
    tokens = _ASCII_WORD.findall(text)
    cjk = _CJK.findall(text)
    tokens.extend(cjk)
    tokens.extend(cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1))
    return tokens


class BM25:
    def __init__(self, docs: list[list[str]], *, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._tf = [Counter(doc) for doc in docs]
        self._doc_len = [len(doc) for doc in docs]
        self._avgdl = (sum(self._doc_len) / len(docs)) if docs else 0.0
        df: Counter[str] = Counter()
        for doc in docs:
            for token in set(doc):
                df[token] += 1
        n = len(docs)
        self._idf = {
            token: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for token, freq in df.items()
        }

    def scores(self, query_tokens: list[str]) -> list[float]:
        result: list[float] = []
        for tf, dl in zip(self._tf, self._doc_len):
            score = 0.0
            length = dl or 1
            for token in query_tokens:
                freq = tf.get(token, 0)
                if not freq:
                    continue
                idf = self._idf.get(token, 0.0)
                denom = freq + self._k1 * (
                    1 - self._b + self._b * length / (self._avgdl or 1)
                )
                score += idf * (freq * (self._k1 + 1)) / denom
            result.append(score)
        return result


def rank(query: str, texts: list[str], k: int = 3) -> list[tuple[int, float]]:
    """Return [(index, score)] for the top-k texts, best first."""
    if not texts:
        return []
    bm25 = BM25([tokenize(t) for t in texts])
    scored = list(enumerate(bm25.scores(tokenize(query))))
    scored = [(i, s) for i, s in scored if s > 0]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[: max(1, k)]
