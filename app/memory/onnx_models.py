"""Local ONNX embedding + cross-encoder reranker (offline, no API).

Uses the models under D:\\models\\onnx-int8 (bge-base-zh / bge-reranker-base).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _resolve_onnx(model_path: str) -> str:
    path = Path(model_path)
    if path.is_dir():
        return str(path / "model.onnx")
    return str(path)


def _load_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    path = model_path if Path(model_path).is_dir() else str(Path(model_path).parent)
    try:
        return AutoTokenizer.from_pretrained(
            path, local_files_only=True, fix_mistral_regex=True
        )
    except TypeError:
        return AutoTokenizer.from_pretrained(path, local_files_only=True)


class OnnxEmbedder:
    def __init__(self, model_path: str, *, dim: int = 768, max_length: int = 512) -> None:
        import onnxruntime as ort

        self.dim = dim
        self._max_length = max_length
        self._tokenizer = _load_tokenizer(model_path)
        self._session = ort.InferenceSession(
            _resolve_onnx(model_path), providers=["CPUExecutionProvider"]
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        encoded = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="np",
        )
        feeds = {
            item.name: encoded[item.name].astype(np.int64)
            for item in self._session.get_inputs()
            if item.name in encoded
        }
        hidden = self._session.run(None, feeds)[0]  # (batch, seq, dim)
        cls = hidden[:, 0]  # BGE uses the CLS token
        norms = np.linalg.norm(cls, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (cls / norms).tolist()


class OnnxCrossEncoderReranker:
    name = "onnx"

    def __init__(self, model_path: str, *, max_length: int = 512) -> None:
        import onnxruntime as ort

        self._max_length = max_length
        self._tokenizer = _load_tokenizer(model_path)
        self._session = ort.InferenceSession(
            _resolve_onnx(model_path), providers=["CPUExecutionProvider"]
        )

    async def rerank(
        self, query: str, candidates: list[dict[str, Any]], k: int
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        pairs = [
            (query, f"{c.get('title', '')} {c.get('text', '')}".strip())
            for c in candidates
        ]
        encoded = self._tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="np",
        )
        feeds = {
            item.name: encoded[item.name].astype(np.int64)
            for item in self._session.get_inputs()
            if item.name in encoded
        }
        logits = np.asarray(self._session.run(None, feeds)[0]).reshape(-1)
        order = sorted(range(len(candidates)), key=lambda i: float(logits[i]), reverse=True)
        return [candidates[i] for i in order[:k]]
