"""Build the retrieval stack (embedder / store / reranker) from settings."""
from __future__ import annotations

from app.config import settings
from app.persistence import get_default_db

from .embedding import HashingEmbedder
from .knowledge import KnowledgeBase
from .rerank import LLMReranker
from .store import SqliteKnowledgeStore


def build_embedder(cfg=None):
    cfg = cfg or settings.retrieval
    if cfg.embedding_provider == "onnx" and cfg.embedding_model_path:
        from .onnx_models import OnnxEmbedder

        return OnnxEmbedder(cfg.embedding_model_path, dim=cfg.embedding_dim)
    return HashingEmbedder()


def build_store(cfg=None):
    cfg = cfg or settings.retrieval
    if cfg.vector_store == "pgvector":
        from .pg_store import PgVectorStore

        return PgVectorStore(cfg.pg_dsn, dim=cfg.embedding_dim)
    return SqliteKnowledgeStore(get_default_db())


def build_reranker(cfg=None, *, llm=None):
    cfg = cfg or settings.retrieval
    if cfg.reranker == "onnx" and cfg.reranker_model_path:
        from .onnx_models import OnnxCrossEncoderReranker

        return OnnxCrossEncoderReranker(cfg.reranker_model_path)
    if cfg.reranker == "llm" and llm is not None:
        return LLMReranker(llm)
    return None


def build_knowledge(cfg=None) -> KnowledgeBase:
    return KnowledgeBase(store=build_store(cfg), embedder=build_embedder(cfg))
