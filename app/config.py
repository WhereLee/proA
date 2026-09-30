"""Application configuration loaded from environment variables (.env supported)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

try:  # optional dependency
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _env_json(name: str, default: Any) -> Any:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class LLMSettings:
    provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "fake").lower())
    api_key: str = field(
        default_factory=lambda: _env("LLM_API_KEY") or _env("OPENAI_API_KEY", "")
    )
    base_url: str = field(
        default_factory=lambda: _env("LLM_BASE_URL")
        or _env("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    model: str = field(default_factory=lambda: _env("LLM_MODEL", "gpt-4o-mini"))
    fallback_models: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            m.strip() for m in _env("LLM_FALLBACK_MODELS", "").split(",") if m.strip()
        )
    )
    timeout: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT", 30.0))
    max_retries: int = field(default_factory=lambda: _env_int("LLM_MAX_RETRIES", 2))
    backoff_base: float = field(default_factory=lambda: _env_float("LLM_BACKOFF_BASE", 0.2))
    temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.2))
    auth_header: str = field(default_factory=lambda: _env("LLM_AUTH_HEADER", "Authorization"))
    auth_prefix: str = field(default_factory=lambda: _env("LLM_AUTH_PREFIX", "Bearer "))
    # Provider-specific extras merged into every request body, e.g.
    # LLM_EXTRA_BODY={"thinking":{"type":"disabled"}}
    extra_body: dict[str, Any] = field(default_factory=lambda: _env_json("LLM_EXTRA_BODY", {}))

    @property
    def models(self) -> list[str]:
        """Primary model first, then fallbacks."""
        return [self.model, *self.fallback_models]


@dataclass(frozen=True)
class AgentSettings:
    max_steps: int = field(default_factory=lambda: _env_int("AGENT_MAX_STEPS", 6))
    max_repeated_tool_calls: int = field(
        default_factory=lambda: _env_int("AGENT_MAX_REPEATED_TOOL_CALLS", 2)
    )
    context_max_tokens: int = field(
        default_factory=lambda: _env_int("AGENT_CONTEXT_MAX_TOKENS", 6000)
    )
    tool_timeout: float = field(default_factory=lambda: _env_float("AGENT_TOOL_TIMEOUT", 5.0))
    tool_attempts: int = field(default_factory=lambda: _env_int("AGENT_TOOL_ATTEMPTS", 2))
    tool_backoff: float = field(default_factory=lambda: _env_float("AGENT_TOOL_BACKOFF", 0.2))
    max_concurrency: int = field(default_factory=lambda: _env_int("AGENT_MAX_CONCURRENCY", 4))
    max_cost_usd: float = field(default_factory=lambda: _env_float("AGENT_MAX_COST_USD", 0.0))
    # Defaults are Xiaomi MiMo mimo-v2.6-pro overseas rates (cache miss, USD/1k).
    price_prompt_per_1k: float = field(
        default_factory=lambda: _env_float("LLM_PRICE_PROMPT_PER_1K", 0.000435)
    )
    price_completion_per_1k: float = field(
        default_factory=lambda: _env_float("LLM_PRICE_COMPLETION_PER_1K", 0.00087)
    )


@dataclass(frozen=True)
class RetrievalSettings:
    embedding_provider: str = field(
        default_factory=lambda: _env("EMBEDDING_PROVIDER", "hashing").lower()
    )
    embedding_model_path: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL_PATH", "")
    )
    embedding_dim: int = field(default_factory=lambda: _env_int("EMBEDDING_DIM", 768))
    vector_store: str = field(
        default_factory=lambda: _env("VECTOR_STORE", "sqlite").lower()
    )
    pg_dsn: str = field(
        default_factory=lambda: _env(
            "PG_DSN", "postgresql://postgres:root@localhost:5432/agent_kb"
        )
    )
    reranker: str = field(default_factory=lambda: _env("RERANKER", "").lower())
    reranker_model_path: str = field(
        default_factory=lambda: _env("RERANKER_MODEL_PATH", "")
    )


@dataclass(frozen=True)
class Settings:
    service_name: str = field(default_factory=lambda: _env("SERVICE_NAME", "agent-ticket"))
    mock_base_url: str = field(
        default_factory=lambda: _env("MOCK_BASE_URL", "http://localhost:8100")
    )
    db_path: str = field(default_factory=lambda: _env("DB_PATH", "data/agent.db"))
    trace_store_limit: int = field(default_factory=lambda: _env_int("TRACE_STORE_LIMIT", 500))
    run_store_limit: int = field(default_factory=lambda: _env_int("RUN_STORE_LIMIT", 500))
    session_store_limit: int = field(
        default_factory=lambda: _env_int("SESSION_STORE_LIMIT", 500)
    )
    llm: LLMSettings = field(default_factory=LLMSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)


settings = Settings()
