"""Runtime selection of the active LLM (overridable in tests)."""
from __future__ import annotations

from app.config import settings

from .fake import FakeLLM
from .provider import OpenAICompatibleLLM
from .resilient import ResilientLLM

_llm: ResilientLLM | None = None


def build_llm() -> ResilientLLM:
    cfg = settings.llm
    if cfg.provider == "openai":
        inner = OpenAICompatibleLLM(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            timeout=cfg.timeout,
            auth_header=cfg.auth_header,
            auth_prefix=cfg.auth_prefix,
            extra_body=cfg.extra_body,
            default_temperature=cfg.temperature,
        )
    else:
        inner = FakeLLM()
    return ResilientLLM(
        inner,
        cfg.models,
        max_retries=cfg.max_retries,
        backoff_base=cfg.backoff_base,
        timeout=cfg.timeout,
    )


def get_llm() -> ResilientLLM:
    global _llm
    if _llm is None:
        _llm = build_llm()
    return _llm


def set_llm(instance: ResilientLLM | None) -> None:
    """Override the active LLM (used by tests). Pass None to reset."""
    global _llm
    _llm = instance
