"""FastAPI application entrypoint: `uvicorn app.main:app --reload`."""
from __future__ import annotations

from fastapi import FastAPI

from app.api import agent, chat, knowledge, research, traces
from app.config import settings


def create_app() -> FastAPI:
    app = FastAPI(title=settings.service_name, version="0.1.0")
    app.include_router(chat.router, tags=["chat"])
    app.include_router(agent.router, tags=["agent"])
    app.include_router(research.router, tags=["research"])
    app.include_router(knowledge.router, tags=["knowledge"])
    app.include_router(traces.router, tags=["traces"])

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "service": settings.service_name,
            "llm_provider": settings.llm.provider,
            "models": settings.llm.models,
        }

    return app


app = create_app()
