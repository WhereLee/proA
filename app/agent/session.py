"""Session store (phase 3: persisted to SQLite, session-isolated)."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.config import settings
from app.llm.base import ChatMessage
from app.persistence import Database, get_default_db


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Session:
    session_id: str
    messages: list[ChatMessage] = field(default_factory=list)
    created_at: str = field(default_factory=_iso)
    updated_at: str = field(default_factory=_iso)


class SessionStore:
    def __init__(self, db: Database | None = None, *, cap: int = 500) -> None:
        self._cache: dict[str, Session] = {}
        self._db = db
        self._cap = cap

    def _persist(self, session: Session) -> None:
        if self._db is None:
            return
        self._db.upsert_session(
            session.session_id,
            {
                "session_id": session.session_id,
                "messages": [{"role": m.role, "content": m.content} for m in session.messages],
                "created_at": session.created_at,
                "updated_at": session.updated_at,
            },
            session.updated_at,
        )

    def get_or_create(self, session_id: str | None = None) -> Session:
        sid = session_id or uuid.uuid4().hex
        session = self._cache.get(sid)
        if session is None:
            session = self._load(sid)
            self._cache[sid] = session
        return session

    def _load(self, session_id: str) -> Session:
        if self._db is not None:
            data = self._db.get_session(session_id)
            if data is not None:
                return Session(
                    session_id=data["session_id"],
                    messages=[
                        ChatMessage(m["role"], m["content"]) for m in data["messages"]
                    ],
                    created_at=data.get("created_at", _iso()),
                    updated_at=data.get("updated_at", _iso()),
                )
        return Session(session_id=session_id)

    def history(self, session_id: str) -> list[ChatMessage]:
        session = self._cache.get(session_id) or self.get_or_create(session_id)
        return list(session.messages)

    def record_turn(self, session_id: str, user_input: str, answer: str | None) -> None:
        session = self.get_or_create(session_id)
        session.messages.append(ChatMessage("user", user_input))
        if answer:
            session.messages.append(ChatMessage("assistant", answer))
        session.updated_at = _iso()
        self._persist(session)
        if self._db is not None:
            self._db.evict_sessions(self._cap)
        if len(self._cache) > self._cap:
            for key in list(self._cache)[: len(self._cache) - self._cap]:
                self._cache.pop(key, None)

    def reset(self, session_id: str) -> None:
        self._cache.pop(session_id, None)
        if self._db is not None:
            self._db.delete_session(session_id)


sessions = SessionStore(get_default_db(), cap=settings.session_store_limit)
