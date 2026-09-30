"""Explicit task state machine (checklist P0: 显式状态与迁移，禁止隐式状态)."""
from __future__ import annotations


class TaskState:
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    FAILED = "failed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


ALL_STATES = {
    TaskState.PENDING,
    TaskState.RUNNING,
    TaskState.AWAITING_CONFIRMATION,
    TaskState.FAILED,
    TaskState.COMPLETED,
    TaskState.CANCELLED,
}

_TRANSITIONS: dict[str, set[str]] = {
    TaskState.PENDING: {TaskState.RUNNING, TaskState.CANCELLED},
    TaskState.RUNNING: {
        TaskState.RUNNING,
        TaskState.AWAITING_CONFIRMATION,
        TaskState.COMPLETED,
        TaskState.FAILED,
        TaskState.CANCELLED,
    },
    TaskState.AWAITING_CONFIRMATION: {
        TaskState.RUNNING,
        TaskState.FAILED,
        TaskState.CANCELLED,
    },
    TaskState.FAILED: set(),
    TaskState.COMPLETED: set(),
    TaskState.CANCELLED: set(),
}


class InvalidTransition(Exception):
    pass


def can_transition(current: str, new: str) -> bool:
    return new in _TRANSITIONS.get(current, set())


def transition(current: str, new: str) -> str:
    if not can_transition(current, new):
        raise InvalidTransition(f"illegal transition: {current} -> {new}")
    return new
