import pytest

from app.agent.statemachine import InvalidTransition, can_transition, transition


def test_valid_transitions():
    assert transition("pending", "running") == "running"
    assert transition("running", "awaiting_confirmation") == "awaiting_confirmation"
    assert transition("awaiting_confirmation", "running") == "running"
    assert transition("running", "completed") == "completed"
    assert transition("running", "failed") == "failed"
    assert transition("running", "cancelled") == "cancelled"


def test_illegal_transitions():
    for current, new in [
        ("completed", "running"),
        ("pending", "completed"),
        ("cancelled", "running"),
        ("failed", "running"),
    ]:
        assert not can_transition(current, new)
        with pytest.raises(InvalidTransition):
            transition(current, new)
