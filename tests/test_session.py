from app.agent.session import SessionStore


def test_session_isolation():
    store = SessionStore()
    store.record_turn("s1", "你好", "你好，有什么可以帮您")
    assert len(store.history("s1")) == 2
    assert store.history("s2") == []


def test_reset_clears_history():
    store = SessionStore()
    store.record_turn("s1", "hi", "hello")
    store.reset("s1")
    assert store.history("s1") == []
