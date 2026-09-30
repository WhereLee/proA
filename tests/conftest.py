import httpx
import pytest

from app.tools import MockClient, build_readonly_registry, build_registry
from mocks import data
from mocks.app import create_app as create_mock_app
from mocks.faults import reset as reset_fault


@pytest.fixture(autouse=True)
def _reset_mock_state():
    reset_fault()
    data.reset()
    yield
    reset_fault()
    data.reset()


@pytest.fixture(autouse=True)
def _hermetic_knowledge(tmp_path):
    """Give every test an isolated SQLite+hashing knowledge base so the suite
    never touches PostgreSQL or loads ONNX models, regardless of .env."""
    from app.memory import HashingEmbedder, KnowledgeBase, set_knowledge
    from app.memory.store import SqliteKnowledgeStore
    from app.persistence import Database

    kb = KnowledgeBase(
        store=SqliteKnowledgeStore(Database(tmp_path / "kb.db")),
        embedder=HashingEmbedder(),
    )
    set_knowledge(kb)
    yield
    set_knowledge(None)


@pytest.fixture
def mock_app():
    return create_mock_app()


@pytest.fixture
def readonly_registry(mock_app):
    transport = httpx.ASGITransport(app=mock_app)
    client = MockClient("http://mock", timeout=2.0, transport=transport)
    return build_readonly_registry(client)


@pytest.fixture
def full_registry(mock_app):
    transport = httpx.ASGITransport(app=mock_app)
    client = MockClient("http://mock", timeout=2.0, transport=transport)
    return build_registry(client, include_writes=True)
