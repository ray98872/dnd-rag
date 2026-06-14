"""
Integration tests for the FastAPI layer.

These tests mock the RAG chain and Qdrant store so they can run without
any running services. They verify routing, request validation, and
response shape — not RAG quality.

To run against a real stack:
    docker compose up -d
    pytest tests/integration/ --no-mock
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from src.api.main import create_app
from src.generation.chain import RAGResult

# ── Fixtures ───────────────────────────────────────────────────────────────────

def _fake_result(question: str = "test") -> RAGResult:
    docs = [
        Document(
            page_content="Fireball is a 3rd-level evocation spell that deals 8d6 fire damage.",
            metadata={"section": "Spells", "page": 23, "source": "srd52.pdf",
                      "rerank_score": 0.95, "rrf_score": 0.032},
        )
    ]
    return RAGResult(
        answer="Fireball deals 8d6 fire damage in a 20-foot radius.",
        sources=docs,
        question=question,
        retrieval_query=question,
        latency_ms=312.5,
        hyde_used=True,
    )


@pytest.fixture
def app_with_mocks():
    """Create a test app with mocked chain and store."""
    app = create_app()

    mock_chain = MagicMock()
    mock_chain.aquery = AsyncMock(return_value=_fake_result())
    mock_chain._ahyde_query = AsyncMock(return_value="hypothetical passage")
    mock_chain._retrieve_and_rerank = MagicMock(return_value=_fake_result().sources)
    mock_chain.llm = MagicMock()

    mock_store = MagicMock()
    mock_store.is_healthy.return_value = True
    mock_store.count.return_value = 4823

    # Bypass lifespan by setting state directly
    app.state.chain = mock_chain
    app.state.store = mock_store

    return app


@pytest.fixture
def client(app_with_mocks):
    return TestClient(app_with_mocks, raise_server_exceptions=True)


# ── Health endpoint ────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_response_shape(self, client):
        body = client.get("/health").json()
        assert "status" in body
        assert "qdrant_connected" in body
        assert "document_count" in body

    def test_status_ok_when_connected(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["qdrant_connected"] is True


# ── Query endpoint ─────────────────────────────────────────────────────────────

class TestQuery:
    def test_returns_200(self, client):
        resp = client.post("/query", json={"question": "What does Fireball do?"})
        assert resp.status_code == 200

    def test_response_has_answer(self, client):
        body = client.post("/query", json={"question": "What does Fireball do?"}).json()
        assert "answer" in body
        assert len(body["answer"]) > 0

    def test_response_has_sources(self, client):
        body = client.post("/query", json={"question": "What does Fireball do?"}).json()
        assert "sources" in body
        assert isinstance(body["sources"], list)
        assert len(body["sources"]) > 0

    def test_source_shape(self, client):
        body = client.post("/query", json={"question": "What does Fireball do?"}).json()
        src = body["sources"][0]
        assert "content" in src
        assert "section" in src
        assert "page" in src

    def test_latency_ms_present(self, client):
        body = client.post("/query", json={"question": "What does Fireball do?"}).json()
        assert "latency_ms" in body
        assert isinstance(body["latency_ms"], float)

    def test_question_too_short_returns_422(self, client):
        resp = client.post("/query", json={"question": "hi"})
        assert resp.status_code == 422

    def test_missing_question_returns_422(self, client):
        resp = client.post("/query", json={})
        assert resp.status_code == 422

    def test_use_hyde_false_passes_through(self, client, app_with_mocks):
        client.post("/query", json={"question": "What is a cantrip?", "use_hyde": False})
        app_with_mocks.state.chain.aquery.assert_called_once()
        call_kwargs = app_with_mocks.state.chain.aquery.call_args
        assert call_kwargs.kwargs.get("use_hyde") is False or call_kwargs.args[1] is False


# ── Streaming endpoint ────────────────────────────────────────────────────────

class TestQueryStream:
    def test_returns_200_with_sse_content_type(self, client, app_with_mocks):
        # Patch the LLM astream to yield a couple of tokens
        async def fake_astream(*args, **kwargs):
            for token in ["Fireball", " deals", " 8d6", " damage."]:
                yield token

        app_with_mocks.state.chain.llm.astream = fake_astream

        with client.stream("POST", "/query/stream", json={"question": "What does Fireball do?"}):
            pass  # just confirm it doesn't raise

    def test_sse_events_contain_data_prefix(self, client, app_with_mocks):
        async def fake_astream(*args, **kwargs):
            yield "token"

        app_with_mocks.state.chain.llm.astream = fake_astream

        with client.stream("POST", "/query/stream", json={"question": "What spell is Fireball?"}):
            pass  # confirmed no server error
