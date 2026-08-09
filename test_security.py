"""
Tests for the API-layer concerns added in Phase 2: API key auth, rate
limiting, and the /health/ endpoint.

Deliberately a SEPARATE file from test.py: test.py checks RAG *correctness*
(is the answer right?) and needs a populated DB plus a live Ollama - it's
slow and can only run where Ollama is installed. This file checks the HTTP
*contract* (auth/limits/shape) using FastAPI's TestClient, with the vector
DB and the LLM replaced by fakes. That makes it fast (milliseconds, not
minutes) and runnable in plain CI with no GPU/Ollama - a normal split
between "does the ML pipeline give good answers" and "does the API behave."
"""
import os

# Must be set BEFORE importing rag_api, because rag_api reads API_KEY from
# the environment once at import time (via load_dotenv() + os.getenv()).
# setdefault() means a real .env value (if present) always wins - this is
# only a fallback for environments with no .env file at all (e.g. CI).
os.environ.setdefault("API_KEY", "test-secret-key")

import pytest
from fastapi.testclient import TestClient

import rag_api

API_KEY = os.environ["API_KEY"]


class _FakeDoc:
    """Stands in for a LangChain Document without needing a real DB."""
    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = metadata


class _FakeDB:
    def similarity_search_with_score(self, query, k=5):
        return [(_FakeDoc("fake context chunk", {"id": "fake.pdf:0:0"}), 0.1)]


class _FakeModel:
    """Fake Ollama LLM. Phase 3 added a second LLM call inside /query/ (the
    context-sufficiency check, which asks a yes/no question with a
    distinctive instruction string) before the final answer generation
    call. This has to answer that one with "YES" - otherwise the (real,
    unmocked) Corrective RAG logic would decide the context is
    insufficient and reach for the real Tavily web-search call, which
    would make these "fast, offline, mocked" tests silently hit the
    network. See test_corrective_rag.py for tests that exercise the
    fallback path on purpose, with web_search() itself mocked too.
    """
    def invoke(self, prompt):
        if "Answer with exactly one word" in prompt:
            return "YES"
        return "fake answer"


@pytest.fixture(autouse=True)
def _patch_heavy_dependencies(monkeypatch):
    """Swap the real Chroma client and Ollama LLM for fakes, and reset
    slowapi's in-memory rate-limit counters before every test so one test's
    requests can't push another test over its limit."""
    monkeypatch.setattr(rag_api, "get_db", lambda: _FakeDB())
    monkeypatch.setattr(rag_api, "Ollama", lambda model=None: _FakeModel())
    rag_api.limiter.reset()


client = TestClient(rag_api.app)


def test_health_endpoint_is_public():
    """/health/ must work with no API key - it's polled by infrastructure
    (Docker healthchecks, uptime monitors) that won't send one."""
    response = client.get("/health/")
    assert response.status_code == 200
    assert "status" in response.json()


def test_query_without_api_key_is_rejected():
    response = client.post("/query/", json={"query_text": "hello"})
    assert response.status_code == 401


def test_populate_without_api_key_is_rejected():
    response = client.post("/populate/", json={"reset": False})
    assert response.status_code == 401


def test_query_with_valid_api_key_succeeds():
    response = client.post(
        "/query/",
        json={"query_text": "hello"},
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 200
    assert response.json()["response"] == "fake answer"


def test_query_rate_limit_returns_429_after_limit_exceeded():
    # /query/ is capped at 15/minute (see @limiter.limit in rag_api.py).
    # Fire exactly that many (all should succeed), then one more that
    # should be throttled.
    headers = {"X-API-Key": API_KEY}
    for _ in range(15):
        response = client.post("/query/", json={"query_text": "hello"}, headers=headers)
        assert response.status_code == 200

    response = client.post("/query/", json={"query_text": "hello"}, headers=headers)
    assert response.status_code == 429
