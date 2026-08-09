"""
Tests for the Corrective RAG decision logic added in Phase 3: the
similarity-threshold check, the LLM sufficiency check, and the web-search
fallback merge in rag_api.py's /query/ endpoint.

Same pattern as test_security.py: FastAPI's TestClient with the vector DB,
LLM, and web_search() all replaced by fakes/mocks, so these run in
milliseconds with no live Ollama and - importantly for this file - no real
Tavily API calls either. Kept in its own file (not test_security.py)
because it's testing RAG *decision logic*, not the auth/rate-limit layer -
a different concern deserves a different file.
"""
import os

os.environ.setdefault("API_KEY", "test-secret-key")

import pytest
from fastapi.testclient import TestClient

import rag_api

API_KEY = os.environ["API_KEY"]
HEADERS = {"X-API-Key": API_KEY}


class _FakeDoc:
    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = metadata


class _FakeDB:
    """distance is configurable per-test: with hnsw:space='cosine',
    similarity = 1 - distance, so a small distance means a highly relevant
    (high-similarity) top match, and a large distance means an irrelevant one."""
    def __init__(self, distance):
        self._distance = distance

    def similarity_search_with_score(self, query, k=5):
        return [(_FakeDoc("some local document chunk", {"id": "fake.pdf:0:0"}), self._distance)]


class _YesModel:
    """Always says the context is sufficient; final answer is canned."""
    def invoke(self, prompt):
        if "Answer with exactly one word" in prompt:
            return "YES"
        return "answered from local documents"


class _NoModel:
    """Always says the context is NOT sufficient, regardless of similarity."""
    def invoke(self, prompt):
        if "Answer with exactly one word" in prompt:
            return "NO"
        return "answered with help from the web"


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    rag_api.limiter.reset()


@pytest.fixture
def client():
    return TestClient(rag_api.app)


def test_high_similarity_and_sufficient_context_skips_web_fallback(client, monkeypatch):
    """The common case: a good local match. No web call should happen at
    all - monkeypatching web_search to raise proves it was never called."""
    monkeypatch.setattr(rag_api, "get_db", lambda: _FakeDB(distance=0.1))  # similarity 0.9
    monkeypatch.setattr(rag_api, "Ollama", lambda model=None, **kwargs: _YesModel())
    monkeypatch.setattr(rag_api, "web_search", lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError("web_search should not be called when local context is good")
    ))

    response = client.post("/query/", json={"query_text": "hello"}, headers=HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["answer_source"] == "documents"
    assert body["web_sources"] == []
    assert body["retrieval_debug"]["context_sufficient"] is True
    assert body["retrieval_debug"]["top_similarity"] == 0.9


def test_low_similarity_skips_sufficiency_check_and_falls_back_to_web(client, monkeypatch):
    """Below SIMILARITY_THRESHOLD, the sufficiency LLM call should be
    skipped entirely (context_sufficient stays None) and web fallback
    should fire directly - proven by _NoModel never being asked to check
    sufficiency, only used for the final answer."""
    monkeypatch.setattr(rag_api, "get_db", lambda: _FakeDB(distance=0.9))  # similarity 0.1
    monkeypatch.setattr(rag_api, "Ollama", lambda model=None, **kwargs: _NoModel())
    monkeypatch.setattr(
        rag_api,
        "web_search",
        lambda query, max_results=3: [
            {"title": "Web Result", "snippet": "web snippet content", "url": "https://example.com"}
        ],
    )

    response = client.post("/query/", json={"query_text": "something off-topic"}, headers=HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["answer_source"] == "documents+web"
    assert body["web_sources"] == ["https://example.com"]
    assert body["retrieval_debug"]["context_sufficient"] is None  # check was skipped
    assert body["retrieval_debug"]["top_similarity"] == pytest.approx(0.1)


def test_high_similarity_but_insufficient_context_falls_back_to_web(client, monkeypatch):
    """Similarity alone isn't enough: a topically-close chunk that doesn't
    actually answer the question should still trigger fallback."""
    monkeypatch.setattr(rag_api, "get_db", lambda: _FakeDB(distance=0.1))  # similarity 0.9
    monkeypatch.setattr(rag_api, "Ollama", lambda model=None, **kwargs: _NoModel())
    monkeypatch.setattr(
        rag_api,
        "web_search",
        lambda query, max_results=3: [
            {"title": "Web Result", "snippet": "web snippet content", "url": "https://example.com"}
        ],
    )

    response = client.post("/query/", json={"query_text": "hello"}, headers=HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["answer_source"] == "documents+web"
    assert body["retrieval_debug"]["context_sufficient"] is False  # check DID run this time


def test_web_fallback_failure_degrades_to_local_context_instead_of_500(client, monkeypatch):
    """A broken/missing Tavily key shouldn't break the whole request - it
    should just mean "answer from local context alone", per
    _run_web_fallback's docstring."""
    monkeypatch.setattr(rag_api, "get_db", lambda: _FakeDB(distance=0.9))  # similarity 0.1, needs fallback
    monkeypatch.setattr(rag_api, "Ollama", lambda model=None, **kwargs: _YesModel())

    def _broken_web_search(*args, **kwargs):
        raise RuntimeError("TAVILY_API_KEY is not set")

    monkeypatch.setattr(rag_api, "web_search", _broken_web_search)

    response = client.post("/query/", json={"query_text": "hello"}, headers=HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["answer_source"] == "documents"  # fell back to local-only, not an error
    assert body["web_sources"] == []
