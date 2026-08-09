from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import os
import secrets
import shutil
import sys

# Windows' default console codepage (cp1252) can't encode the emoji used in
# this file's print() calls (e.g. "✨ Clearing..."), which raises
# UnicodeEncodeError and 500s the request the moment reset=True is used.
# Forcing stdout/stderr to UTF-8 fixes this regardless of what codepage the
# terminal launching uvicorn happens to be in. No-op on Linux/macOS, where
# UTF-8 is already the default.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# See requirements.txt for why this is Linux-only: chromadb needs SQLite
# >= 3.35, older Linux base images (our future Docker image) ship less than
# that, and pysqlite3-binary is the drop-in fix - but it has no Windows
# wheel, so we skip the swap when it isn't installed (e.g. local Windows dev).
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass
from langchain.vectorstores import Chroma
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from sklearn.metrics.pairwise import cosine_similarity
from embedding_function import get_embedding_function
from create_db import calculate_chunk_ids
from web_search import web_search

from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import httpx

# Reads .env into os.environ (API_KEY, OLLAMA_BASE_URL, ...) if the file
# exists. In Docker (Phase 4) these are passed as real container env vars
# instead, so this is a no-op there - same code works in both places.
load_dotenv()

# Constants
CHROMA_PATH = "chroma"
# BUG FIX: this was "data", but the PDFs that ship with the repo live in
# docs/ (create_db.py has always pointed there). With "data" the loader
# pointed at a directory that doesn't exist, so /populate/ silently produced
# zero chunks. Matching create_db.py's DATA_PATH here.
DATA_PATH = "docs"
# Ollama's HTTP API address. Configurable because it changes between
# environments: localhost during bare-metal dev, but the Docker Compose
# service name ("http://ollama:11434") once we containerize in Phase 4.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Initialize FastAPI app
app = FastAPI()

# ---------------------------------------------------------------------------
# Rate limiting (slowapi - a FastAPI/Starlette port of the well-known
# Flask-Limiter). Keyed by client IP via get_remote_address(), which is
# fine for a single-instance demo behind no proxy. Behind a real load
# balancer you'd key off a forwarded-for header or the API key itself
# instead, so one noisy tenant sharing an IP/NAT can't exhaust everyone
# else's quota - not a concern for this project, but worth knowing why
# get_remote_address isn't the production-grade choice.
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)  # adds Retry-After / X-RateLimit-* headers

# ---------------------------------------------------------------------------
# API key auth. A single shared-secret key - not per-user accounts/OAuth/JWT
# - is deliberately the simplest thing that could work: this is a demo API
# with one consumer (you, via the Streamlit UI or curl), not a multi-tenant
# product. secrets.compare_digest is used instead of `==` so the comparison
# takes constant time regardless of how many leading characters match,
# closing a timing side-channel that would otherwise let an attacker guess
# the key one byte at a time by measuring response latency.
# ---------------------------------------------------------------------------
API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(provided_key: str | None = Depends(api_key_header)):
    if not API_KEY:
        # Fail CLOSED: if the server has no key configured, refuse every
        # protected request rather than silently running unauthenticated.
        # The opposite default (skip the check when API_KEY is unset) is
        # the more common footgun - it's easy to forget to set the env var
        # in a new environment and not notice the API is wide open until
        # something bad happens.
        raise HTTPException(status_code=500, detail="Server API key is not configured.")
    if not provided_key or not secrets.compare_digest(provided_key, API_KEY):
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")
    return provided_key


# BUG FIX: the original code built ONE Chroma client at import time and
# reused it for the app's entire lifetime. That breaks /populate/ with
# reset=True: clear_database() shutil.rmtree's the chroma/ directory out
# from under a client that still has it open, and on Windows deleting a
# file that's open in the same process either fails outright
# (PermissionError) or leaves the client pointed at now-nonexistent files.
# Building a fresh client per call (matching the pattern already used in
# create_db.py and query.py) avoids holding a long-lived handle at all.
#
# collection_metadata={"hnsw:space": "cosine"} (Phase 3): Chroma's default
# distance function is raw squared L2 (Euclidean) distance - unbounded
# above 0, where the "good" range depends on the embedding model's vector
# magnitudes. That makes a retrieval-quality threshold nearly impossible to
# pick or justify without extensive tuning. Cosine distance is bounded and
# has a standard, portable interpretation (cosine_similarity = 1 - cosine
# distance, roughly 0=unrelated to 1=near-identical for text embeddings),
# which is what SIMILARITY_THRESHOLD below assumes. This only takes effect
# when a collection is first CREATED - changing it later needs a fresh
# /populate/ reset=True, which is why Phase 3 needs one clean re-populate.
def get_db() -> Chroma:
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function(),
        collection_metadata={"hnsw:space": "cosine"},
    )

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# ---------------------------------------------------------------------------
# Corrective RAG (Phase 3) configuration
# ---------------------------------------------------------------------------
# SIMILARITY_THRESHOLD was measured empirically on this corpus (the
# ConocoPhillips proxy statement) with nomic-embed-text + cosine distance,
# not guessed: on-topic queries ("board of directors' compensation",
# "what is the company name?") scored 0.56-0.66 top-chunk similarity;
# clearly off-topic queries ("pizza recipe", "training a puppy") scored
# 0.43-0.49. 0.5 sits in the gap and correctly separated all four samples -
# though the margin was sometimes slim (as little as 0.005), which is a
# realistic finding: embedding similarity is a noisy, topical relevance
# signal, not a precise one. That slim margin is exactly why the LLM
# sufficiency check below exists as a second, more expensive but more
# precise layer, rather than trusting this number alone. A different
# corpus (more/less topically diverse) would likely need a re-tuned
# threshold - hence this being a runtime env var, not a hardcoded literal.
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))

# Lets you turn the web fallback off entirely (e.g. offline dev, or a
# deployment that intentionally never wants to leave the local corpus)
# without touching code.
WEB_FALLBACK_ENABLED = os.getenv("WEB_FALLBACK_ENABLED", "true").lower() == "true"
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "3"))

# Unlike SIMILARITY_THRESHOLD, the sufficiency prompt is a Python constant,
# not an env var. Reasoning: a numeric threshold is an ops knob you might
# reasonably retune per-deployment/corpus without touching code; a prompt
# is program logic whose exact wording changes model behavior and belongs
# in code review and git history, not a silent runtime string swap that
# leaves no diff and no reviewer.
SUFFICIENCY_CHECK_PROMPT_TEMPLATE = """You are checking whether the given context contains enough information to answer the question.

Context:
{context}

Question: {question}

Does the context above contain enough information to answer the question? Answer with exactly one word, "YES" or "NO", and nothing else."""


def _check_context_sufficiency(context_text: str, question: str) -> bool:
    """Ask the LLM whether the retrieved context actually answers the
    question - a check embedding similarity can't make on its own.
    Similarity measures topical closeness ("this chunk is about the same
    subject"), not whether it states the specific fact being asked for -
    e.g. a chunk about "executive compensation policy" can score high
    similarity against "what is the CEO's salary?" while never actually
    naming a number. This is the second, more precise (but slower/costlier)
    layer of the two-stage relevance check.
    """
    prompt = SUFFICIENCY_CHECK_PROMPT_TEMPLATE.format(context=context_text, question=question)
    model = Ollama(model="mistral")
    result = model.invoke(prompt).strip().lower()
    if "yes" in result:
        return True
    if "no" in result:
        return False
    # Rare with a small local model, but if the response is neither, fail
    # toward MORE context rather than less: treat it as insufficient so we
    # augment with web results instead of silently under-answering.
    return False


def _run_web_fallback(query_text: str) -> tuple[str, list[str]]:
    """Fetch web results and format them to match the local-chunk context
    format (joined by "---"), so the final prompt treats both sources
    uniformly. Returns (context_text, source_urls).

    Any failure here (missing/invalid TAVILY_API_KEY, network error,
    Tavily downtime) is caught and degrades to "no web context" rather than
    propagating - a broken web fallback should mean "answer from local
    context alone, if any", not a 500 for the whole /query/ request.
    """
    try:
        results = web_search(query_text, max_results=WEB_SEARCH_MAX_RESULTS)
    except Exception as exc:
        print(f"Web fallback failed, continuing without it: {exc}")
        return "", []

    if not results:
        return "", []

    context_text = "\n\n---\n\n".join(f"{r['title']}\n{r['snippet']}" for r in results)
    urls = [r["url"] for r in results]
    return context_text, urls


# -------------------------------
# 📌 API Models
# -------------------------------
class QueryRequest(BaseModel):
    query_text: str


class CompareRequest(BaseModel):
    query_1: str
    query_2: str


class PopulateDBRequest(BaseModel):
    reset: bool = False


# -------------------------------
# 📌 Database Population
# -------------------------------
def clear_database():
    """Delete existing ChromaDB directory."""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


def load_documents():
    """Load PDFs from the data directory."""
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def populate_database(reset=False):
    """Populate ChromaDB with documents."""
    if reset:
        print("✨ Clearing and Resetting Database")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    db = get_db()  # built AFTER clear_database(), never a stale handle

    # BUG FIX: the original code called db.add_documents(chunks) with no ids,
    # so Chroma auto-assigned a random UUID to every chunk on every call.
    # Calling /populate/ twice (e.g. after uploading a second PDF) duplicated
    # every chunk that was already in the DB, silently degrading retrieval
    # quality (the same passage shows up multiple times, crowding out other
    # real matches in the top-k results). create_db.py already solved this
    # with content-derived, deterministic ids ("source:page:chunk_index") -
    # reusing that here instead of inventing a second dedup scheme.
    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_ids = set(db.get(include=[])["ids"])
    new_chunks = [c for c in chunks_with_ids if c.metadata["id"] not in existing_ids]

    if new_chunks:
        new_ids = [c.metadata["id"] for c in new_chunks]
        db.add_documents(new_chunks, ids=new_ids)
        db.persist()

    return {
        "message": f"Database populated with {len(new_chunks)} new chunks "
                    f"({len(chunks_with_ids) - len(new_chunks)} already existed)"
    }


@app.get("/health/")
async def health_check():
    """Liveness/readiness probe.

    Deliberately excluded from API-key auth and rate limiting: it's meant to
    be polled frequently by infrastructure (Docker's HEALTHCHECK in Phase 4,
    a load balancer, uptime monitoring) that has no API key and shouldn't
    need one just to ask "are you alive?".

    It reports whether Ollama itself is reachable, not just whether this
    FastAPI process is running - that distinction matters because uvicorn
    can be up while Ollama is still starting/loading a model, in which case
    /populate/ and /query/ would fail even though this process looks fine.
    """
    ollama_ok = False
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2.0)
        ollama_ok = response.status_code == 200
    except httpx.RequestError:
        ollama_ok = False

    return {
        "status": "ok" if ollama_ok else "degraded",
        "ollama_reachable": ollama_ok,
    }


@app.post("/populate/")
@limiter.limit("5/minute")  # embedding a whole PDF is CPU-heavy; keep it rare
async def populate_db(request: Request, payload: PopulateDBRequest, _api_key: str = Depends(verify_api_key)):
    """Endpoint to populate the database with documents."""
    return populate_database(reset=payload.reset)


# -------------------------------
# 📌 Query Processing (RAG)
# -------------------------------
@app.post("/query/")
@limiter.limit("15/minute")  # the main hot path, but still LLM-bound - keep modest
async def query_rag(request: Request, payload: QueryRequest, _api_key: str = Depends(verify_api_key)):
    """Search for relevant documents, correct for low-quality retrieval by
    falling back to live web search when needed, and generate a response.

    Corrective RAG flow:
      1. Retrieve top-k chunks + cosine similarity from Chroma.
      2. If the best score is below SIMILARITY_THRESHOLD, skip straight to
         web fallback - running the (slower, LLM-based) sufficiency check
         would just be a redundant extra call to confirm what the numbers
         already show clearly enough.
      3. Otherwise, ask the LLM whether the retrieved context actually
         answers the question - a check similarity alone can't make (see
         _check_context_sufficiency's docstring for why).
      4. If either check fails, fetch live web results and merge them into
         the context before generating the final answer, so a bad/missing
         local match doesn't mean a bad/missing answer.
    """
    results = get_db().similarity_search_with_score(payload.query_text, k=5)

    doc_context = "\n\n---\n\n".join(doc.page_content for doc, _distance in results) if results else ""
    doc_sources = [doc.metadata.get("id", "Unknown") for doc, _distance in results]
    # Chroma returns cosine DISTANCE (0=identical) because get_db() creates
    # the collection with hnsw:space="cosine" - similarity is 1 - distance.
    top_similarity = (1 - results[0][1]) if results else 0.0

    needs_fallback = top_similarity < SIMILARITY_THRESHOLD
    context_sufficient = None  # None = sufficiency check was skipped entirely
    if not needs_fallback:
        context_sufficient = _check_context_sufficiency(doc_context, payload.query_text)
        needs_fallback = not context_sufficient

    context_text = doc_context
    web_sources = []
    answer_source = "documents"

    if needs_fallback and WEB_FALLBACK_ENABLED:
        web_context, web_sources = _run_web_fallback(payload.query_text)
        if web_context:
            context_text = f"{doc_context}\n\n---\n\n{web_context}" if doc_context else web_context
            answer_source = "documents+web" if doc_context else "web_fallback"

    if not context_text:
        raise HTTPException(status_code=404, detail="No relevant documents or web results found.")

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=payload.query_text)

    print(f"Generated prompt: {prompt}")

    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    return {
        "response": response_text,
        "sources": doc_sources,
        "web_sources": web_sources,
        # Phase 5's Streamlit UI shows this directly so users can see
        # whether an answer came from the uploaded documents or the web.
        "answer_source": answer_source,
        "retrieval_debug": {
            "top_similarity": round(top_similarity, 4),
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "context_sufficient": context_sufficient,
        },
    }


# -------------------------------
# 📌 Compare Embeddings
# -------------------------------
def get_embedding(text):
    """Generate embedding for a given text."""
    embedding_model = get_embedding_function()
    return embedding_model.embed_query(text)


@app.post("/compare/")
async def compare_documents(request: CompareRequest):
    """Compare two documents based on embeddings."""

    db = get_db()
    results_1 = db.similarity_search(request.query_1, k=1)
    results_2 = db.similarity_search(request.query_2, k=1)

    if not results_1 or not results_2:
        raise HTTPException(status_code=404, detail="One or both queries did not match any document.")

    # Generate embeddings for comparison
    embedding_1 = get_embedding(results_1[0].page_content)
    embedding_2 = get_embedding(results_2[0].page_content)

    # Compute cosine similarity
    similarity = cosine_similarity([embedding_1], [embedding_2])[0][0]

    return {
        "query_1": request.query_1,
        "query_2": request.query_2,
        "similarity_score": round(similarity, 4),
        "source_1": results_1[0].metadata,
        "source_2": results_2[0].metadata,
    }

# -------------------------------
# ✅ Run FastAPI Server
# -------------------------------
# Start the server: uvicorn rag_api:app --host 0.0.0.0 --port 8000 --reload
