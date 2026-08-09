# 📄 Agentic RAG Document Q&A API

A **Retrieval-Augmented Generation** API that answers questions over your own PDFs, and knows when its own retrieval isn't good enough to trust — falling back to a live web search instead of confidently answering from the wrong context. Built on **FastAPI + ChromaDB + Ollama (Mistral)** via **LangChain**, with API-key auth, rate limiting, Docker Compose, and a Streamlit demo UI.

Forked from [smshelar/rag_pipeline](https://github.com/smshelar/rag_pipeline) and extended in five phases — see the commit history for the full story (each commit message explains the *why*, not just the *what*).

---

## ✅ Features

- **RAG over your PDFs** — upload, chunk, embed (`nomic-embed-text`), and query via Mistral
- **Corrective RAG** — a cosine-similarity threshold *and* an LLM sufficiency check decide whether local context is good enough; if not, it falls back to a live [Tavily](https://tavily.com) web search and merges the results before answering
- **API key auth** on the endpoints that cost compute (`/populate/`, `/upload/`, `/query/`)
- **Rate limiting** (slowapi) to keep CPU-bound endpoints from being hammered
- **`/health/`** — checks Ollama (and Chroma, in Docker) are actually reachable, not just that the process is alive
- **Dockerized** — `docker-compose.yml` runs the API and a standalone ChromaDB server as separate services
- **Streamlit UI** — upload PDFs, chat, see cited sources and whether each answer came from your documents, the web, or both
- **Tests** — fast mocked tests for the API contract (auth/rate-limit/Corrective-RAG logic, no live models needed) and slower tests for actual RAG correctness (needs Ollama)

---

## 🚀 Quickstart

### Option A — Bare metal

**1. Prerequisites**
- Python 3.12 (see [`requirements.txt`](requirements.txt) for why not 3.13/3.14)
- [Ollama](https://ollama.com/) installed and running, with the two models pulled:
  ```bash
  ollama pull mistral
  ollama pull nomic-embed-text
  ```

**2. Set up the environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
cp .env.example .env         # then fill in API_KEY and TAVILY_API_KEY
```

**3. Run the API**
```bash
uvicorn rag_api:app --host 0.0.0.0 --port 8000 --reload
```

**4. Run the Streamlit UI** (in a second terminal)
```bash
streamlit run app.py
```

### Option B — Docker

Ollama still runs on the host (see the note in `docker-compose.yml` for why it isn't containerized); the API and ChromaDB run in containers.

```bash
cp .env.example .env   # then fill in API_KEY and TAVILY_API_KEY
docker compose up -d
streamlit run app.py   # UI still runs bare-metal, pointed at http://localhost:8000
```

---

## 🚀 API Endpoints

All request/response bodies below are JSON unless noted. Endpoints marked 🔒 require an `X-API-Key` header matching `API_KEY` in `.env`.

### `GET /health/`
Liveness/readiness probe — no auth, no rate limit (meant for infra polling).
```json
{"status": "ok", "ollama_reachable": true, "chroma_reachable": true}
```
(`chroma_reachable` only appears when `CHROMA_MODE=http`, i.e. in Docker.)

### 🔒 `POST /upload/`
Upload a PDF (multipart form, field name `file`) — saves it into the corpus and indexes it immediately. What the Streamlit UI's upload button calls.
```json
{"filename": "report.pdf", "message": "Database populated with 42 new chunks (0 already existed)"}
```

### 🔒 `POST /populate/`
Re-index everything already sitting in `docs/` on the server. Mainly for ops/CLI use and the test suite — a browser can't hand this endpoint a server-side directory path (that's what `/upload/` is for).
```json
{"reset": true}
```
```json
{"message": "Database populated with 676 new chunks (0 already existed)"}
```

### 🔒 `POST /query/`
Ask a question. Runs the Corrective RAG flow: similarity threshold → (if it clears the bar) LLM sufficiency check → web fallback if either check fails.
```json
{"query_text": "What is the company name?"}
```
```json
{
  "response": "The company name is ConocoPhillips.",
  "sources": ["docs/2024-conocophillips-proxy-statement.pdf:6:0", "..."],
  "web_sources": [],
  "answer_source": "documents",
  "retrieval_debug": {
    "top_similarity": 0.565,
    "similarity_threshold": 0.5,
    "context_sufficient": true
  }
}
```
`answer_source` is one of `"documents"`, `"web_fallback"`, or `"documents+web"`.

### `POST /compare/`
Cosine similarity between the top retrieved chunk for two queries. Not auth-gated or rate-limited (out of scope for the auth work — see Phase 2 commit).
```json
{"query_1": "board of directors", "query_2": "executive compensation"}
```

---

## ⚙️ Configuration

See [`.env.example`](.env.example) for the full list with explanations. Highlights:

| Variable | Purpose |
|---|---|
| `API_KEY` | Required in `X-API-Key` for the protected endpoints above |
| `TAVILY_API_KEY` | Free key from [tavily.com](https://tavily.com) for web fallback |
| `SIMILARITY_THRESHOLD` | Cosine similarity floor before web fallback kicks in (default `0.5`, empirically measured — see `rag_api.py`) |
| `WEB_FALLBACK_ENABLED` | Set `false` to disable web fallback entirely |
| `CHROMA_MODE` | `local` (embedded, default) or `http` (standalone server — what Docker uses) |
| `OLLAMA_BASE_URL` | Where to find Ollama's API (`localhost` bare-metal, `host.docker.internal` in Docker) |

---

## 🧪 Running Tests

```bash
pytest
```
Runs three files:
- `test_rag_pipeline.py` — RAG *correctness* (needs a live, populated Ollama; slower)
- `test_security.py` — auth, rate limiting, `/health/` (fully mocked; fast)
- `test_corrective_rag.py` — the threshold/sufficiency/fallback decision logic (fully mocked; fast)

---

## 🏗️ Folder Structure

```
📁 rag_pipeline
│-- 📂 docs/                     # PDFs to index (populate/upload target)
│-- 📂 chroma/                   # Local embedded ChromaDB storage (bare-metal mode only)
│-- 📜 rag_api.py                # FastAPI app - auth, rate limiting, Corrective RAG, all endpoints
│-- 📜 embedding_function.py     # Shared Ollama embedding config
│-- 📜 web_search.py             # Tavily web-search wrapper (Corrective RAG fallback)
│-- 📜 create_db.py              # CLI: populate ChromaDB from docs/
│-- 📜 query.py                  # CLI: ask a question against the DB directly
│-- 📜 embeddings_compare.py     # Standalone script version of /compare/ (not used by the API)
│-- 📜 app.py                    # Streamlit UI - a pure HTTP client of rag_api.py
│-- 📜 test_rag_pipeline.py      # Pytest: RAG correctness
│-- 📜 test_security.py          # Pytest: auth/rate-limit/health
│-- 📜 test_corrective_rag.py    # Pytest: Corrective RAG decision logic
│-- 📜 Dockerfile                # FastAPI app image
│-- 📜 docker-compose.yml        # app + chroma services
│-- 📜 requirements.txt          # Dependencies (audited, not a raw pip freeze - see file comments)
│-- 📜 .env.example              # Documents required env vars (no real secrets)
```

---

## 🔗 References
- [LangChain Docs](https://python.langchain.com/)
- [Ollama](https://ollama.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [ChromaDB](https://github.com/chroma-core/chroma)
- [Tavily](https://tavily.com/)
