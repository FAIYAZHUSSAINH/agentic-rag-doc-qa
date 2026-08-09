"""Streamlit demo UI for the Agentic RAG Document Q&A API (Phase 5).

Deliberately a thin HTTP client of rag_api.py, not a second implementation
of the RAG pipeline. The version of this file in the original forked repo
(and its helper load_model.py, now deleted) called query.py/create_db.py's
internal functions directly - which completely bypassed everything built
in Phases 2-4: API key auth, rate limiting, and the Corrective RAG
threshold/sufficiency-check/web-fallback logic in /query/ all live in
rag_api.py, so a UI that sidesteps the API doesn't actually demonstrate
any of it. Every action below goes through the same HTTP endpoints a curl
command or another service would use - "keep it to Streamlit, demo UI not
a product build" still means it should exercise the real system.
"""
import os

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL_DEFAULT = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY_DEFAULT = os.getenv("API_KEY", "")

st.set_page_config(page_title="Agentic RAG Document Q&A", page_icon="📄", layout="centered")


# ---------------------------------------------------------------------------
# Thin API client helpers - every network call funnels through _unwrap() so
# the "what does this status code mean to a human" mapping lives in one
# place instead of being repeated in every caller.
# ---------------------------------------------------------------------------
def _headers(api_key: str) -> dict:
    return {"X-API-Key": api_key} if api_key else {}


def _unwrap(resp: httpx.Response):
    if resp.status_code == 401:
        return None, "Invalid or missing API key."
    if resp.status_code == 429:
        return None, "Rate limit exceeded - wait a moment and try again."
    if not resp.is_success:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        return None, f"API error {resp.status_code}: {detail}"
    return resp.json(), None


def api_health(base_url: str):
    try:
        return httpx.get(f"{base_url}/health/", timeout=5).json()
    except httpx.RequestError:
        return None


def api_upload(base_url: str, api_key: str, filename: str, file_bytes: bytes):
    try:
        resp = httpx.post(
            f"{base_url}/upload/",
            headers=_headers(api_key),
            files={"file": (filename, file_bytes, "application/pdf")},
            # CPU-only embedding of a full PDF has taken several minutes in
            # testing this session - a short timeout here would just make
            # a working upload look like a broken one.
            timeout=900,
        )
    except httpx.RequestError as exc:
        return None, f"Can't reach the API at {base_url} ({exc}). Is it running?"
    return _unwrap(resp)


def api_reset_database(base_url: str, api_key: str):
    try:
        resp = httpx.post(
            f"{base_url}/populate/", headers=_headers(api_key), json={"reset": True}, timeout=900
        )
    except httpx.RequestError as exc:
        return None, f"Can't reach the API at {base_url} ({exc}). Is it running?"
    return _unwrap(resp)


def api_query(base_url: str, api_key: str, query_text: str):
    try:
        resp = httpx.post(
            f"{base_url}/query/",
            headers=_headers(api_key),
            json={"query_text": query_text},
            # Worst case per rag_api.py's Corrective RAG flow: a
            # sufficiency check + a web fallback + the final generation
            # call - up to 3 sequential CPU-bound Ollama calls.
            timeout=300,
        )
    except httpx.RequestError as exc:
        return None, f"Can't reach the API at {base_url} ({exc}). Is it running?"
    return _unwrap(resp)


# ---------------------------------------------------------------------------
# Rendering helper - shared between replaying chat history and rendering a
# just-received answer, so the two code paths can't drift out of sync.
# ---------------------------------------------------------------------------
_ANSWER_SOURCE_LABELS = {
    "documents": "📄 From your documents",
    "web_fallback": "🌐 From the web (your documents didn't have this)",
    "documents+web": "📄🌐 From your documents + the web",
}


def render_answer_metadata(turn: dict):
    label = _ANSWER_SOURCE_LABELS.get(turn.get("answer_source"), turn.get("answer_source"))
    if label:
        st.caption(label)

    sources = turn.get("sources") or []
    web_sources = turn.get("web_sources") or []
    if sources or web_sources:
        with st.expander(f"Sources ({len(sources) + len(web_sources)})"):
            for s in sources:
                st.markdown(f"- 📄 `{s}`")
            for s in web_sources:
                st.markdown(f"- 🌐 {s}")


# ---------------------------------------------------------------------------
# Sidebar: connection settings, document upload, database reset, health
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("⚙️ Connection")
    base_url = st.text_input("API base URL", value=API_BASE_URL_DEFAULT).rstrip("/")
    api_key = st.text_input("API key", value=API_KEY_DEFAULT, type="password")

    health = api_health(base_url)
    if health is None:
        st.error("🔴 API unreachable")
    elif health.get("status") == "ok":
        st.success("🟢 API healthy")
    else:
        st.warning(f"🟡 API degraded — {health}")

    st.divider()
    st.subheader("📄 Documents")

    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Index uploaded PDFs", disabled=not uploaded_files):
        for uploaded_file in uploaded_files:
            with st.spinner(f"Indexing {uploaded_file.name} (CPU embedding can take a few minutes)..."):
                result, error = api_upload(base_url, api_key, uploaded_file.name, uploaded_file.getvalue())
            if error:
                st.error(f"{uploaded_file.name}: {error}")
            else:
                st.success(f"{uploaded_file.name}: {result['message']}")

    st.divider()
    if st.button("🗑️ Reset database", help="Deletes all indexed chunks and starts fresh"):
        with st.spinner("Resetting..."):
            result, error = api_reset_database(base_url, api_key)
        if error:
            st.error(error)
        else:
            st.success(result["message"])

# ---------------------------------------------------------------------------
# Main area: chat
# ---------------------------------------------------------------------------
st.title("📄 Agentic RAG Document Q&A")
st.caption(
    "Ask questions about the indexed documents. Falls back to a live web "
    "search when the local match is weak or doesn't actually answer the "
    "question (Phase 3's Corrective RAG check) - watch the source label "
    "below each answer."
)

if "history" not in st.session_state:
    st.session_state.history = []  # [{"role", "content", "sources", "web_sources", "answer_source"}, ...]

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant":
            render_answer_metadata(turn)

question = st.chat_input("Ask a question about your documents...")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking... (CPU inference can take a minute or two)"):
            result, error = api_query(base_url, api_key, question)

        if error:
            st.error(error)
            st.session_state.history.append({"role": "assistant", "content": f"⚠️ {error}"})
        else:
            st.markdown(result["response"])
            turn = {
                "role": "assistant",
                "content": result["response"],
                "sources": result.get("sources", []),
                "web_sources": result.get("web_sources", []),
                "answer_source": result.get("answer_source"),
            }
            render_answer_metadata(turn)
            st.session_state.history.append(turn)
