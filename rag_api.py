from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
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

# Constants
CHROMA_PATH = "chroma"
# BUG FIX: this was "data", but the PDFs that ship with the repo live in
# docs/ (create_db.py has always pointed there). With "data" the loader
# pointed at a directory that doesn't exist, so /populate/ silently produced
# zero chunks. Matching create_db.py's DATA_PATH here.
DATA_PATH = "docs"

# Initialize FastAPI app
app = FastAPI()


# BUG FIX: the original code built ONE Chroma client at import time and
# reused it for the app's entire lifetime. That breaks /populate/ with
# reset=True: clear_database() shutil.rmtree's the chroma/ directory out
# from under a client that still has it open, and on Windows deleting a
# file that's open in the same process either fails outright
# (PermissionError) or leaves the client pointed at now-nonexistent files.
# Building a fresh client per call (matching the pattern already used in
# create_db.py and query.py) avoids holding a long-lived handle at all.
def get_db() -> Chroma:
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


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


@app.post("/populate/")
async def populate_db(request: PopulateDBRequest):
    """Endpoint to populate the database with documents."""
    return populate_database(reset=request.reset)


# -------------------------------
# 📌 Query Processing (RAG)
# -------------------------------
@app.post("/query/")
async def query_rag(request: QueryRequest):
    """Search for relevant documents and generate a response."""
    # Search the DB
    results = get_db().similarity_search_with_score(request.query_text, k=5)

    if not results:
        raise HTTPException(status_code=404, detail="No relevant documents found.")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=request.query_text)

    print(f"Generated prompt: {prompt}")

    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", "Unknown") for doc, _ in results]

    return {
        "response": response_text,
        "sources": sources
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
