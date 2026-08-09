import os

from langchain_community.embeddings.ollama import OllamaEmbeddings

# BUG FIX (Phase 4): OllamaEmbeddings defaults to http://localhost:11434
# when no base_url is given. That's correct for bare-metal dev, but inside
# the Docker container "localhost" means the container itself, not the
# host machine Ollama actually runs on - every embed call failed with
# "Connection refused" until this was wired to the same OLLAMA_BASE_URL
# env var rag_api.py already used for its /health/ check (which is why
# /health/ reported ollama_reachable:true while /populate/ still failed -
# the health check and the actual embedding client weren't using the same
# configuration).
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_BASE_URL)
    return embeddings
