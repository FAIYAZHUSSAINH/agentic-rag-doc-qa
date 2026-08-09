"""Web search fallback for Corrective RAG (Phase 3).

Used by /query/ in rag_api.py when the locally retrieved document chunks
are either too dissimilar from the question (below SIMILARITY_THRESHOLD) or
an LLM sufficiency check decides they don't actually answer it. In both
cases we widen the context with a handful of live web results before
generating the final answer, instead of confidently answering from
irrelevant/incomplete local context.

Uses Tavily (tavily.com) rather than a scraped DuckDuckGo wrapper: Tavily is
purpose-built for feeding LLMs (results come back as clean title/url/content
tuples meant to be dropped straight into a prompt, no HTML to strip), at the
cost of needing a free API key. See TAVILY_API_KEY in .env.example.
"""
import os

from tavily import TavilyClient


def web_search(query: str, max_results: int = 3) -> list[dict]:
    """Return up to max_results {title, snippet, url} dicts from Tavily.

    Deliberately does NOT catch exceptions here - a missing/invalid API key
    or a network error should be visible to the caller (rag_api.py), which
    decides how to degrade (fall back to local-context-only rather than
    500ing the whole request). Swallowing errors in this low-level function
    would hide real configuration problems (e.g. forgetting to set
    TAVILY_API_KEY) behind a silent "web search found nothing."
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Get a free key at https://tavily.com "
            "and add it to your .env file (see .env.example)."
        )

    client = TavilyClient(api_key=api_key)
    response = client.search(query=query, max_results=max_results)

    return [
        {
            "title": r.get("title", ""),
            "snippet": r.get("content", ""),
            "url": r.get("url", ""),
        }
        for r in response.get("results", [])
    ]
