"""
tools/registry.py

LangChain tool definitions. configure() injects the shared singletons at
startup; the tools close over them so each call gets the right pipeline/client.
"""

import logging
from typing import Optional

from langchain_core.tools import tool

log = logging.getLogger(__name__)

_rag_pipeline = None
_memory_store = None
_openai_client = None


def configure(rag_pipeline, memory_store, openai_client):
    """Wire up shared dependencies. Called once at app startup."""
    global _rag_pipeline, _memory_store, _openai_client
    _rag_pipeline = rag_pipeline
    _memory_store = memory_store
    _openai_client = openai_client


@tool
def search_knowledge_base(query: str, source_filter: Optional[str] = None) -> str:
    """Search the knowledge base and return relevant passages ranked by similarity.
    Pass source_filter to restrict results to a specific document."""
    if _rag_pipeline is None:
        return "Error: RAG pipeline not initialized."

    try:
        results = _rag_pipeline.query(query, source_filter=source_filter)
        if not results:
            return "No relevant information found in the knowledge base."

        formatted = []
        for i, r in enumerate(results, 1):
            score = r.get("similarity_score", 0)
            formatted.append(
                f"[{i}] Source: {r['source']} (relevance: {score:.2f})\n{r['content']}"
            )
        return "\n\n---\n\n".join(formatted)
    except Exception as e:
        log.error("search_knowledge_base failed: %s", e)
        return f"Search failed: {e}"


@tool
def ingest_document(text: str, source: str) -> str:
    """Chunk, embed, and store a document so it's searchable later.
    source should be a short label like a filename or URL."""
    if _rag_pipeline is None:
        return "Error: RAG pipeline not initialized."

    try:
        doc_id = _rag_pipeline.ingest(text, source=source)
        return f"Document ingested successfully. doc_id={doc_id}, source={source}"
    except Exception as e:
        log.error("ingest_document failed: %s", e)
        return f"Ingestion failed: {e}"


@tool
def recall_memory(key: str) -> str:
    """Look up a fact stored in session memory by keyword."""
    if _memory_store is None:
        return "Error: Memory store not initialized."

    try:
        history = _memory_store.get_full_history()
        # Search for the most recent assistant message that mentioned this key
        for msg in reversed(history):
            if key.lower() in msg.get("content", "").lower():
                return f"Found reference to '{key}':\n{msg['content'][:500]}"
        return f"No stored memory found for key: '{key}'"
    except Exception as e:
        return f"Memory recall failed: {e}"


@tool
def save_to_memory(key: str, value: str) -> str:
    """Store a fact in session memory under the given key for later retrieval."""
    if _memory_store is None:
        return "Error: Memory store not initialized."

    try:
        _memory_store.add(
            role="tool",
            content=f"[Saved memory] {key}: {value}",
            tool_name="save_to_memory",
        )
        return f"Saved to memory — {key}: {value}"
    except Exception as e:
        return f"Memory save failed: {e}"


@tool
def summarize_text(text: str, max_words: int = 150) -> str:
    """Summarize text to roughly max_words words using the LLM."""
    if _openai_client is None:
        return "Error: OpenAI client not initialized."

    try:
        import os
        response = _openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": f"Summarize the following text in approximately {max_words} words. Be specific and preserve key facts.",
                },
                {"role": "user", "content": text[:4000]},
            ],
            temperature=0,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Summarization failed: {e}"


TOOL_REGISTRY = [
    search_knowledge_base,
    ingest_document,
    recall_memory,
    save_to_memory,
    summarize_text,
]


def get_tools() -> list:
    return TOOL_REGISTRY
