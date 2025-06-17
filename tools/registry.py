"""
tools/registry.py

Tool definitions for the agent system. Each tool is a plain Python function
wrapped with LangChain's @tool decorator. The orchestrator passes these to
the LLM as callable functions during tool-calling turns.

Tools available:
  - search_knowledge_base: RAG retrieval from pgvector
  - ingest_document: add a document to the knowledge base
  - recall_memory: read a named fact from session memory
  - save_to_memory: write a named fact to session memory
  - summarize_text: LLM-powered text summarization
"""

import logging
from typing import Optional

from langchain_core.tools import tool

log = logging.getLogger(__name__)

# These are set at runtime by the agent graph before tool execution
_rag_pipeline = None
_memory_store = None
_openai_client = None


def configure(rag_pipeline, memory_store, openai_client):
    """Inject dependencies. Call this before running the agent graph."""
    global _rag_pipeline, _memory_store, _openai_client
    _rag_pipeline = rag_pipeline
    _memory_store = memory_store
    _openai_client = openai_client


@tool
def search_knowledge_base(query: str, source_filter: Optional[str] = None) -> str:
    """
    Search the knowledge base for information relevant to the query.
    Returns the most relevant passages ranked by semantic similarity.

    Args:
        query: Natural language question or search phrase
        source_filter: Optional — restrict search to a specific document source

    Returns:
        Ranked passages with source attribution
    """
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
    """
    Add a document to the knowledge base by chunking, embedding, and storing it.
    Use this when the user provides new information that should be searchable later.

    Args:
        text: The document content to ingest
        source: A label or identifier for this document (e.g., filename, URL)

    Returns:
        Confirmation with the assigned document ID
    """
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
    """
    Recall a specific piece of information stored in session memory.
    Use this to retrieve facts or context saved in earlier turns.

    Args:
        key: The label used when the fact was saved

    Returns:
        The stored value, or a message if not found
    """
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
    """
    Save a fact or piece of information to session memory for later recall.

    Args:
        key: A short label describing what this fact is
        value: The information to store

    Returns:
        Confirmation message
    """
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
    """
    Summarize a long piece of text concisely.

    Args:
        text: The text to summarize
        max_words: Target length of the summary in words

    Returns:
        A concise summary
    """
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
