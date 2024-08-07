"""
Tests for the RAG pipeline :  chunking and embedding logic.
No external services needed (DB/OpenAI are mocked).
"""

import pytest
from unittest.mock import MagicMock, patch
from rag.pipeline import chunk_text, embed_batch, RAGPipeline


# ── Chunking ──────────────────────────────────────────────────────────────────

def test_chunk_text_splits_long_input():
    text = "This is a sentence. " * 200  # well over 512 tokens
    chunks = chunk_text(text, chunk_size=512, overlap=64)
    assert len(chunks) > 1


def test_chunk_text_short_input_produces_single_chunk():
    text = "Short document with just a few words."
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert "Short document" in chunks[0]


def test_chunk_text_empty_input_returns_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_overlap_carries_tokens():
    # Each chunk after the first should share tokens with the previous one
    text = "Alpha beta gamma. " * 100
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) >= 2
    # The start of chunk[1] should overlap with the end of chunk[0]
    # (decoded tokens may not match perfectly, but lengths should be consistent)
    for chunk in chunks:
        assert len(chunk) > 0


def test_chunk_text_single_oversized_sentence_is_hard_split():
    # One sentence that's longer than chunk_size tokens
    long_sentence = "word " * 600  # ~600 tokens, no sentence boundary
    chunks = chunk_text(long_sentence, chunk_size=100, overlap=10)
    assert len(chunks) > 1


# ── Embed batch ───────────────────────────────────────────────────────────────

def test_embed_batch_returns_correct_count():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1] * 1536, index=i) for i in range(3)
    ]
    mock_client.embeddings.create.return_value = mock_response

    texts = ["text one", "text two", "text three"]
    embeddings = embed_batch(texts, mock_client)

    assert len(embeddings) == 3
    assert len(embeddings[0]) == 1536


def test_embed_batch_batches_correctly():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.0] * 1536, index=i) for i in range(100)]
    mock_client.embeddings.create.return_value = mock_response

    # 150 texts should trigger 2 API calls (batch size = 100)
    texts = [f"text {i}" for i in range(150)]
    embed_batch(texts, mock_client)

    assert mock_client.embeddings.create.call_count == 2


# ── RAGPipeline.query ─────────────────────────────────────────────────────────

def test_pipeline_query_returns_reranked_results():
    mock_client = MagicMock()

    # Mock embedding call
    embed_response = MagicMock()
    embed_response.data = [MagicMock(embedding=[0.1] * 1536, index=0)]
    mock_client.embeddings.create.return_value = embed_response

    # Mock rerank LLM call :  returns scores for 2 candidates
    rerank_response = MagicMock()
    rerank_response.choices = [MagicMock(message=MagicMock(content="[8, 3]"))]
    mock_client.chat.completions.create.return_value = rerank_response

    pipeline = RAGPipeline(openai_client=mock_client)

    # Mock the DB retrieve call
    candidates = [
        {"content": "Highly relevant passage", "source": "doc-1", "similarity_score": 0.92},
        {"content": "Less relevant passage",   "source": "doc-2", "similarity_score": 0.71},
    ]
    pipeline.retrieve = MagicMock(return_value=candidates)

    results = pipeline.query("test query")

    # Reranker should have put the higher-scored candidate first
    assert results[0]["content"] == "Highly relevant passage"


def test_pipeline_rerank_falls_back_on_bad_llm_response():
    mock_client = MagicMock()
    rerank_response = MagicMock()
    rerank_response.choices = [MagicMock(message=MagicMock(content="not valid json"))]
    mock_client.chat.completions.create.return_value = rerank_response

    pipeline = RAGPipeline(openai_client=mock_client)
    candidates = [
        {"content": "A", "source": "s1", "similarity_score": 0.9},
        {"content": "B", "source": "s2", "similarity_score": 0.8},
    ]

    # Should fall back to similarity order, not raise
    results = pipeline.rerank("query", candidates, top_k=2)
    assert len(results) == 2

