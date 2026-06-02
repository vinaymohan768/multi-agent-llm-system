"""
rag/pipeline.py

RAG pipeline: ingest → retrieve → rerank.

Chunking splits on sentence boundaries rather than fixed token counts so chunks
stay semantically coherent. Embeddings are batched at 100 per call. Reranking
uses a cheap LLM score pass over the top-K candidates before returning context.
"""

import os
import re
import uuid
import logging
import json
from typing import Optional

import numpy as np
import psycopg2
import psycopg2.extras
import tiktoken
from openai import OpenAI

log = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMS  = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", "512"))       # tokens
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "64"))     # tokens
TOP_K           = int(os.getenv("RETRIEVAL_TOP_K", "5"))
RERANK_K        = int(os.getenv("RERANK_TOP_K", "3"))

DB_DSN = (
    f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
    f"port={os.getenv('POSTGRES_PORT', '5432')} "
    f"dbname={os.getenv('POSTGRES_DB', 'agentdb')} "
    f"user={os.getenv('POSTGRES_USER', 'agent')} "
    f"password={os.getenv('POSTGRES_PASSWORD', 'agent')}"
)


# ── Chunking ──────────────────────────────────────────────────────────────────

def _tokenize(text: str, enc) -> list[int]:
    return enc.encode(text)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into token-bounded chunks that respect sentence boundaries."""
    enc = tiktoken.get_encoding("cl100k_base")

    # Split into sentences (handles common abbreviations)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_tokens: list[int] = []

    for sentence in sentences:
        sentence_tokens = _tokenize(sentence, enc)

        # If a single sentence exceeds chunk_size, hard-split it
        if len(sentence_tokens) > chunk_size:
            for i in range(0, len(sentence_tokens), chunk_size - overlap):
                segment = sentence_tokens[i : i + chunk_size]
                chunks.append(enc.decode(segment))
            continue

        if len(current_tokens) + len(sentence_tokens) > chunk_size:
            if current_tokens:
                chunks.append(enc.decode(current_tokens))
            # Start next chunk with overlap from end of previous
            current_tokens = current_tokens[-overlap:] + sentence_tokens
        else:
            current_tokens.extend(sentence_tokens)

    if current_tokens:
        chunks.append(enc.decode(current_tokens))

    return chunks


# ── Embeddings ────────────────────────────────────────────────────────────────

def embed_batch(texts: list[str], client: OpenAI) -> list[list[float]]:
    """Embed texts in batches of 100. Returns vectors in the same order as input."""
    BATCH_SIZE = 100
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        batch_embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


# ── Pipeline ──────────────────────────────────────────────────────────────────

class RAGPipeline:
    def __init__(self, openai_client: OpenAI, db_conn=None):
        self.client = openai_client
        self._conn = db_conn
        self._enc = tiktoken.get_encoding("cl100k_base")

    @property
    def conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(DB_DSN)
            self._conn.autocommit = True
        return self._conn

    def ingest(self, text: str, source: str, metadata: dict = None) -> str:
        """
        Chunk, embed, and store a document.
        Returns the doc_id assigned to this ingestion batch.
        """
        doc_id = str(uuid.uuid4())
        metadata = metadata or {}
        chunks = chunk_text(text)

        if not chunks:
            log.warning("No chunks produced for source=%s", source)
            return doc_id

        log.info("Ingesting source=%s | chunks=%d", source, len(chunks))
        embeddings = embed_batch(chunks, self.client)

        rows = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            token_count = len(self._enc.encode(chunk))
            rows.append((
                doc_id, source, i, chunk, token_count,
                embedding, json.dumps(metadata),
            ))

        with self.conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO document_chunks
                    (doc_id, source, chunk_index, content, token_count, embedding, metadata)
                VALUES %s
                """,
                rows,
                template="(%s, %s, %s, %s, %s, %s::vector, %s::jsonb)",
                page_size=50,
            )

        log.info("Ingested %d chunks for doc_id=%s", len(chunks), doc_id)
        return doc_id

    def retrieve(self, query: str, top_k: int = TOP_K, source_filter: Optional[str] = None) -> list[dict]:
        """
        Embed the query and run cosine similarity search against stored chunks.
        Returns ranked list of {content, source, score, chunk_index}.
        """
        query_embedding = embed_batch([query], self.client)[0]
        embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"

        sql = """
            SELECT
                content,
                source,
                chunk_index,
                metadata,
                1 - (embedding <=> %s::vector) AS similarity_score
            FROM document_chunks
        """
        params: list = [embedding_str]

        if source_filter:
            sql += " WHERE source = %s"
            params.append(source_filter)

        sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params.extend([embedding_str, top_k])

        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            results = [dict(r) for r in cur.fetchall()]

        log.debug("Retrieved %d chunks for query='%s...'", len(results), query[:60])
        return results

    def rerank(self, query: str, candidates: list[dict], top_k: int = RERANK_K) -> list[dict]:
        """Score each candidate 0–10 for relevance to the query, return top_k."""
        if not candidates:
            return []

        scoring_prompt = (
            f"Query: {query}\n\n"
            "Rate each passage's relevance to the query from 0 (irrelevant) to 10 (highly relevant).\n"
            "Respond with ONLY a JSON array of integers, one per passage, in order.\n\n"
            + "\n\n".join(
                f"Passage {i+1}:\n{c['content'][:400]}"
                for i, c in enumerate(candidates)
            )
        )

        try:
            response = self.client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": scoring_prompt}],
                temperature=0,
                max_tokens=50,
            )
            scores = json.loads(response.choices[0].message.content.strip())
            if len(scores) != len(candidates):
                raise ValueError("Score count mismatch")
        except Exception as e:
            log.warning("Reranking failed (%s), falling back to similarity order", e)
            return candidates[:top_k]

        scored = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [c for c, _ in scored[:top_k]]

    def query(self, query: str, source_filter: Optional[str] = None) -> list[dict]:
        """Full pipeline: retrieve → rerank → return top chunks."""
        candidates = self.retrieve(query, top_k=TOP_K, source_filter=source_filter)
        return self.rerank(query, candidates, top_k=RERANK_K)
