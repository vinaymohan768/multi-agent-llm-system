-- init.sql — Multi-agent LLM system schema

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── Document chunks with embeddings ──────────────────────────────────────────
-- Stores ingested documents chunked and embedded for RAG retrieval.
CREATE TABLE IF NOT EXISTS document_chunks (
    id              UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    doc_id          TEXT            NOT NULL,     -- groups chunks from same source doc
    source          TEXT            NOT NULL,     -- file path, URL, or label
    chunk_index     INT             NOT NULL,
    content         TEXT            NOT NULL,
    token_count     INT             NOT NULL,
    embedding       vector(1536),                 -- text-embedding-3-small dimensions
    metadata        JSONB           NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- IVFFlat index for approximate nearest-neighbor search.
-- lists=100 is a reasonable starting point for up to ~1M vectors.
-- Rebuild with higher lists as corpus grows.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON document_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id
    ON document_chunks (doc_id);

CREATE INDEX IF NOT EXISTS idx_chunks_source
    ON document_chunks (source);


-- ── Conversation memory ───────────────────────────────────────────────────────
-- Persists full message history per session.
CREATE TABLE IF NOT EXISTS conversation_memory (
    id              BIGSERIAL       PRIMARY KEY,
    session_id      TEXT            NOT NULL,
    role            TEXT            NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content         TEXT            NOT NULL,
    tool_name       TEXT,                         -- populated when role = 'tool'
    metadata        JSONB           NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_memory_session_time
    ON conversation_memory (session_id, created_at ASC);


-- ── Memory summaries ──────────────────────────────────────────────────────────
-- LLM-generated rolling summaries to compress long conversation context.
CREATE TABLE IF NOT EXISTS memory_summaries (
    id              BIGSERIAL       PRIMARY KEY,
    session_id      TEXT            NOT NULL UNIQUE,
    summary         TEXT            NOT NULL,
    message_count   INT             NOT NULL,     -- how many messages this summary covers
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_summaries_session
    ON memory_summaries (session_id);


-- ── Agent run logs ────────────────────────────────────────────────────────────
-- Audit trail for every agent invocation: which node ran, what it produced.
CREATE TABLE IF NOT EXISTS agent_runs (
    id              BIGSERIAL       PRIMARY KEY,
    session_id      TEXT            NOT NULL,
    run_id          UUID            NOT NULL DEFAULT uuid_generate_v4(),
    node_name       TEXT            NOT NULL,
    input_tokens    INT,
    output_tokens   INT,
    latency_ms      INT,
    status          TEXT            NOT NULL CHECK (status IN ('success', 'error', 'skipped')),
    error           TEXT,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_runs_session
    ON agent_runs (session_id, created_at DESC);
