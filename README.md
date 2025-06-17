# multi-agent-llm-system

Production-grade multi-agent LLM system built with LangGraph, OpenAI, and pgvector. Features an orchestrator-routed agent graph, full RAG pipeline with semantic reranking, and PostgreSQL-backed conversation memory with rolling summarization.

---

## Architecture

```
User Message
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│                    Orchestrator Node                     │
│  Classifies intent: "retrieve" | "analyze" | "respond"  │
└──────────────┬───────────────────┬──────────────────────┘
               │                   │
       ┌───────▼──────┐   ┌────────▼─────┐
       │   Retriever  │   │   Analyzer   │
       │    Agent     │   │    Agent     │
       │              │   │              │
       │ search_kb    │   │ search_kb    │
       │ tool call    │   │ + synthesis  │
       └───────┬──────┘   └────────┬─────┘
               │                   │
               └─────────┬─────────┘
                         │
                ┌────────▼────────┐
                │   Tool Node     │  ← executes tool calls
                │                 │
                │ search_kb       │
                │ ingest_doc      │
                │ recall_memory   │
                │ save_to_memory  │
                │ summarize_text  │
                └────────┬────────┘
                         │
                ┌────────▼────────┐
                │    Responder    │  ← final synthesis
                └────────┬────────┘
                         │
                   Final Response
```

**Memory layer** (runs alongside every turn):
```
PostgreSQL
├── conversation_memory   — full message history per session
├── memory_summaries      — LLM-generated rolling summaries (bounds context window)
├── document_chunks       — RAG corpus (pgvector embeddings)
└── agent_runs            — audit log of every node execution
```

---

## Key Design Decisions

**LangGraph StateGraph** — The agent graph is a typed, inspectable state machine. Every node receives the full state and returns a partial update. This makes the flow deterministic and debuggable — you can replay any state without re-running the whole graph.

**Intent-based routing** — The orchestrator classifies the user's query before routing. This avoids running expensive RAG retrieval on conversational queries that don't need it. Three intents: `retrieve` (knowledge lookup), `analyze` (multi-source synthesis), `respond` (direct answer).

**Sentence-aware chunking + LLM reranking** — Documents are chunked at sentence boundaries using tiktoken to respect token limits, then embeddings are stored with an IVFFlat index for sub-linear ANN search. Retrieved candidates are reranked by the LLM using relevance scoring before being passed as context.

**Rolling memory summarization** — When conversation history exceeds 20 messages, older messages are summarized by the LLM and stored as a single system message. The most recent 10 messages are always kept verbatim. This keeps context windows bounded without losing semantic continuity.

**Tool call depth guard** — The graph tracks `tool_calls_made` per turn and caps recursion at 3 to prevent runaway tool loops. Beyond that, it routes straight to the responder.

---

## RAG Pipeline

```
Document text
     │
     ▼
Sentence-aware chunking (512 tokens, 64 token overlap)
     │
     ▼
OpenAI text-embedding-3-small → 1536-dim vectors
     │
     ▼
pgvector IVFFlat index (cosine similarity)
     │
     ▼ query time
Cosine ANN search → top-5 candidates
     │
     ▼
LLM relevance reranking → top-3 chunks
     │
     ▼
Context passed to agent
```

---

## Getting Started

**Prerequisites:** Docker, Docker Compose, OpenAI API key

```bash
git clone https://github.com/vinaymohan768/multi-agent-llm-system
cd multi-agent-llm-system

cp .env.example .env
# Add your OPENAI_API_KEY to .env

docker compose up --build
```

API available at `http://localhost:8000`. Swagger UI at `http://localhost:8000/docs`.

---

## Usage Examples

**Send a message:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the capital of France?", "session_id": "session-001"}'
```

**Ingest a document:**
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "LangGraph is a library for building stateful, multi-actor applications with LLMs...",
    "source": "langgraph-docs"
  }'
```

**Ask a question about ingested content:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is LangGraph used for?", "session_id": "session-001"}'
```

**View conversation history:**
```bash
curl http://localhost:8000/sessions/session-001/history
```

---

## Project Structure

```
multi-agent-llm-system/
├── agents/
│   ├── graph.py          # LangGraph StateGraph: orchestrator + specialist nodes
│   └── __init__.py
├── rag/
│   ├── pipeline.py       # Chunking, embedding, pgvector retrieval, LLM reranking
│   └── __init__.py
├── memory/
│   ├── store.py          # PostgreSQL-backed memory with rolling summarization
│   └── __init__.py
├── tools/
│   ├── registry.py       # Tool definitions: search_kb, ingest, memory, summarize
│   └── __init__.py
├── api/
│   └── main.py           # FastAPI: /chat, /ingest, /sessions
├── db/
│   └── init.sql          # pgvector schema, IVFFlat index, audit log table
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Tech Stack

`Python 3.11` `LangGraph 0.2` `LangChain 0.3` `OpenAI API` `pgvector` `PostgreSQL 16` `FastAPI` `Docker Compose`
