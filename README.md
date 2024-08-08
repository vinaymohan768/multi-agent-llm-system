# Multi-Agent LLM System

![CI](https://github.com/vinaymohan768/multi-agent-llm-system/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-purple)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?logo=openai&logoColor=white)

A production-ready multi-agent system built with LangGraph. An orchestrator routes each request to the right specialist agent (retrieval, analysis, or direct response), which can call tools, search a pgvector knowledge base, and maintain context across long conversations via rolling LLM summarization.

---

## Architecture

```
User Message
     │
     ▼
┌──────────────────────────────────────────────┐
│               Orchestrator                    │
│   Classifies intent → retrieve / analyze /   │
│   respond — avoids RAG on simple queries      │
└──────────┬─────────────────┬─────────────────┘
           │                 │
   ┌───────▼──────┐  ┌───────▼──────┐
   │   Retriever  │  │   Analyzer   │
   │  knowledge   │  │  synthesis + │
   │  base search │  │  multi-source│
   └───────┬──────┘  └───────┬──────┘
           └────────┬─────────┘
                    │
           ┌────────▼────────┐
           │   Tool Node     │   search_knowledge_base
           │  (LangGraph     │   ingest_document
           │   prebuilt)     │   recall_memory
           └────────┬────────┘   save_to_memory
                    │            summarize_text
           ┌────────▼────────┐
           │    Responder    │  ← synthesizes final answer
           └────────┬────────┘
                    │
              Final Response
```

**Memory layer — runs on every turn:**
```
PostgreSQL 16
├── conversation_memory   full message history per session
├── memory_summaries      LLM-generated rolling summaries (keeps context bounded)
├── document_chunks       RAG corpus with pgvector IVFFlat embeddings
└── agent_runs            audit log for every node execution
```

---

## Technical Highlights

**Intent-based routing** — The orchestrator classifies each query before routing, skipping RAG retrieval entirely for conversational messages. Three intents: `retrieve` (knowledge lookup), `analyze` (multi-source synthesis), `respond` (direct answer). This avoids unnecessary API calls and latency on simple turns.

**Sentence-aware chunking** — Documents are split at sentence boundaries using tiktoken rather than naive byte slicing, so chunks stay semantically coherent. Embeddings use `text-embedding-3-small` (1536 dims) stored in pgvector with an IVFFlat index for sub-linear ANN search.

**LLM reranking** — After cosine similarity retrieval (top-5 candidates), a second LLM pass scores each chunk's relevance 0–10 and returns the top-3. Lightweight alternative to a dedicated cross-encoder with no extra infrastructure.

**Rolling summarization** — When session history exceeds 20 messages, the LLM summarizes the older portion and stores it in `memory_summaries`. The next turn always sees: [rolling summary system message] + last 10 messages verbatim. Context window stays bounded without losing continuity.

**Tool call depth guard** — The graph tracks `tool_calls_made` per turn and caps at 3. Beyond that, it routes directly to the responder. Prevents runaway tool loops that burn tokens without adding value.

---

## RAG Pipeline

```
Input text
    │
    ▼  sentence-aware chunking (512 tokens, 64-token overlap)
    │
    ▼  OpenAI text-embedding-3-small → 1536-dim vectors
    │
    ▼  stored in pgvector with IVFFlat index (cosine)
    │
    ▼  at query time: ANN search → top-5 candidates
    │
    ▼  LLM relevance scoring → top-3 reranked chunks
    │
    ▼  context injected into agent state
```

---

## Getting Started

**Requirements:** Docker, Docker Compose, OpenAI API key

```bash
git clone https://github.com/vinaymohan768/multi-agent-llm-system
cd multi-agent-llm-system

cp .env.example .env
# Set OPENAI_API_KEY in .env

docker compose up --build
```

API runs at `http://localhost:8000` — Swagger UI at `http://localhost:8000/docs`

---

## API Examples

**Chat with the agent:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is LangGraph?", "session_id": "demo-session"}'
```

**Ingest a document into the knowledge base:**
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "LangGraph is a library for building stateful multi-actor LLM applications...", "source": "docs"}'
```

**Ask a question that triggers RAG retrieval:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How does LangGraph handle state?", "session_id": "demo-session"}'
```

**View full conversation history:**
```bash
curl http://localhost:8000/sessions/demo-session/history
```

**Clear session memory:**
```bash
curl -X DELETE http://localhost:8000/sessions/demo-session
```

---

## Project Structure

```
multi-agent-llm-system/
├── agents/
│   └── graph.py          # LangGraph StateGraph — orchestrator, retriever, analyzer, responder
├── rag/
│   └── pipeline.py       # Chunking, embedding, pgvector retrieval, LLM reranking
├── memory/
│   └── store.py          # PostgreSQL-backed memory with rolling summarization
├── tools/
│   └── registry.py       # Tool definitions: search_kb, ingest, memory, summarize
├── api/
│   └── main.py           # FastAPI: /chat, /ingest, /sessions/{id}
├── db/
│   └── init.sql          # pgvector schema, IVFFlat index, audit log
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Stack

`Python 3.11` · `LangGraph 0.2` · `LangChain 0.3` · `OpenAI API` · `pgvector` · `PostgreSQL 16` · `FastAPI` · `Docker Compose` · `tiktoken`
