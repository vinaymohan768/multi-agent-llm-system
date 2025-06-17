"""
api/main.py

FastAPI service for the multi-agent LLM system.

Endpoints:
  POST /chat                    — send a message, get an agent response
  POST /ingest                  — add a document to the knowledge base
  GET  /sessions/{id}/history   — retrieve conversation history
  DELETE /sessions/{id}         — clear session memory
  GET  /health                  — liveness check
"""

import os
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from openai import OpenAI
from pydantic import BaseModel

from agents import build_graph
from memory import ConversationMemory
from rag import RAGPipeline
from tools import configure as configure_tools

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("api")

# ── Singletons ────────────────────────────────────────────────────────────────

openai_client: Optional[OpenAI] = None
agent_graph = None
rag_pipeline: Optional[RAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global openai_client, agent_graph, rag_pipeline

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    openai_client = OpenAI(api_key=api_key)
    rag_pipeline = RAGPipeline(openai_client=openai_client)
    agent_graph = build_graph()

    log.info("Agent graph compiled. RAG pipeline ready.")
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="Multi-Agent LLM System",
    description="LangGraph-based multi-agent system with RAG and persistent memory",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response models ───────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None   # auto-generated if not provided


class ChatResponse(BaseModel):
    session_id: str
    response: str
    intent: Optional[str] = None
    tool_calls_made: int = 0


class IngestRequest(BaseModel):
    text: str
    source: str


class IngestResponse(BaseModel):
    doc_id: str
    source: str
    message: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": os.getenv("LLM_MODEL", "gpt-4o-mini")}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())

    # Build session memory and configure tools with this session's context
    memory = ConversationMemory(
        session_id=session_id,
        openai_client=openai_client,
    )
    configure_tools(
        rag_pipeline=rag_pipeline,
        memory_store=memory,
        openai_client=openai_client,
    )

    # Persist the user message
    memory.add("user", req.message)

    # Build context from memory (summary + recent messages)
    context_messages = memory.get_context()

    # Construct the initial graph state
    initial_state = {
        "messages": [HumanMessage(content=req.message)],
        "session_id": session_id,
        "intent": None,
        "retrieved_context": [],
        "tool_calls_made": 0,
        "final_response": None,
    }

    try:
        result = agent_graph.invoke(initial_state)
    except Exception as e:
        log.error("Agent graph error: %s", e)
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    final = result.get("final_response") or "I was unable to generate a response."

    # Persist the assistant response
    memory.add("assistant", final)

    return ChatResponse(
        session_id=session_id,
        response=final,
        intent=result.get("intent"),
        tool_calls_made=result.get("tool_calls_made", 0),
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")

    try:
        doc_id = rag_pipeline.ingest(req.text, source=req.source)
    except Exception as e:
        log.error("Ingestion error: %s", e)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    return IngestResponse(
        doc_id=doc_id,
        source=req.source,
        message=f"Document ingested successfully into knowledge base.",
    )


@app.get("/sessions/{session_id}/history")
def get_history(session_id: str):
    memory = ConversationMemory(session_id=session_id, openai_client=openai_client)
    history = memory.get_full_history()
    if not history:
        raise HTTPException(status_code=404, detail="Session not found or empty")
    return {"session_id": session_id, "messages": history}


@app.delete("/sessions/{session_id}")
def clear_session(session_id: str):
    memory = ConversationMemory(session_id=session_id, openai_client=openai_client)
    memory.clear()
    return {"session_id": session_id, "status": "cleared"}
