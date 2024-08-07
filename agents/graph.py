"""
agents/graph.py

LangGraph orchestration for the multi-agent system.

Flow:
  START → orchestrator → retriever | analyzer | responder
                                  ↓
                             tool_node (if tool calls pending)
                                  ↓
                             responder → END

The orchestrator classifies the user's intent and hands off to the right
specialist. Retriever and analyzer can call tools; responder always produces
the final answer.
"""

import os
import json
import logging
import time
from typing import Annotated, TypedDict, Literal, Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import get_tools

log = logging.getLogger(__name__)

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    intent: Optional[str]           # classified by orchestrator: "retrieve" | "analyze" | "respond"
    retrieved_context: list[str]    # accumulated RAG results across turns
    tool_calls_made: int            # track tool usage depth to prevent loops
    final_response: Optional[str]   # set by responder node


# ── LLM ───────────────────────────────────────────────────────────────────────

def _get_llm(tools=None) -> ChatOpenAI:
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    if tools:
        llm = llm.bind_tools(tools)
    return llm


# ── Orchestrator ───────────────────────────────────────────────────────────────

ORCHESTRATOR_SYSTEM = """You are the orchestrator of a multi-agent system. Classify the user's request into one of three intents:

- "retrieve": needs information looked up from the knowledge base
- "analyze": needs deeper analysis or synthesis across multiple sources
- "respond": small talk, simple factual question, or enough context already exists

Reply with JSON only: {"intent": "retrieve" | "analyze" | "respond", "reasoning": "one sentence"}
"""

def orchestrator_node(state: AgentState) -> dict:
    last_user_message = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), ""
    )

    llm = _get_llm()
    response = llm.invoke([
        SystemMessage(content=ORCHESTRATOR_SYSTEM),
        HumanMessage(content=last_user_message),
    ])

    try:
        parsed = json.loads(response.content)
        intent = parsed.get("intent", "respond")
    except (json.JSONDecodeError, AttributeError):
        intent = "respond"

    log.info("orchestrator | session=%s intent=%s", state.get("session_id"), intent)
    return {"intent": intent}


# ── Retriever ──────────────────────────────────────────────────────────────────

RETRIEVER_SYSTEM = """You are a retrieval specialist. Search the knowledge base to find passages that answer the user's question.

Use search_knowledge_base. If the first result isn't useful, try a rephrased query. Summarize what you find.
"""

def retriever_node(state: AgentState) -> dict:
    tools = get_tools()
    llm = _get_llm(tools=tools)

    messages = [
        SystemMessage(content=RETRIEVER_SYSTEM),
        *state["messages"][-5:],
    ]
    response = llm.invoke(messages)

    retrieved = list(state.get("retrieved_context", []))
    if response.content:
        retrieved.append(response.content)

    return {
        "messages": [response],
        "retrieved_context": retrieved,
        "tool_calls_made": state.get("tool_calls_made", 0) + 1,
    }


# ── Analyzer ───────────────────────────────────────────────────────────────────

ANALYZER_SYSTEM = """You are an analytical specialist. Synthesize information from multiple sources to provide a thorough, well-reasoned answer.

Search the knowledge base as needed. Cross-reference sources, flag conflicts, and back up conclusions with evidence.
"""

def analyzer_node(state: AgentState) -> dict:
    tools = get_tools()
    llm = _get_llm(tools=tools)

    context_block = ""
    if state.get("retrieved_context"):
        context_block = "\n\n[Previously retrieved]\n" + "\n---\n".join(state["retrieved_context"])

    messages = [
        SystemMessage(content=ANALYZER_SYSTEM + context_block),
        *state["messages"][-5:],
    ]
    response = llm.invoke(messages)

    return {
        "messages": [response],
        "tool_calls_made": state.get("tool_calls_made", 0) + 1,
    }


# ── Responder ──────────────────────────────────────────────────────────────────

RESPONDER_SYSTEM = """You are the final step. Given the conversation and any retrieved context, write a clear, accurate answer. Cite sources naturally where you use them."""

def responder_node(state: AgentState) -> dict:
    llm = _get_llm()

    context_block = ""
    if state.get("retrieved_context"):
        context_block = "\n\n[Knowledge base context]\n" + "\n---\n".join(state["retrieved_context"])

    messages = [
        SystemMessage(content=RESPONDER_SYSTEM + context_block),
        *state["messages"][-8:],
    ]
    response = llm.invoke(messages)

    return {
        "messages": [response],
        "final_response": response.content,
    }


# ── Routing ────────────────────────────────────────────────────────────────────

def route_from_orchestrator(state: AgentState) -> Literal["retriever", "analyzer", "responder"]:
    intent = state.get("intent", "respond")
    if intent == "retrieve":
        return "retriever"
    if intent == "analyze":
        return "analyzer"
    return "responder"


def route_after_agent(state: AgentState) -> Literal["tools", "responder"]:
    """Go to tools if the agent made tool calls; otherwise hand off to responder."""
    # Hard cap at 3 tool-calling rounds to prevent runaway loops
    if state.get("tool_calls_made", 0) >= 3:
        return "responder"

    last = state["messages"][-1] if state["messages"] else None
    if last and hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    return "responder"


# ── Graph ──────────────────────────────────────────────────────────────────────

def build_graph():
    tools = get_tools()

    graph = StateGraph(AgentState)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("analyzer", analyzer_node)
    graph.add_node("responder", responder_node)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "orchestrator")

    graph.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {"retriever": "retriever", "analyzer": "analyzer", "responder": "responder"},
    )

    # route_after_agent now only returns "tools" | "responder": no END in the map
    graph.add_conditional_edges("retriever", route_after_agent, {"tools": "tools", "responder": "responder"})
    graph.add_conditional_edges("analyzer", route_after_agent, {"tools": "tools", "responder": "responder"})

    graph.add_edge("tools", "responder")
    graph.add_edge("responder", END)

    return graph.compile()

