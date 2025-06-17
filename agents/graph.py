"""
agents/graph.py

LangGraph multi-agent orchestration.

Graph structure:
                    ┌─────────────┐
                    │   START     │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ orchestrator│  ← routes based on intent
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼──┐  ┌──────▼──┐  ┌─────▼──────┐
       │retriever│  │analyzer │  │  responder  │
       │  agent  │  │  agent  │  │  (direct)   │
       └──────┬──┘  └──────┬──┘  └─────┬───────┘
              │            │            │
              └────────────▼────────────┘
                    ┌──────┴──────┐
                    │  tool_node  │  ← executes tool calls
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │    END      │
                    └─────────────┘

Node responsibilities:
  orchestrator  — classifies user intent, routes to the right specialist
  retriever     — knowledge-base search using RAG tools
  analyzer      — deeper analysis, cross-referencing multiple sources
  responder     — synthesizes final answer from accumulated context
  tool_node     — executes any tool calls made by the agents
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


# ── LLM setup ─────────────────────────────────────────────────────────────────

def _get_llm(tools=None) -> ChatOpenAI:
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    if tools:
        llm = llm.bind_tools(tools)
    return llm


# ── Node: Orchestrator ────────────────────────────────────────────────────────

ORCHESTRATOR_SYSTEM = """You are the orchestrator of a multi-agent system. Your job is to:
1. Understand the user's intent
2. Route to the appropriate specialist agent

Classify the request into one of three intents:
- "retrieve": User needs information looked up from the knowledge base
- "analyze": User needs deeper analysis, comparison, or synthesis across multiple sources
- "respond": User is making small talk, asking a simple factual question, or the context already has enough information

Respond with a JSON object: {"intent": "retrieve" | "analyze" | "respond", "reasoning": "brief explanation"}
"""

def orchestrator_node(state: AgentState) -> dict:
    """Classifies intent and routes to the appropriate agent."""
    messages = state["messages"]
    last_user_message = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
    )

    llm = _get_llm()
    response = llm.invoke([
        SystemMessage(content=ORCHESTRATOR_SYSTEM),
        HumanMessage(content=f"User message: {last_user_message}"),
    ])

    try:
        parsed = json.loads(response.content)
        intent = parsed.get("intent", "respond")
    except (json.JSONDecodeError, AttributeError):
        intent = "respond"

    log.info("Orchestrator | session=%s intent=%s", state.get("session_id"), intent)
    return {"intent": intent}


# ── Node: Retriever Agent ─────────────────────────────────────────────────────

RETRIEVER_SYSTEM = """You are a retrieval specialist. Your job is to find relevant information from the knowledge base to answer the user's question.

Use the search_knowledge_base tool to find relevant passages. If the first search doesn't return useful results, try a rephrased query.
After retrieving, summarize the key points found.
"""

def retriever_node(state: AgentState) -> dict:
    """Searches the knowledge base and accumulates context."""
    tools = get_tools()
    llm = _get_llm(tools=tools)

    messages = [
        SystemMessage(content=RETRIEVER_SYSTEM),
        *state["messages"][-5:],  # last 5 messages for context
    ]

    response = llm.invoke(messages)

    tool_calls_made = state.get("tool_calls_made", 0)
    retrieved = list(state.get("retrieved_context", []))

    if response.content:
        retrieved.append(response.content)

    return {
        "messages": [response],
        "retrieved_context": retrieved,
        "tool_calls_made": tool_calls_made + 1,
    }


# ── Node: Analyzer Agent ──────────────────────────────────────────────────────

ANALYZER_SYSTEM = """You are an analytical specialist. You synthesize information from multiple sources, identify patterns, and provide deep analysis.

You have access to the knowledge base via search tools. Use them to gather comprehensive information before analyzing.
Your analysis should:
- Cross-reference multiple sources where relevant
- Identify key patterns or insights
- Highlight any conflicting information
- Provide a structured, well-reasoned conclusion
"""

def analyzer_node(state: AgentState) -> dict:
    """Performs deep analysis using retrieved context + additional searches."""
    tools = get_tools()
    llm = _get_llm(tools=tools)

    context_block = ""
    if state.get("retrieved_context"):
        context_block = "\n\n[Previously retrieved context]\n" + "\n---\n".join(
            state["retrieved_context"]
        )

    messages = [
        SystemMessage(content=ANALYZER_SYSTEM + context_block),
        *state["messages"][-5:],
    ]

    response = llm.invoke(messages)
    return {
        "messages": [response],
        "tool_calls_made": state.get("tool_calls_made", 0) + 1,
    }


# ── Node: Responder ───────────────────────────────────────────────────────────

RESPONDER_SYSTEM = """You are the final response synthesizer. Given the conversation history and any retrieved context, produce a clear, helpful, and accurate final answer to the user.

Be concise but complete. If you're drawing on retrieved information, cite the source naturally in your response.
"""

def responder_node(state: AgentState) -> dict:
    """Synthesizes the final response from accumulated context."""
    llm = _get_llm()

    context_block = ""
    if state.get("retrieved_context"):
        context_block = "\n\n[Context from knowledge base]\n" + "\n---\n".join(
            state["retrieved_context"]
        )

    messages = [
        SystemMessage(content=RESPONDER_SYSTEM + context_block),
        *state["messages"][-8:],
    ]

    response = llm.invoke(messages)
    return {
        "messages": [response],
        "final_response": response.content,
    }


# ── Routing logic ─────────────────────────────────────────────────────────────

def route_from_orchestrator(state: AgentState) -> Literal["retriever", "analyzer", "responder"]:
    intent = state.get("intent", "respond")
    if intent == "retrieve":
        return "retriever"
    elif intent == "analyze":
        return "analyzer"
    return "responder"


def route_after_agent(state: AgentState) -> Literal["tools", "responder", END]:
    """After retriever/analyzer: if tool calls pending, execute them. Otherwise, go to responder."""
    messages = state["messages"]
    last = messages[-1] if messages else None

    # Cap tool call depth to prevent infinite loops
    if state.get("tool_calls_made", 0) >= 3:
        return "responder"

    if last and hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    return "responder"


# ── Graph assembly ─────────────────────────────────────────────────────────────

def build_graph():
    """Assemble and compile the LangGraph agent graph."""
    tools = get_tools()
    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("analyzer", analyzer_node)
    graph.add_node("responder", responder_node)
    graph.add_node("tools", tool_node)

    # Entry point
    graph.add_edge(START, "orchestrator")

    # Orchestrator → specialist
    graph.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {
            "retriever": "retriever",
            "analyzer": "analyzer",
            "responder": "responder",
        },
    )

    # Specialists → tools or responder
    graph.add_conditional_edges("retriever", route_after_agent,
                                {"tools": "tools", "responder": "responder", END: END})
    graph.add_conditional_edges("analyzer", route_after_agent,
                                {"tools": "tools", "responder": "responder", END: END})

    # Tools loop back to the agent that called them
    # (LangGraph ToolNode routes back to the calling node automatically with prebuilt)
    graph.add_edge("tools", "responder")

    # Responder always ends
    graph.add_edge("responder", END)

    return graph.compile()
