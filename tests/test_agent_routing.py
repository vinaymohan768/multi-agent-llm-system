"""
Tests for LangGraph routing logic — no LLM calls needed.
"""

import pytest
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage, AIMessage

from agents.graph import route_from_orchestrator, route_after_agent, AgentState


def make_state(intent=None, tool_calls_made=0, last_message=None) -> AgentState:
    messages = [HumanMessage(content="test")]
    if last_message:
        messages.append(last_message)
    return AgentState(
        messages=messages,
        session_id="test-session",
        intent=intent,
        retrieved_context=[],
        tool_calls_made=tool_calls_made,
        final_response=None,
    )


# ── route_from_orchestrator ───────────────────────────────────────────────────

def test_route_retrieve():
    state = make_state(intent="retrieve")
    assert route_from_orchestrator(state) == "retriever"


def test_route_analyze():
    state = make_state(intent="analyze")
    assert route_from_orchestrator(state) == "analyzer"


def test_route_respond():
    state = make_state(intent="respond")
    assert route_from_orchestrator(state) == "responder"


def test_route_defaults_to_responder_on_unknown_intent():
    state = make_state(intent="something_unexpected")
    assert route_from_orchestrator(state) == "responder"


def test_route_defaults_to_responder_on_none():
    state = make_state(intent=None)
    assert route_from_orchestrator(state) == "responder"


# ── route_after_agent ─────────────────────────────────────────────────────────

def test_routes_to_tools_when_tool_calls_pending():
    ai_msg = MagicMock(spec=AIMessage)
    ai_msg.tool_calls = [{"name": "search_knowledge_base", "args": {"query": "test"}}]

    state = make_state(tool_calls_made=0, last_message=ai_msg)
    assert route_after_agent(state) == "tools"


def test_routes_to_responder_when_no_tool_calls():
    ai_msg = MagicMock(spec=AIMessage)
    ai_msg.tool_calls = []

    state = make_state(tool_calls_made=1, last_message=ai_msg)
    assert route_after_agent(state) == "responder"


def test_routes_to_responder_when_depth_cap_reached():
    ai_msg = MagicMock(spec=AIMessage)
    ai_msg.tool_calls = [{"name": "search_knowledge_base", "args": {}}]

    # tool_calls_made >= 3 → hard cap, go to responder regardless
    state = make_state(tool_calls_made=3, last_message=ai_msg)
    assert route_after_agent(state) == "responder"


def test_routes_to_responder_on_empty_messages():
    state = AgentState(
        messages=[],
        session_id="s",
        intent=None,
        retrieved_context=[],
        tool_calls_made=0,
        final_response=None,
    )
    assert route_after_agent(state) == "responder"
