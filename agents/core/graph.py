"""Orchestrator graph.

    START -> route --(agent chosen)--> agent_<name> -> END        (one node per registered agent)
                  '--(none / unknown)--> direct -> END

`build_orchestrator(registry)` creates one node and one edge per registered agent, so adding an agent never
requires editing this file. `Orchestrator` is a thin wrapper that streams progress and returns the answer.
"""

from __future__ import annotations

import sys
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from agents.core.nodes import (
    OrchestratorState,
    make_agent_node,
    make_direct,
    make_router,
    select_route,
)
from agents.core.registry import AgentRegistry

RETRY = RetryPolicy(max_attempts=5, initial_interval=2.0, backoff_factor=2.0, max_interval=30.0)  # API rate limits
RECURSION_LIMIT = 25


def build_orchestrator(
    registry: AgentRegistry,
    *,
    router_llm: Any = None,
    chat_llm: Any = None,
    checkpointer: Any = None,
):
    """Compile the orchestrator graph for the agents currently in `registry`."""
    g = StateGraph(OrchestratorState)
    g.add_node("route", make_router(registry, router_llm), retry_policy=RETRY)
    g.add_node("direct", make_direct(registry, chat_llm), retry_policy=RETRY)

    targets = {"direct": "direct"}
    for spec in registry:
        node = f"agent_{spec.name}"
        g.add_node(node, make_agent_node(spec))  # the agent handles its own retries
        g.add_edge(node, END)
        targets[node] = node

    g.add_edge(START, "route")
    g.add_conditional_edges("route", select_route(registry), targets)
    g.add_edge("direct", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


class Orchestrator:
    """Conversation-level wrapper around the compiled graph."""

    def __init__(self, graph: Any, registry: AgentRegistry) -> None:
        self.graph = graph
        self.registry = registry

    async def seed(self, thread_id: str, history: list[tuple[str, str]]) -> None:
        """Load earlier (role, text) turns, e.g. from the database, into a thread so the router sees them."""
        msgs = [HumanMessage(content=c) if r == "human" else AIMessage(content=c) for r, c in history if c.strip()]
        if msgs:
            config = {"configurable": {"thread_id": thread_id}}
            await self.graph.aupdate_state(config, {"messages": msgs}, as_node="direct")

    async def ask(self, text: str, thread_id: str = "default", verbose: bool = True, birth: dict | None = None) -> str:
        """Send one user message. `birth` (the user's birth details dict) is put into the graph state, from where
        it is handed to every agent that declared wants_birth=True."""
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
        payload: dict[str, Any] = {"messages": [HumanMessage(content=text)]}
        if birth:
            payload["birth"] = birth
        async for update in self.graph.astream(payload, config, stream_mode="updates"):
            route = (update.get("route") or {}) if isinstance(update, dict) else {}
            if verbose and route:
                print(f"  [route] {route.get('route')} - {route.get('route_reason', '')}", file=sys.stderr)
        snapshot = await self.graph.aget_state(config)
        for m in reversed(snapshot.values.get("messages", [])):
            if isinstance(m, AIMessage) and not m.tool_calls:
                return str(m.content).strip() if isinstance(m.content, str) else "".join(
                    b.get("text", "") if isinstance(b, dict) else str(b) for b in m.content
                ).strip()
        return ""
