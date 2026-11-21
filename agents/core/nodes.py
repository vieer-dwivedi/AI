"""Orchestrator nodes.

  make_router      LLM decides which registered agent (or "none") should handle the latest user message
  make_direct      general answer when no agent fits
  make_agent_node  factory: wraps one AgentSpec as a graph node (one node per agent)
  select_route     conditional-edge function: route name -> next node
"""

from __future__ import annotations

from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from agents.core.llm import llm as default_llm
from agents.core.registry import AgentRegistry, AgentSpec

NONE = "none"
ROUTER_HISTORY = 8  # last messages the router looks at
DIRECT_HISTORY = 10  # last messages the direct node looks at
CLIP = 600  # characters per message shown to the router


class OrchestratorState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    route: str  # agent name or "none"
    route_reason: str
    birth: Optional[dict]  # the user's birth details (see agents/state/birth.py); handed to agents with wants_birth


class RouteDecision(BaseModel):
    """Who should handle the user's latest message."""

    agent: str = Field(description="Exact name of one available agent, or 'none' if no agent clearly fits.")
    reason: str = Field(default="", description="One short sentence explaining the choice.")


# ----------------------------------------------------------------------------------------------
def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(b.get("text", "")) if isinstance(b, dict) else str(b) if isinstance(b, str) else str(getattr(b, "text", ""))
            for b in content
        )
    return str(content)


def _last_user_text(messages: list[AnyMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return _text(m.content).strip()
    return ""


def _transcript(messages: list[AnyMessage], limit: int) -> str:
    lines = []
    for m in messages[-limit:]:
        who = "User" if isinstance(m, HumanMessage) else f"Assistant[{m.name}]" if getattr(m, "name", None) else "Assistant"
        text = _text(m.content).strip()
        lines.append(f"{who}: {text if len(text) <= CLIP else text[:CLIP] + '...'}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
def make_router(registry: AgentRegistry, router_llm: Any = None):
    """Node that sets state['route'] to an agent name or 'none'."""
    structured = (router_llm or default_llm).with_structured_output(RouteDecision)

    async def route(state: OrchestratorState) -> dict:
        messages = state["messages"]
        prompt = (
            "You are the router of a multi-agent assistant. Decide who should handle the user's LATEST message.\n\n"
            f"AVAILABLE AGENTS:\n{registry.describe()}\n- {NONE}: greetings, general chat, questions about the "
            "assistant itself, or anything no agent clearly covers.\n\n"
            "RULES\n"
            "1. Pick an agent only if its description clearly covers the request.\n"
            "2. If the previous assistant message came from an agent (shown as Assistant[name]) and the latest user "
            "message answers its question or continues the same topic, pick that same agent.\n"
            "3. Never invent agent names. When unsure, pick 'none'.\n\n"
            f"CONVERSATION (oldest first):\n{_transcript(messages, ROUTER_HISTORY)}\n\n"
            "Return the agent name and a one-sentence reason."
        )
        decision: RouteDecision = await structured.ainvoke(prompt)
        name = (decision.agent or NONE).strip().lower()
        if name not in registry.names():
            name = NONE
        return {"route": name, "route_reason": decision.reason}

    return route


def make_direct(registry: AgentRegistry, chat_llm: Any = None):
    """Node that answers directly when no agent was chosen."""
    model = chat_llm or default_llm

    async def direct(state: OrchestratorState) -> dict:
        system = SystemMessage(
            content=(
                "You are a helpful, concise assistant. The user's latest message is general conversation that no "
                "specialist agent handles, so answer it yourself. Do not claim abilities you do not have. If it "
                "helps, you may mention that these specialist agents exist:\n" + (registry.describe() or "(none)")
            )
        )
        history = [
            HumanMessage(content=_text(m.content)) if isinstance(m, HumanMessage)
            else AIMessage(content=_text(m.content))
            for m in state["messages"][-DIRECT_HISTORY:]
            if isinstance(m, (HumanMessage, AIMessage)) and _text(m.content).strip()
        ]
        ai = await model.ainvoke([system, *history])
        return {"messages": [AIMessage(content=_text(ai.content).strip(), name="assistant")]}

    return direct


def make_agent_node(spec: AgentSpec):
    """Node that hands the latest user message to one plugged-in agent."""

    async def run_agent(state: OrchestratorState, config: RunnableConfig) -> dict:
        question = _last_user_text(state["messages"])
        thread = (config.get("configurable") or {}).get("thread_id", "default")
        birth = state.get("birth")
        extra = {"birth": birth} if (spec.wants_birth and birth) else {}  # birth details travel via the state
        try:
            answer = (await spec.run(question, f"{thread}:{spec.name}", **extra) or "").strip()
        except Exception as exc:  # one failing agent must not crash the orchestrator
            answer = f"The {spec.name} agent failed: {type(exc).__name__}: {exc}"
        return {"messages": [AIMessage(content=answer or f"The {spec.name} agent returned no answer.", name=spec.name)]}

    return run_agent


def select_route(registry: AgentRegistry):
    """Conditional-edge function: maps state['route'] to the next node name."""
    names = set(registry.names())

    def pick(state: OrchestratorState) -> str:
        route: Optional[str] = state.get("route")
        return f"agent_{route}" if route in names else "direct"

    return pick
