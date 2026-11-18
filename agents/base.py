# =====================================================================
# agents/base.py
# In simple words: shared helpers for the three specialist agents.
#  - build_agent(): makes a "deep agent" (LangChain's create_deep_agent)
#    using the model chosen in .env (through init_chat_model).
#  - run_agent():   gives the agent a task and returns its final words.
#  - incident_brief(): the standard "page" text every agent receives.
# =====================================================================
from __future__ import annotations

import logging
from typing import Sequence

from deepagents import create_deep_agent
from langchain_core.tools import BaseTool

from core.llm import get_chat_model
from core.schemas import Incident

log = logging.getLogger(__name__)

# Tool-calling loops can be long; this stops runaway loops.
RECURSION_LIMIT = 60


def build_agent(role: str, tools: Sequence[BaseTool], system_prompt: str):
    """Create one deep agent. `role` picks the model (TRIAGE_LLM_MODEL etc.),
    falling back to LLM_MODEL. The system prompt must contain [ROLE:<role>]."""
    return create_deep_agent(
        model=get_chat_model(role),
        tools=list(tools),
        system_prompt=system_prompt,
        name=f"{role}-agent",
    )


def _text(content) -> str:
    """Final message content may be a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content or "")


async def run_agent(agent, task: str) -> str:
    """Run the agent on a task and return its final text answer."""
    result = await agent.ainvoke({"messages": [{"role": "user", "content": task}]},
                                 config={"recursion_limit": RECURSION_LIMIT})
    return _text(result["messages"][-1].content).strip()


def incident_brief(incident: Incident) -> str:
    """The same 'page' text for every agent, as simple 'key: value' lines."""
    lines = [
        f"incident_id: {incident.id}",
        f"title: {incident.title}",
        f"service: {incident.service}",
        f"environment: {incident.environment}",
        f"alarm: {incident.alarm_name}",
        f"triggered_at: {incident.triggered_at}",
        f"source: {incident.source}",
    ]
    lines += [f"{k}: {v}" for k, v in incident.labels.items()]
    lines.append(f"description: {incident.description}")
    return "\n".join(lines)
