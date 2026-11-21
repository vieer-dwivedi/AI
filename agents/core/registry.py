"""Plug-in contract between the orchestrator and the agents.

An agent plugs into the orchestrator by providing an `AgentSpec`:
  name         short lowercase id ("dasha")
  description  when the router should pick this agent (the router LLM reads this text)
  run          async (text, thread_id) -> answer text; thread_id lets the agent keep its own memory
  wants_birth  True if the agent uses the user's birth details. The orchestrator then calls
               run(text, thread_id, birth=<dict>) with the details from the graph state (only when the state has
               them), so such an agent's `run` must accept a `birth` keyword. Default False: run(text, thread_id).

`AgentRegistry` holds the specs. The graph builder creates one node + one edge per registered agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Awaitable, Callable, Iterator

RESERVED_NAMES = {"none", "route", "direct"}


@dataclass(frozen=True)
class AgentSpec:
    name: str
    description: str
    run: Callable[..., Awaitable[str]]
    wants_birth: bool = False


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, AgentSpec] = {}

    def register(self, spec: AgentSpec) -> AgentSpec:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", spec.name):
            raise ValueError(f"Agent name '{spec.name}' must be lowercase letters, digits or underscore.")
        if spec.name in RESERVED_NAMES:
            raise ValueError(f"Agent name '{spec.name}' is reserved.")
        if spec.name in self._agents:
            raise ValueError(f"Agent '{spec.name}' is already registered.")
        if not spec.description.strip():
            raise ValueError(f"Agent '{spec.name}' needs a description so the router knows when to use it.")
        self._agents[spec.name] = spec
        return spec

    def get(self, name: str) -> AgentSpec | None:
        return self._agents.get(name)

    def names(self) -> list[str]:
        return list(self._agents)

    def describe(self) -> str:
        """One line per agent, for prompts."""
        return "\n".join(f"- {s.name}: {s.description}" for s in self._agents.values())

    def __iter__(self) -> Iterator[AgentSpec]:
        return iter(self._agents.values())

    def __len__(self) -> int:
        return len(self._agents)
