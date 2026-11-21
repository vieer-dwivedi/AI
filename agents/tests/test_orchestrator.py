"""Orchestrator test with NO LLM and NO MCP calls (fake router, fake chat model, fake agents).

Run from the project root:   .venv/bin/python -m agents.tests.test_orchestrator
"""
import asyncio

from langchain_core.messages import AIMessage

from agents.core.graph import Orchestrator, build_orchestrator
from agents.core.nodes import RouteDecision
from agents.core.registry import AgentRegistry, AgentSpec

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""))


class FakeRouter:
    """Routes by keyword; records the prompts it saw."""

    def __init__(self):
        self.prompts = []

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        convo = prompt.split("CONVERSATION (oldest first):")[1].split("\n\nReturn the agent name")[0]
        text = convo.strip().splitlines()[-1].lower()  # latest user message (last line)
        if "ghost" in text:
            return RouteDecision(agent="ghost", reason="invented agent")
        for name in ("dasha", "finance"):
            if name in text:
                return RouteDecision(agent=name, reason=f"mentions {name}")
        return RouteDecision(agent="none", reason="general")


class FakeChat:
    async def ainvoke(self, messages):
        return AIMessage(content="direct-answer")


def make_agent(name, calls):
    async def run(text, thread_id):
        calls.append((name, text, thread_id))
        return f"{name}-answer"

    return AgentSpec(name=name, description=f"{name} test agent", run=run)


async def main():
    calls = []
    registry = AgentRegistry()
    registry.register(make_agent("dasha", calls))
    router = FakeRouter()
    orch = Orchestrator(build_orchestrator(registry, router_llm=router, chat_llm=FakeChat()), registry)

    check("routes to agent", await orch.ask("tell me my dasha", "t1", verbose=False) == "dasha-answer")
    check("agent got text + namespaced thread", calls[-1] == ("dasha", "tell me my dasha", "t1:dasha"), str(calls))
    check("general chat -> direct", await orch.ask("hello there", "t1", verbose=False) == "direct-answer")
    check("agent not called for direct", len(calls) == 1)
    check("unknown agent name -> direct", await orch.ask("ghost question", "t1", verbose=False) == "direct-answer")
    check("router prompt lists agent description", "dasha: dasha test agent" in router.prompts[0])
    check("router sees earlier agent turn", "Assistant[dasha]: dasha-answer" in router.prompts[1])

    # adding a second agent needs no graph change
    registry.register(make_agent("finance", calls))
    orch2 = Orchestrator(build_orchestrator(registry, router_llm=FakeRouter(), chat_llm=FakeChat()), registry)
    check("second agent plugs in", await orch2.ask("finance help", "t2", verbose=False) == "finance-answer")
    check("first agent still works", await orch2.ask("dasha please", "t2", verbose=False) == "dasha-answer")
    nodes = set(orch2.graph.get_graph().nodes)
    check("graph has node per agent", {"route", "direct", "agent_dasha", "agent_finance"} <= nodes, str(nodes))

    # a failing agent must not crash the orchestrator
    async def boom(text, thread_id):
        raise RuntimeError("kaput")

    bad = AgentRegistry()
    bad.register(AgentSpec(name="dasha", description="x", run=boom))
    orch3 = Orchestrator(build_orchestrator(bad, router_llm=FakeRouter(), chat_llm=FakeChat()), bad)
    ans = await orch3.ask("dasha", "t3", verbose=False)
    check("failing agent reported, not raised", "dasha agent failed" in ans and "kaput" in ans, ans)

    # registry validation
    for bad_name in ("none", "Bad Name", "dasha"):
        try:
            registry.register(make_agent(bad_name, calls))
            check(f"registry rejects '{bad_name}'", False)
        except ValueError:
            check(f"registry rejects '{bad_name}'", True)

    print(f"\n{sum(results)}/{len(results)} checks passed")


asyncio.run(main())
