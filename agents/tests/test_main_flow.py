"""main.py flow test: temp SQLite DB, fake orchestrator (no LLM, no MCP).

Run from the project root:   .venv/bin/python -m agents.tests.test_main_flow
"""
import asyncio
import builtins
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from langchain_core.messages import AIMessage

import agents.state.createdb as db
import agents.state.birth as birth_mod
import main as app
from agents.core.graph import Orchestrator, build_orchestrator
from agents.core.nodes import RouteDecision
from agents.core.registry import AgentRegistry, AgentSpec

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""))


class FakeRouter:
    def __init__(self):
        self.prompts = []

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        return RouteDecision(agent="none", reason="test")


class FakeChat:
    def __init__(self):
        self.fail = False

    async def ainvoke(self, messages):
        if self.fail:
            raise RuntimeError("quota")
        return AIMessage(content=f"echo:{messages[-1].content}")


router, chat = FakeRouter(), FakeChat()


def fake_orchestrator():
    reg = AgentRegistry()

    async def run(text, tid):
        return "x"

    reg.register(AgentSpec(name="dasha", description="d", run=run))
    return Orchestrator(build_orchestrator(reg, router_llm=router, chat_llm=chat), reg)


@asynccontextmanager
async def fake_create():
    yield fake_orchestrator()


def run_main(inputs):
    it = iter(inputs)
    original = builtins.input
    builtins.input = lambda *a, **k: next(it)
    try:
        asyncio.run(app.main())
    finally:
        builtins.input = original


with tempfile.TemporaryDirectory() as tmp:
    db.DB_PATH = Path(tmp) / "t.db"
    app.create_orchestrator = fake_create
    db.init_db()
    ALICE = birth_mod.BirthDetails(name="Alice", date_of_birth="1990-05-15", time_of_birth="10:30", place_name="Mumbai, India",
                                   latitude=19.07, longitude=72.87, timezone="Asia/Kolkata")
    birth_mod.save_birth("alice", ALICE)  # alice already has birth details on file

    run_main(["alice", "y", "y", "hello", "second", "exit"])  # user id, use saved birth, new conversation
    cid = db.get_latest_conversation("alice")
    check("conversation saved", cid is not None)
    check("both turns saved in order", db.load_messages(cid) == [
        ("human", "hello"), ("ai", "echo:hello"), ("human", "second"), ("ai", "echo:second")], str(db.load_messages(cid)))

    # resume: history is seeded, so the router sees the earlier turns
    router.prompts.clear()
    run_main(["alice", "y", "n", "third", "exit"])
    check("resume keeps same conversation", db.get_latest_conversation("alice") == cid)
    check("router saw earlier turns after resume", "echo:second" in router.prompts[0] and "hello" in router.prompts[0])
    check("resumed turn appended", db.load_messages(cid)[-2:] == [("human", "third"), ("ai", "echo:third")])

    # a failing turn is not saved, chat continues
    chat.fail = True
    run_main(["alice", "y", "n", "boom", "exit"])
    check("failed turn not saved", db.load_messages(cid)[-1] == ("ai", "echo:third"), str(db.load_messages(cid)[-1]))

    # a first-time user is asked for birth details, which are saved under the user id; /birth changes them
    chat.fail = False
    birth_mod.lookup_place = lambda q: {"address": "Pune, India", "latitude": 18.52, "longitude": 73.86, "timezone": "Asia/Kolkata"}
    run_main(["carol", "Carol", "01-02-1985", "06:15", "Pune", "y", "y", "hi", "/birth", "", "", "", "Pune", "exit"])
    saved = birth_mod.load_birth("carol")
    check("first-time user: birth details asked and saved under user id",
          saved is not None and saved.to_state()["date_of_birth"] == "1985-02-01" and saved.timezone == "Asia/Kolkata", str(saved))
    check("/birth in the chat keeps the values on Enter and the chat continues", saved is not None and saved.name == "Carol")
    check("carol's chat turn saved", db.load_messages(db.get_latest_conversation("carol"))[:2] == [("human", "hi"), ("ai", "echo:hi")])

print(f"\n{sum(results)}/{len(results)} checks passed")
