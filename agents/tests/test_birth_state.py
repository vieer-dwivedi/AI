"""Birth details: validation, prompts, per-user storage, and passing through STATE to the agents.
NO LLM calls and NO MCP server (fake router / chat / tools, temp SQLite file).

Run from the project root:   .venv/bin/python -m agents.tests.test_birth_state
"""
import asyncio
import json
import tempfile
from datetime import date, time
from pathlib import Path

from langchain_core.messages import AIMessage

import agents.state.createdb as db
from agents.core.graph import Orchestrator, build_orchestrator
from agents.core.nodes import RouteDecision
from agents.core.registry import AgentRegistry, AgentSpec
from agents.dasha import DashaAgent
from agents.state.birth import (
    BirthDetails,
    get_birth_for_user,
    load_birth,
    parse_date,
    parse_time,
    prompt_birth_details,
    save_birth,
)

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""))


def raises(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return True
    return False


def scripted(answers):
    """input_fn that returns the scripted answers in order and records the prompts it was shown."""
    it = iter(answers)
    prompts = []

    def fn(prompt):
        prompts.append(prompt)
        return next(it)

    fn.prompts = prompts
    return fn


MUMBAI = {"address": "Mumbai, Maharashtra, India", "latitude": 19.0760, "longitude": 72.8777, "timezone": "Asia/Kolkata"}
SAMPLE = BirthDetails(name="Asha", date_of_birth=date(1990, 5, 15), time_of_birth=time(10, 30),
                      place_name="Mumbai, India", latitude=19.07, longitude=72.87, timezone="Asia/Kolkata")


def make_lookup(table):
    calls = []

    def lookup(place):
        calls.append(place)
        return table.get(place)

    lookup.calls = calls
    return lookup


class FakeRouter:
    def __init__(self):
        pass

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, prompt):
        convo = prompt.split("CONVERSATION (oldest first):")[1].split("\n\nReturn the agent name")[0]
        text = convo.strip().splitlines()[-1].lower()
        for name in ("dasha", "plain"):
            if name in text:
                return RouteDecision(agent=name, reason="keyword")
        return RouteDecision(agent="none", reason="general")


class FakeChat:
    async def ainvoke(self, messages):
        return AIMessage(content="direct")


class FakeTool:
    """Stands in for the MCP tool get_dasha_basis (which validates the birth details)."""

    def __init__(self):
        self.name = "get_dasha_basis"
        self.description = "basis"
        self.args_schema = {"type": "object", "properties": {"state": {"type": "object"}}, "required": ["state"]}
        self.calls = []
        self.fail = False

    async def ainvoke(self, args):
        self.calls.append(args)
        if self.fail:
            raise RuntimeError("Unknown IANA timezone")
        return json.dumps({"chart": "D1", "lagna": {"sign": "Gemini"}, "moon": {"sign": "Taurus"},
                           "dasha_basis": {"vimsottari_starting_lord": "Rahu"}, "person": {"timezone_offset_hours": 5.5}})


async def main():
    # ------------------------------------------------------------------ parsing + model validation
    check("date DD-MM-YYYY", parse_date("15-05-1990") == date(1990, 5, 15))
    check("date YYYY-MM-DD", parse_date("1990-05-15") == date(1990, 5, 15))
    check("date DD/MM/YYYY", parse_date("15/05/1990") == date(1990, 5, 15))
    check("impossible date rejected", raises(parse_date, "31-02-1990"))
    check("future date rejected", raises(parse_date, "01-01-2999"))
    check("year before 1900 rejected", raises(parse_date, "01-01-1850"))
    check("time 24h", parse_time("18:45") == time(18, 45))
    check("time 12h pm", parse_time("6:45 pm") == time(18, 45))
    check("time 12h am", parse_time("6:45AM") == time(6, 45))
    check("time 25:00 rejected", raises(parse_time, "25:00"))
    check("bad latitude rejected", raises(lambda: BirthDetails(**{**SAMPLE.to_state(), "latitude": 95})))
    check("bad timezone rejected", raises(lambda: BirthDetails(**{**SAMPLE.to_state(), "timezone": "Mars/Base"})))
    check("blank place rejected", raises(lambda: BirthDetails(**{**SAMPLE.to_state(), "place_name": "  "})))
    state = SAMPLE.to_state()
    check("to_state keys = what the Dasha agent saves",
          set(state) == {"name", "date_of_birth", "time_of_birth", "place_name", "latitude", "longitude", "timezone"})
    check("to_state is plain ISO text", state["date_of_birth"] == "1990-05-15" and state["time_of_birth"] == "10:30", str(state))
    check("to_state round trip", BirthDetails(**state) == SAMPLE)

    # ------------------------------------------------------------------ prompts (scripted keyboard)
    inp = scripted(["Asha", "15-05-1990", "10:30", "Mumbai", "y"])
    b = prompt_birth_details(input_fn=inp, print_fn=lambda *a: None, lookup=make_lookup({"Mumbai": MUMBAI}))
    check("happy path", b.to_state() == {**state, "place_name": "Mumbai", "latitude": 19.076, "longitude": 72.8777}, str(b.to_state()))

    inp = scripted(["Ravi", "31-02-1990", "1990-05-15", "25:00", "6:45 pm", "Nowhere", "y", "abc", "19.07", "72.87", ""])
    b = prompt_birth_details(input_fn=inp, print_fn=lambda *a: None, lookup=make_lookup({}))
    check("invalid date / time are asked again", b.date_of_birth == date(1990, 5, 15) and b.time_of_birth == time(18, 45))
    check("unknown place -> manual lat/lon, timezone guessed from coordinates",
          (b.latitude, b.longitude, b.timezone) == (19.07, 72.87, "Asia/Kolkata"), str((b.latitude, b.longitude, b.timezone)))

    lookup = make_lookup({"Paris": {**MUMBAI, "address": "Paris, Texas, USA"},
                          "Paris, France": {"address": "Paris, France", "latitude": 48.85, "longitude": 2.35, "timezone": "Europe/Paris"}})
    inp = scripted(["Luc", "01-01-1980", "08:00", "Paris", "n", "Paris, France", "y"])
    b = prompt_birth_details(input_fn=inp, print_fn=lambda *a: None, lookup=lookup)
    check("'n' to the found place asks again", lookup.calls == ["Paris", "Paris, France"] and b.timezone == "Europe/Paris", str(lookup.calls))

    lookup = make_lookup({})
    inp = scripted(["", "", "", ""])  # Enter four times: keep everything
    b = prompt_birth_details(existing=SAMPLE, input_fn=inp, print_fn=lambda *a: None, lookup=lookup)
    check("Enter keeps existing values, no new lookup", b == SAMPLE and lookup.calls == [])

    # ------------------------------------------------------------------ storage under the user id
    with tempfile.TemporaryDirectory() as tmp:
        db.DB_PATH = Path(tmp) / "t.db"
        db.init_db()
        check("no profile yet", load_birth("u1") is None)
        save_birth("u1", SAMPLE)
        check("save + load round trip", load_birth("u1") == SAMPLE)
        save_birth("u1", BirthDetails(**{**SAMPLE.to_state(), "name": "Renamed"}))
        check("saving again replaces (one profile per user)", load_birth("u1").name == "Renamed")
        check("other users are separate", load_birth("u2") is None)

        lookup = make_lookup({"Mumbai": MUMBAI})
        inp = scripted(["Asha", "15-05-1990", "10:30", "Mumbai", "y"])
        got = get_birth_for_user("u3", input_fn=inp, print_fn=lambda *a: None, lookup=lookup)
        check("new user: asked once and saved", load_birth("u3") == got and got.name == "Asha")

        inp = scripted(["y"])
        again = get_birth_for_user("u3", input_fn=inp, print_fn=lambda *a: None, lookup=lookup)
        check("returning user: saved details reused, nothing re-asked", again == got and len(inp.prompts) == 1 and lookup.calls == ["Mumbai"])

        inp = scripted(["n", "Bob", "", "", ""])
        changed = get_birth_for_user("u3", input_fn=inp, print_fn=lambda *a: None, lookup=lookup)
        check("returning user can change details; saved", changed.name == "Bob" and load_birth("u3").name == "Bob")

    # ------------------------------------------------------------------ birth travels through the orchestrator STATE
    seen = []

    async def dasha_run(text, thread_id, birth=None):
        seen.append(("dasha", thread_id, birth))
        return "dasha-answer"

    async def plain_run(text, thread_id):  # an agent that knows nothing about birth details
        seen.append(("plain", thread_id))
        return "plain-answer"

    registry = AgentRegistry()
    registry.register(AgentSpec(name="dasha", description="d", run=dasha_run, wants_birth=True))
    registry.register(AgentSpec(name="plain", description="p", run=plain_run))
    orch = Orchestrator(build_orchestrator(registry, router_llm=FakeRouter(), chat_llm=FakeChat()), registry)

    ans = await orch.ask("dasha now", "t1", verbose=False, birth=state)
    check("agent with wants_birth gets the details from state", ans == "dasha-answer" and seen[-1] == ("dasha", "t1:dasha", state), str(seen))
    await orch.ask("dasha again", "t1", verbose=False)
    check("details stay in the thread state (no need to pass them again)", seen[-1][2] == state, str(seen[-1]))
    ans = await orch.ask("plain please", "t1", verbose=False, birth=state)
    check("agent without wants_birth is called as before", ans == "plain-answer" and seen[-1] == ("plain", "t1:plain"), str(seen[-1]))
    await orch.ask("dasha fresh thread", "t2", verbose=False)
    check("no details in state -> agent gets none (it will ask the user)", seen[-1][2] is None, str(seen[-1]))
    snapshot = await orch.graph.aget_state({"configurable": {"thread_id": "t1"}})
    check("orchestrator state holds `birth`", snapshot.values.get("birth") == state)

    # ------------------------------------------------------------------ Dasha agent: seeding its own state
    tool = FakeTool()
    agent = DashaAgent([tool])
    cfg = {"configurable": {"thread_id": "s1"}}
    upd, err = await agent.seed_birth(cfg, state)
    check("seed: details validated through the MCP tool", err is None and len(tool.calls) == 1 and tool.calls[0] == {"state": state}, str(tool.calls))
    check("seed: birth + seeded_birth in the state update", upd["birth"] == state and upd["seeded_birth"] == state, str(upd.keys()))
    check("seed: check data carries lagna", json.loads(upd["_check"])["lagna"] == "Gemini")

    saved = {k: v for k, v in upd.items() if k != "_check"}
    await agent.graph.aupdate_state(cfg, saved, as_node="wrap_up")  # same as a finished turn
    upd2, err2 = await agent.seed_birth(cfg, state)
    check("same details again: nothing re-validated", upd2 == {} and err2 is None and len(tool.calls) == 1)

    new_state = {**state, "name": "Asha K"}
    upd3, _ = await agent.seed_birth(cfg, new_state)
    check("changed profile replaces the saved details", upd3["birth"]["name"] == "Asha K" and len(tool.calls) == 2)

    await agent.graph.aupdate_state(cfg, {"birth": {**state, "place_name": "Pune"}}, as_node="wrap_up")  # user corrected it in chat
    upd4, _ = await agent.seed_birth(cfg, state)
    check("details corrected in chat are not overwritten by the unchanged profile", upd4 == {})

    tool.fail = True
    upd5, err5 = await agent.seed_birth({"configurable": {"thread_id": "s2"}}, state)
    check("rejected details -> clear error, nothing seeded", upd5 == {} and err5 and "could not use" in err5, str(err5))

    print(f"\n{sum(results)}/{len(results)} checks passed")


asyncio.run(main())
