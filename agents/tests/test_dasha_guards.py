"""Offline checks of the Dasha agent's code guards (NO LLM calls, NO MCP server).

Run from the project root:   .venv/bin/python -m agents.tests.test_dasha_guards
"""
from langchain_core.messages import HumanMessage, ToolMessage

from agents import dasha
from agents.dasha import DashaAgent, REMEMBER_TOOL, _plan_block, _ungrounded, _ungrounded_lords

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""))


BIRTH = {"date_of_birth": "1990-05-15", "time_of_birth": "10:30", "timezone": "Asia/Kolkata"}
ok_tool = ToolMessage(content='{"mahadasha": "Rahu♄", "start": "2024-01-01", "end": "2042-01-01"}',
                      tool_call_id="1", name="get_current_dasha", status="success")
bad_tool = ToolMessage(content="Error: Saturn 2031-09-09", tool_call_id="2", name="get_current_dasha", status="error")
user = HumanMessage(content="Which dasha am I running? Born 1990-05-15")
msgs = [user, ok_tool, bad_tool]

# ---- dates / years ------------------------------------------------------------------------------
d, y = _ungrounded(msgs, "Rahu runs 2024-01-01 to 2042-01-01.", BIRTH)
check("dates from a tool result are grounded", not d and not y, f"{d} {y}")
d, y = _ungrounded(msgs, "It ends on 2043-03-03 and again in 2050.", BIRTH)
check("invented date and year are flagged", d == ["2043-03-03"] and y == ["2050"], f"{d} {y}")
d, _ = _ungrounded(msgs, "Saturn ends 2031-09-09", BIRTH)
check("a failed tool result is not evidence", d == ["2031-09-09"], str(d))

# ---- lords ---------------------------------------------------------------------------------------
check("lord from a tool result is grounded", _ungrounded_lords(msgs, "Rahu Mahadasha is running.") == [])
check("invented lord is flagged", _ungrounded_lords(msgs, "You are in Venus Mahadasha, Mars antardasha.") == ["Mars", "Venus"])
check("lord from a failed tool is not evidence", _ungrounded_lords(msgs, "Saturn Mahadasha") == ["Saturn"])
check("lord the user named is allowed", _ungrounded_lords([HumanMessage(content="How long is the Venus dasha?")], "Venus Mahadasha lasts 20 years.") == [])
check("no period claim, no flag", _ungrounded_lords(msgs, "Mars is a planet.") == [])

# ---- blocked calls -------------------------------------------------------------------------------
check("re-saving unchanged birth is blocked", DashaAgent._blocked(REMEMBER_TOOL, BIRTH, {"birth_or_options_changed": False}) is not None)
check("changed birth may be saved", DashaAgent._blocked(REMEMBER_TOOL, BIRTH, {"birth_or_options_changed": True}) is None)
check("first save is allowed", DashaAgent._blocked(REMEMBER_TOOL, None, {"birth_or_options_changed": False}) is None)
check("dasha tool blocked while info is missing", DashaAgent._blocked("get_dasha_at_age", BIRTH, {"missing_info": ["age"]}) is not None)
check("dasha tool allowed when nothing is missing", DashaAgent._blocked("get_current_dasha", BIRTH, {"missing_info": []}) is None)

# ---- scope routing / refusal ---------------------------------------------------------------------
check("out of scope -> refuse", DashaAgent._after_plan({"plan": {"in_scope": False}}) == "refuse")
check("in scope -> agent", DashaAgent._after_plan({"plan": {"in_scope": True}}) == "agent")
check("no plan -> agent (prompt rules + verifier still apply)", DashaAgent._after_plan({"plan": None}) == "agent")
out = DashaAgent._refuse({"plan": {"in_scope": False, "refusal": "x" * 600}})
check("overlong refusal replaced by the fixed one", out["messages"][0].content == dasha.DEFAULT_REFUSAL and out["verdict"] == "refused")

# ---- plan block ----------------------------------------------------------------------------------
block = _plan_block({"intent": "current dasha", "missing_info": ["birth time"], "steps": ["get_current_dasha() - now"]})
check("plan block tells the agent to ask, not call tools", "MISSING INFORMATION" in block and "Call NO dasha tool" in block)

print(f"\n{sum(results)}/{len(results)} checks passed")
raise SystemExit(0 if all(results) else 1)
