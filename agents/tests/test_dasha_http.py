"""Dasha agent <-> SEPARATE Dasha MCP server over HTTP (no LLM calls).

Start the server first in another terminal:   python mcp/dasha/main.py --http --port 8765
Then, from the project root:                  DASHA_MCP_URL=http://127.0.0.1:8765/mcp python -m agents.tests.test_dasha_http
"""
import asyncio

from agents.dasha import DASHA_MCP_URL, REMEMBER_TOOL, dasha_agent

BIRTH = {"name": "Test", "date_of_birth": "1990-05-15", "time_of_birth": "10:30", "place_name": "Mumbai, IN",
         "latitude": 19.07, "longitude": 72.87, "timezone": "Asia/Kolkata"}
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""))


async def main():
    print("URL:", DASHA_MCP_URL)

    # server not running -> clear, actionable error (port 1 is never open)
    try:
        async with dasha_agent(url="http://127.0.0.1:1/mcp"):
            check("unreachable -> RuntimeError", False)
    except RuntimeError as exc:
        check("unreachable -> RuntimeError with start command", "mcp/dasha/main.py --http" in str(exc), str(exc)[:150])

    async with dasha_agent() as agent:
        check("13 tools over HTTP", len(agent.tools) == 13, str(len(agent.tools)))
        text, err, birth = await agent._run_call(REMEMBER_TOOL, dict(BIRTH), None)
        check("remember_birth_details", not err and birth is not None, text[:150])
        text, err, _ = await agent._run_call("get_current_dasha", {"on_date": "2026-11-18", "depth": 2}, birth)
        check("current dasha (Rahu / Sun)", not err and "Rahu" in text and "Sun" in text, text[:150])
        text, err, _ = await agent._run_call("get_current_dasha", {"system": "nope"}, birth)
        check("tool error flagged", err, text[:120])

    print(f"\n{sum(results)}/{len(results)} checks passed")


asyncio.run(main())
