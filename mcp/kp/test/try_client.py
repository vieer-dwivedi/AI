"""Smoke test: start the KP MCP server over stdio and call a few tools through a real MCP client.

Run from anywhere:  .venv/bin/python mcp/kp/test/try_client.py
"""

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = str(Path(__file__).resolve().parent.parent / "main.py")
STATE = {
    "name": "Test", "date_of_birth": "1990-05-15", "time_of_birth": "10:30",
    "place_name": "Mumbai, IN", "latitude": 19.07, "longitude": 72.87, "timezone": "Asia/Kolkata",
}
MOMENT = {"place_name": "Mumbai, IN", "latitude": 19.07, "longitude": 72.87, "timezone": "Asia/Kolkata",
          "date": "2026-11-20", "time": "15:00"}
CALLS = [
    ("lookup_longitude", {"longitude": 108.99944}),
    ("get_cusps", {"state": STATE}),
    ("judge_event", {"state": STATE, "event": "service"}),
    ("get_current_kp_dasha", {"state": STATE, "on_date": "2026-11-20", "depth": 3}),
    ("get_ruling_planets", {"moment": MOMENT, "birth": STATE}),
    ("judge_horary", {"moment": MOMENT, "number": 100, "event": "property_purchase"}),
    ("judge_event", {"state": STATE, "event": "not_an_event"}),
]


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=[SERVER])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            print("tools:", len(tools))
            for name, args in CALLS:
                res = await session.call_tool(name, args)
                text = res.content[0].text if res.content else ""
                print(f"\n=== {name} {'(ERROR)' if res.isError else ''}")
                print(text[:700] + (" ..." if len(text) > 700 else ""))


asyncio.run(main())
