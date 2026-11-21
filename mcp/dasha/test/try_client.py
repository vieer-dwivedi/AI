"""Smoke test: start the Dasha MCP server over stdio and call a few tools (real PyJHora).

Run from anywhere:  python mcp/dasha/test/try_client.py
"""

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = str(Path(__file__).resolve().parent.parent / "dasha_server.py")
STATE = {
    "name": "Test", "date_of_birth": "1990-05-15", "time_of_birth": "10:30",
    "place_name": "Mumbai, IN", "latitude": 19.07, "longitude": 72.87, "timezone": "Asia/Kolkata",
}
CALLS = [
    ("list_dasha_systems", {}),
    ("get_dasha_basis", {"state": STATE}),
    ("get_dasha_balance", {"state": STATE}),
    ("get_current_dasha", {"state": STATE, "depth": 3}),
]


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=[SERVER])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("tools:", len((await session.list_tools()).tools))
            for name, args in CALLS:
                res = await session.call_tool(name, args)
                text = res.content[0].text if res.content else ""
                print(f"\n=== {name} {'(ERROR)' if res.isError else ''}")
                print(text[:1500] + (" ..." if len(text) > 1500 else ""))


asyncio.run(main())
