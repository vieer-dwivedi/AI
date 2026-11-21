"""Real end-to-end test of the Dasha agent (needs .env with a working API key and internet).

Run from the project root:   .venv/bin/python -m agents.tests.try_dasha_agent
"""

import asyncio

from agents.dasha import dasha_agent

QUESTIONS = [
    "I was born on 1990-05-15 at 10:30 in Mumbai, India. Which Vimsottari mahadasha, antardasha and pratyantardasha am I running today?",
    "When does my current antardasha end and which one comes next?",
    "List all my Jupiter antardashas with dates.",
]


async def main() -> None:
    async with dasha_agent() as agent:
        for q in QUESTIONS:
            print(f"\n=== YOU: {q}")
            print("=== AGENT:", await agent.ask(q, thread_id="test"))


asyncio.run(main())
