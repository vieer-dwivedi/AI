"""Live test of the Dasha agent with birth details that YOU enter (needs the Dasha MCP server + a working API key).

Start the Dasha MCP server first in another terminal:   python mcp/dasha/main.py --http
Then, from the project root:

    .venv/bin/python -m agents.tests.try_agent_birth                  type birth details, then ask questions
    .venv/bin/python -m agents.tests.try_agent_birth --user alice     load / save the details under user id 'alice'
    .venv/bin/python -m agents.tests.try_agent_birth --sample         skip typing, use a built-in sample birth
    .venv/bin/python -m agents.tests.try_agent_birth --state-only     NO LLM: only check the details reach the agent
                                                                      state and pass the MCP validation (no quota used)
    .venv/bin/python -m agents.tests.try_agent_birth --orchestrator   go through the orchestrator (router + agents)
    .venv/bin/python -m agents.tests.try_agent_birth -q "my current dasha?" -q "next antardasha?"   fixed questions

What it proves: the birth details are put into STATE (`birth=` argument), the agent validates them once, shows your
lagna / Moon sign, and then answers WITHOUT asking you for the details again.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import date, time

from agents.state.birth import BirthDetails, get_birth_for_user, prompt_birth_details

SAMPLE = BirthDetails(
    name="Sample", date_of_birth=date(1990, 5, 15), time_of_birth=time(10, 30),
    place_name="Mumbai, India", latitude=19.07, longitude=72.87, timezone="Asia/Kolkata",
)
DEFAULT_QUESTION = "Which Vimsottari mahadasha, antardasha and pratyantardasha am I running today?"


def collect_birth(args: argparse.Namespace) -> BirthDetails:
    """Sample, saved profile under --user, or typed in."""
    if args.sample:
        return SAMPLE
    if args.user:
        return get_birth_for_user(args.user)
    return prompt_birth_details()


async def check_state_only(agent, thread_id: str, birth_state: dict) -> bool:
    """No LLM: hand the details to the agent exactly like main.py does and show what ends up in its state."""
    config = {"configurable": {"thread_id": thread_id}}
    update, error = await agent.seed_birth(config, birth_state)
    if error:
        print(f"FAIL  the agent rejected the birth details: {error}")
        return False
    summary = json.loads(update.pop("_check"))
    print("PASS  birth details accepted by the agent (validated through the MCP server)")
    print("      state['birth']        =", json.dumps(update["birth"], ensure_ascii=False))
    print(f"      lagna={summary.get('lagna')}  moon={summary.get('moon')}")
    print(f"      timezone offset at birth = {summary.get('timezone_offset_hours')} h")
    return True


async def chat(ask, thread_id: str, birth_state: dict, questions: list[str]) -> None:
    """Ask the fixed questions, then (unless -q was used) keep chatting until an empty line."""
    queue = list(questions) or [DEFAULT_QUESTION]
    interactive = not questions
    while True:
        if queue:
            question = queue.pop(0)
            print(f"\nYou: {question}")
        elif interactive:
            question = (await asyncio.to_thread(input, "\nYou (empty line to quit): ")).strip()
            if not question:
                return
        else:
            return
        answer = await ask(question, thread_id, birth_state)
        print(f"\nAgent: {answer}")


async def run(args: argparse.Namespace) -> int:
    """Returns the process exit code (0 = ok)."""
    birth = collect_birth(args)
    birth_state = birth.to_state()
    print("\nBirth details passed to the agent state:")
    print(json.dumps(birth_state, indent=2, ensure_ascii=False))
    thread_id = f"try-birth-{args.user or 'anon'}"

    if args.orchestrator and not args.state_only:
        from orchestrator import create_orchestrator

        async with create_orchestrator() as orchestrator:
            async def ask(q, tid, b):
                return await orchestrator.ask(q, thread_id=tid, birth=b)

            await chat(ask, thread_id, birth_state, args.question)
        return 0

    from agents.dasha import dasha_agent

    async with dasha_agent() as agent:
        if args.state_only:
            return 0 if await check_state_only(agent, thread_id, birth_state) else 1

        async def ask(q, tid, b):
            return await agent.ask(q, thread_id=tid, birth=b)

        await chat(ask, thread_id, birth_state, args.question)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the Dasha agent with birth details you enter.")
    parser.add_argument("--user", help="user id: load the birth details saved under it, or save the ones you type")
    parser.add_argument("--sample", action="store_true", help="use a built-in sample birth instead of typing")
    parser.add_argument("--state-only", action="store_true", help="no LLM: only check the details reach the agent state")
    parser.add_argument("--orchestrator", action="store_true", help="go through the orchestrator instead of the agent alone")
    parser.add_argument("-q", "--question", action="append", default=[], help="question to ask (repeatable)")
    args = parser.parse_args()
    try:
        raise SystemExit(asyncio.run(run(args)))
    except (EOFError, KeyboardInterrupt):
        print("\nBye!")
    except RuntimeError as exc:  # e.g. the Dasha MCP server is not running
        raise SystemExit(f"\nStartup error: {exc}")


if __name__ == "__main__":
    main()
