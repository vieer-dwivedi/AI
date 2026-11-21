"""Console chat through the orchestrator (router + plugged-in agents), with conversations saved in SQLite.

Run from the project root:   python main.py
"""

import asyncio

from agents.state.birth import get_birth_for_user, update_birth_for_user
from agents.state.createdb import (
    create_conversation,
    get_latest_conversation,
    init_db,
    load_messages,
    save_message,
)
from orchestrator import create_orchestrator


def ask_user_id() -> str:
    while True:
        user_id = input("Enter your user id: ").strip()
        if user_id:
            return user_id
        print("User id cannot be empty.")


def ask_yes_no(question: str) -> bool:
    while True:
        answer = input(f"{question} (y/n): ").strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please type y or n.")


def choose_conversation(user_id: str) -> str:
    """Ask whether to start a new conversation; otherwise resume the latest one."""
    if ask_yes_no("Start a new conversation?"):
        conversation_id = create_conversation(user_id)
        print(f"New conversation created: {conversation_id}")
        return conversation_id

    conversation_id = get_latest_conversation(user_id)
    if conversation_id is None:
        conversation_id = create_conversation(user_id)
        print(f"No previous conversation found. New one created: {conversation_id}")
    else:
        print(f"Resuming conversation: {conversation_id}")
    return conversation_id


async def main() -> None:
    init_db()

    user_id = ask_user_id()
    birth = get_birth_for_user(user_id)  # saved under the user id; asked once, then reused
    conversation_id = choose_conversation(user_id)

    history = load_messages(conversation_id)
    for role, content in history:
        print(f"{'You' if role == 'human' else 'AI'}: {content}\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as exc:  # e.g. the Dasha MCP server is not running
        raise SystemExit(f"\nStartup error: {exc}")
