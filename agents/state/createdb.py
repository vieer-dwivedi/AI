"""Local SQLite database for saving conversations.

Run directly to create the database file:
    python -m agents.state.createdb
"""

import sqlite3
import uuid
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "chat.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Create tables if they do not exist."""
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                user_id         TEXT NOT NULL,
                created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_conversations_user
                ON conversations (user_id, created_at);

            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role            TEXT NOT NULL CHECK (role IN ('human', 'ai')),
                content         TEXT NOT NULL,
                created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages (conversation_id, id);

            CREATE TABLE IF NOT EXISTS birth_profiles (
                user_id       TEXT PRIMARY KEY,
                name          TEXT NOT NULL,
                date_of_birth TEXT NOT NULL,
                time_of_birth TEXT NOT NULL,
                place_name    TEXT NOT NULL,
                latitude      REAL NOT NULL,
                longitude     REAL NOT NULL,
                timezone      TEXT NOT NULL,
                updated_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )


def create_conversation(user_id: str) -> str:
    """Create a new conversation (with a fresh UUID) under the given user."""
    conversation_id = str(uuid.uuid4())
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO conversations (conversation_id, user_id) VALUES (?, ?)",
            (conversation_id, user_id),
        )
    return conversation_id


def get_latest_conversation(user_id: str) -> str | None:
    """Return the user's most recent conversation id, or None if they have none."""
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT conversation_id FROM conversations
            WHERE user_id = ?
            ORDER BY created_at DESC, rowid DESC
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()
    return row[0] if row else None


def save_message(conversation_id: str, role: str, content: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
            (conversation_id, role, content),
        )


def load_messages(conversation_id: str) -> list[tuple[str, str]]:
    """Return [(role, content), ...] for a conversation in order."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
    return rows


BIRTH_FIELDS = ("name", "date_of_birth", "time_of_birth", "place_name", "latitude", "longitude", "timezone")


def save_birth_profile(user_id: str, data: dict) -> None:
    """Insert or replace the birth details stored under a user id (one profile per user)."""
    values = [data[k] for k in BIRTH_FIELDS]
    with get_connection() as conn:
        conn.execute(
            f"""
            INSERT INTO birth_profiles (user_id, {', '.join(BIRTH_FIELDS)})
            VALUES (?, {', '.join('?' for _ in BIRTH_FIELDS)})
            ON CONFLICT(user_id) DO UPDATE SET
                {', '.join(f'{k} = excluded.{k}' for k in BIRTH_FIELDS)},
                updated_at = CURRENT_TIMESTAMP
            """,
            (user_id, *values),
        )


def load_birth_profile(user_id: str) -> dict | None:
    """Return the user's stored birth details as a dict, or None if they have none."""
    with get_connection() as conn:
        row = conn.execute(
            f"SELECT {', '.join(BIRTH_FIELDS)} FROM birth_profiles WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    return dict(zip(BIRTH_FIELDS, row)) if row else None


if __name__ == "__main__":
    init_db()
    print(f"Database ready at: {DB_PATH}")
