# =====================================================================
# memory/long_term.py
# In simple words: the agent's LONG-TERM memory of past incidents.
# Every resolved incident is saved in ChromaDB (a vector database). When a
# new incident arrives, the agent can ask "have we seen something like
# this before?" and get the earlier root cause and fix.
# The first time it runs, it loads a few example incidents from
# sample_data/past_incidents.json so there is something to find.
# =====================================================================
from __future__ import annotations

import json
from datetime import datetime, timezone

from langchain_chroma import Chroma
from langchain_core.tools import tool

from core.config import get_settings
from core.embeddings import get_embeddings, spec_slug
from tools.common import jdump, safe_tool

_stores: dict[tuple, Chroma] = {}


def _doc_text(i: dict) -> str:
    """The text we embed: everything useful for matching a future incident."""
    return (f"{i['title']}. Root cause: {i.get('root_cause', '')}. "
            f"Resolution: {i.get('resolution', '')}. Tags: {i.get('tags', '')}")


def _open_store() -> Chroma:
    """Open the Chroma collection (cached). Does NOT add any seed data."""
    s = get_settings()
    name = f"past_incidents_{spec_slug()}"
    key = (str(s.chroma_path), name)
    if key not in _stores:
        _stores[key] = Chroma(collection_name=name, embedding_function=get_embeddings(),
                              persist_directory=str(s.chroma_path),
                              collection_metadata={"hnsw:space": "cosine"})
    return _stores[key]


def _add(inc: dict) -> None:
    meta = {**inc, "saved_at": datetime.now(timezone.utc).isoformat()}
    _open_store().add_texts([_doc_text(inc)], metadatas=[meta], ids=[inc["id"]])


def get_store() -> Chroma:
    """Open the store and, the first time only, load the example incidents.
    Every public function goes through here, so seeding always happens
    BEFORE anything is read or saved."""
    store = _open_store()
    if not getattr(store, "_seed_checked", False):
        store._seed_checked = True
        seed_file = get_settings().sample_data_path / "past_incidents.json"
        if store._collection.count() == 0 and seed_file.exists():
            for inc in json.loads(seed_file.read_text()):
                _add({"id": inc["id"], "title": inc["title"], "severity": inc.get("severity", ""),
                      "root_cause": inc.get("root_cause", ""),
                      "resolution": inc.get("resolution", ""), "tags": inc.get("tags", "")})
    return store


def save_incident(id: str, title: str, severity: str = "", root_cause: str = "",
                  resolution: str = "", tags: str = "") -> None:
    """Store (or update) one incident. Re-saving the same id overwrites it."""
    get_store()                      # make sure the examples are loaded first
    _add({"id": id, "title": title, "severity": severity or "", "root_cause": root_cause or "",
          "resolution": resolution or "", "tags": tags or ""})


def search_incidents(query: str, k: int = 3, min_relevance: float = 0.05) -> list[dict]:
    """Find the most similar past incidents."""
    hits = get_store().similarity_search_with_score(query, k=k)
    out = []
    for d, dist in hits:
        rel = round(1 - dist, 3)
        if rel >= min_relevance:
            m = d.metadata
            out.append({"id": m["id"], "title": m["title"], "severity": m["severity"],
                        "root_cause": m["root_cause"], "resolution": m["resolution"],
                        "relevance": rel})
    return out


@tool
@safe_tool
def search_past_incidents(query: str, k: int = 3) -> str:
    """Search long-term memory for similar PAST incidents (title, root cause,
    resolution). Use symptoms or the incident title as the query. A high
    relevance (above ~0.3) means a genuinely similar earlier incident."""
    return jdump({"query": query, "similar_incidents": search_incidents(query, k)})


MEMORY_TOOLS = [search_past_incidents]
