# =====================================================================
# tools/runbook_search.py
# In simple words: "RAG over internal runbooks". The runbooks (markdown
# files in runbooks/) are cut into sections, stored in ChromaDB, and the
# agent can ask "how do I fix X?" and get back the best-matching sections.
# If you edit a runbook, the index rebuilds itself automatically.
# =====================================================================
from __future__ import annotations

import hashlib
import re

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool

from core.config import get_settings
from core.embeddings import get_embeddings, spec_slug
from tools.common import jdump, safe_tool

_stores: dict[tuple, Chroma] = {}


def load_runbook_chunks() -> list[Document]:
    """Read every runbook and split it by '## ' headings."""
    docs: list[Document] = []
    for path in sorted(get_settings().runbook_path.glob("*.md")):
        text = path.read_text()
        title = (re.search(r"^#\s+(.+)$", text, re.M) or [None, path.stem])[1]
        for part in re.split(r"(?m)^## ", text)[1:]:
            heading, _, body = part.partition("\n")
            docs.append(Document(
                page_content=f"{title} - {heading.strip()}\n{body.strip()}",
                metadata={"file": path.name, "title": title, "section": heading.strip()}))
    return docs


def get_runbook_store() -> Chroma:
    """Open (or build) the runbook vector index."""
    s = get_settings()
    docs = load_runbook_chunks()
    if not docs:
        raise FileNotFoundError(f"No runbooks found in {s.runbook_path}")
    # The collection name changes if the runbooks OR the embedding model change,
    # which triggers a clean rebuild.
    content_hash = hashlib.md5("".join(d.page_content for d in docs).encode()).hexdigest()[:8]
    name = f"runbooks_{spec_slug()}_{content_hash}"
    key = (str(s.chroma_path), name)
    if key not in _stores:
        store = Chroma(collection_name=name, embedding_function=get_embeddings(),
                       persist_directory=str(s.chroma_path),
                       collection_metadata={"hnsw:space": "cosine"})
        if store._collection.count() == 0:
            store.add_documents(docs, ids=[f"{d.metadata['file']}#{d.metadata['section']}" for d in docs])
        _stores[key] = store
    return _stores[key]


def search_runbook_chunks(query: str, k: int = 3) -> list[dict]:
    """Plain-Python search (used by tests and the tool)."""
    hits = get_runbook_store().similarity_search_with_score(query, k=k)
    return [{"runbook": d.metadata["title"], "file": d.metadata["file"],
             "section": d.metadata["section"],
             "relevance": round(1 - dist, 3),          # cosine distance -> similarity
             "text": d.page_content[:900]} for d, dist in hits]


@tool
@safe_tool
def search_runbooks(query: str, k: int = 3) -> str:
    """Search the internal runbooks (knowledge base) for how to diagnose or fix
    a problem. Describe the symptoms in plain words, for example
    'ECS tasks killed OOM exit code 137' or 'postgres connection slots exhausted'.
    Returns the best-matching runbook sections with a relevance score (0-1)."""
    return jdump({"query": query, "results": search_runbook_chunks(query, k)}, limit=7000)


RUNBOOK_TOOLS = [search_runbooks]
