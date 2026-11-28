"""Parse contract PDFs by clause and embed them (plus the playbook) into ChromaDB (Step 6)."""
import os
import re
import sqlite3
from functools import lru_cache
from pathlib import Path

import chromadb
import pdfplumber
from chromadb.utils import embedding_functions

from app.config import settings

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = settings.database_url.replace("sqlite:///", "")
CLAUSE_RE = re.compile(r"^(\d+\.\d+)\s+(.*)")


@lru_cache(maxsize=1)
def _client():
    return chromadb.PersistentClient(path=str(ROOT / settings.chroma_path))


@lru_cache(maxsize=1)
def _ef():
    """Build the embedding function from EMBEDDING_MODEL=provider:model."""
    provider, _, model = settings.embedding_model.partition(":")
    if provider == "st":
        return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model)
    if provider == "ollama":
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        return embedding_functions.OllamaEmbeddingFunction(url=host, model_name=model)
    raise RuntimeError(f"Unknown EMBEDDING_MODEL provider '{provider}'. Use st:<model> or ollama:<model>")


def get_collection(name: str):
    return _client().get_or_create_collection(name, embedding_function=_ef())


def split_clauses(text: str) -> list[tuple[str, str]]:
    """Split contract text into (clause_no, clause_text), one chunk per numbered clause."""
    clauses, current = [], None
    for line in text.splitlines():
        m = CLAUSE_RE.match(line.strip())
        if m:
            if current:
                clauses.append(current)
            current = [m.group(1), line.strip()]
        elif current and line.strip():
            current[1] += " " + line.strip()
    if current:
        clauses.append(current)
    return [(no, body) for no, body in clauses]


def ingest_contracts() -> int:
    _client().delete_collection("contracts") if "contracts" in [c.name for c in _client().list_collections()] else None
    col = get_collection("contracts")
    con = sqlite3.connect(ROOT / DB_PATH)
    rows = con.execute("SELECT id, customer_id, pdf_path FROM contracts").fetchall()
    ids, docs, metas = [], [], []
    for contract_id, customer_id, pdf_path in rows:
        with pdfplumber.open(ROOT / pdf_path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        for clause_no, body in split_clauses(text):
            ids.append(f"{contract_id}-{clause_no}")
            docs.append(body)
            metas.append({"customer_id": customer_id, "contract_id": contract_id, "clause_no": clause_no})
    col.add(ids=ids, documents=docs, metadatas=metas)
    return len(ids)


def ingest_playbook() -> int:
    if "playbook" in [c.name for c in _client().list_collections()]:
        _client().delete_collection("playbook")
    col = get_collection("playbook")
    lines = [l[2:].strip() for l in (ROOT / "data/playbook.md").read_text().splitlines() if l.startswith("- ")]
    col.add(ids=[f"rule-{i}" for i in range(len(lines))], documents=lines,
            metadatas=[{"source": "playbook"} for _ in lines])
    return len(lines)


if __name__ == "__main__":
    print(f"Ingested {ingest_contracts()} contract clauses, {ingest_playbook()} playbook rules")