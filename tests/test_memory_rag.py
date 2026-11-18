# ---------------------------------------------------------------------
# tests/test_memory_rag.py
# Checks: embeddings are stable, runbook search finds the RIGHT runbook for
# each symptom, long-term memory finds similar past incidents and can save
# new ones, short-term notebook records findings.
# ---------------------------------------------------------------------
import json

import pytest

from core.embeddings import HashingEmbeddings, get_embeddings
from memory import long_term, short_term
from tools import runbook_search
from tools.runbook_search import load_runbook_chunks, search_runbook_chunks, search_runbooks


@pytest.fixture(autouse=True)
def fresh_caches():
    runbook_search._stores.clear(); long_term._stores.clear(); short_term.clear_context()
    yield
    runbook_search._stores.clear(); long_term._stores.clear(); short_term.clear_context()


def test_embeddings_are_deterministic_and_normalised():
    e = HashingEmbeddings()
    a, b = e.embed_query("ecs oom killed"), e.embed_query("ecs oom killed")
    assert a == b and abs(sum(x * x for x in a) - 1.0) < 1e-6
    sim = lambda x, y: sum(p * q for p, q in zip(x, y))
    assert sim(a, e.embed_query("ECS tasks OOM killed")) > sim(a, e.embed_query("lambda throttling"))
    assert type(get_embeddings()).__name__ == "HashingEmbeddings"


def test_runbooks_are_chunked_by_section():
    chunks = load_runbook_chunks()
    assert len(chunks) > 20
    assert {"Symptoms", "Remediation"} <= {c.metadata["section"] for c in chunks}


@pytest.mark.parametrize("query,expected_file", [
    ("ECS tasks OOMKilled exit code 137 java.lang.OutOfMemoryError heap", "ecs-oom-killed.md"),
    ("postgres remaining connection slots reserved HikariPool connection not available", "rds-connection-exhaustion.md"),
    ("No space left on device disk full /var/log debug logging", "disk-full-ec2.md"),
    ("NullPointerException right after release 5xx spike roll back deployment", "bad-deployment-rollback.md"),
    ("AKS pod CrashLoopBackOff kubectl restart", "aks-pod-crashloop.md"),
])
def test_runbook_search_top_hit(query, expected_file):
    hits = search_runbook_chunks(query, k=3)
    assert hits[0]["file"] == expected_file, [h["file"] for h in hits]
    assert 0 < hits[0]["relevance"] <= 1


def test_runbook_tool_output_is_json():
    out = json.loads(search_runbooks.invoke({"query": "disk full", "k": 2}))
    assert len(out["results"]) == 2 and "text" in out["results"][0]


def test_runbook_index_is_reused_not_rebuilt():
    s1 = runbook_search.get_runbook_store(); n = s1._collection.count()
    runbook_search._stores.clear()                      # simulate a restart
    s2 = runbook_search.get_runbook_store()
    assert s2._collection.count() == n                  # loaded from disk, no duplicates


def test_past_incident_search_finds_similar():
    hits = long_term.search_incidents("ECS task OOM exit code 137 after cache change", k=2)
    assert hits[0]["id"] == "INC-2031" and hits[0]["relevance"] > 0.2
    hits = long_term.search_incidents("postgres connection slots exhausted by batch job", k=2)
    assert hits[0]["id"] == "INC-2044"


def test_save_incident_then_find_it_and_no_duplicates():
    long_term.save_incident("INC-9001", "redis eviction storm in session cache",
                            "P2", "maxmemory too low", "raised maxmemory", "redis cache eviction")
    long_term.save_incident("INC-9001", "redis eviction storm in session cache",
                            "P2", "maxmemory too low", "raised maxmemory", "redis cache eviction")
    hits = long_term.search_incidents("redis cache evicting keys", k=5)
    assert [h["id"] for h in hits].count("INC-9001") == 1 and hits[0]["id"] == "INC-9001"
    assert long_term.get_store()._collection.count() == 7   # 6 seed + 1 new (overwrite, not duplicate)


def test_short_term_notebook():
    short_term.add_finding("INC-1", "triage", "5xx rising")
    short_term.add_finding("INC-1", "diagnosis", "OOM after deploy")
    ctx = short_term.get_context("INC-1")
    assert [f["agent"] for f in ctx["findings"]] == ["triage", "diagnosis"]
    assert short_term.get_context("INC-2")["findings"] == []     # incidents are isolated
    out = json.loads(short_term.record_finding.invoke(
        {"incident_id": "INC-1", "agent": "x", "finding": "hello"}))
    assert out["saved"]["finding"] == "hello"
    short_term.clear_context("INC-1")
    assert short_term.get_context("INC-1")["findings"] == []
