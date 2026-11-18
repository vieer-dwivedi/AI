# ---------------------------------------------------------------------
# tests/test_orchestrator.py
# End-to-end: the full LangGraph workflow, for every scenario, plus the
# routing rules, failure handling and side effects (report, Slack outbox,
# long-term memory).
# ---------------------------------------------------------------------
import json

import pytest

import agents.orchestrator as orch
from agents.orchestrator import run_incident_async
from core.config import get_settings, reload_settings
from core.scenarios import incident_from_scenario, list_scenarios
from core.schemas import Incident
from memory.long_term import search_incidents
from memory.short_term import clear_context


@pytest.fixture(autouse=True)
def _fresh():
    clear_context()
    yield
    clear_context()


def pick(monkeypatch, name):
    monkeypatch.setenv("MOCK_SCENARIO", name)
    reload_settings()
    return incident_from_scenario(name)


@pytest.mark.parametrize("name", list_scenarios())
async def test_full_run(monkeypatch, name):
    inc = pick(monkeypatch, name)
    r = await run_incident_async(inc)
    assert r.status == "completed" and r.error is None
    assert r.severity in {"P1", "P2", "P3"} and r.root_cause
    assert len(r.proposed_actions) >= 1
    assert "## Root cause" in r.report_markdown and "## Proposed remediation" in r.report_markdown
    out = get_settings().output_path
    assert (out / "reports" / f"{inc.id}.md").exists()
    assert (out / "slack_outbox.jsonl").exists()                   # dry-run Slack (no webhook set)


async def test_slack_messages_cover_the_whole_run(monkeypatch):
    await run_incident_async(pick(monkeypatch, "ecs_oom"))
    lines = (get_settings().output_path / "slack_outbox.jsonl").read_text().splitlines()
    texts = [json.loads(x)["text"] for x in lines]
    assert len(texts) == 4                                         # start, severity, root cause, summary
    assert "Investigating" in texts[0] and "Root cause" in texts[2] and "Report ready" in texts[3]


async def test_incident_saved_to_long_term_memory(monkeypatch):
    inc = pick(monkeypatch, "ecs_oom")
    await run_incident_async(inc)
    hits = search_incidents("checkout-api out of memory OOMKilled", k=5)
    saved = next(h for h in hits if h["id"] == inc.id)
    assert "UNVERIFIED" in saved["resolution"]                     # proposals are never presented as proven fixes


async def test_past_incidents_are_found_and_reported(monkeypatch):
    r = await run_incident_async(pick(monkeypatch, "ecs_oom"))
    assert any(s["id"] == "INC-2031" for s in r.similar_incidents)   # the seeded OOM incident
    assert "INC-2031" in r.report_markdown


async def test_rerun_does_not_list_itself_as_similar(monkeypatch):
    inc = pick(monkeypatch, "ecs_oom")
    await run_incident_async(inc)
    r2 = await run_incident_async(inc)
    assert inc.id not in {s["id"] for s in r2.similar_incidents}


async def test_p4_noise_skips_diagnosis(monkeypatch):
    # Staging + no errors in logs -> triage says P4 -> graph must skip the expensive agents.
    monkeypatch.setenv("MOCK_SCENARIO", "ecs_oom"); reload_settings()
    called = []

    async def fake_triage(inc):
        called.append("triage")
        return {"severity": "P4", "response": "log only", "summary": "noise"}

    async def boom(*a, **k):
        called.append("diagnosis")
        raise AssertionError("diagnosis must not run for P4")

    monkeypatch.setattr(orch, "run_triage", fake_triage)
    monkeypatch.setattr(orch, "run_diagnosis", boom)
    r = await run_incident_async(incident_from_scenario("ecs_oom"))
    assert called == ["triage"] and r.severity == "P4" and r.status == "completed"
    assert r.proposed_actions == []


async def test_agent_crash_gives_failed_result_not_exception(monkeypatch):
    pick(monkeypatch, "ecs_oom")

    async def broken(inc):
        raise RuntimeError("401 invalid api key")

    monkeypatch.setattr(orch, "run_triage", broken)
    r = await run_incident_async(incident_from_scenario("ecs_oom"))
    assert r.status == "failed" and "invalid api key" in r.error
    assert "Problems during the run" in r.report_markdown


async def test_webhook_style_incident_without_labels_gets_log_group(monkeypatch):
    monkeypatch.setenv("MOCK_SCENARIO", ""); reload_settings()       # all scenarios in one account
    inc = Incident(id="INC-HOOK-1", source="alertmanager", title="High error rate",
                   service="payments-service", alarm_name="payments-service-5xx-high")
    r = await run_incident_async(inc)
    assert r.status == "completed"
    assert "NullPointerException" in r.report_markdown or "bad_deployment" in r.report_markdown


async def test_cloudwatch_alert_without_db_hint_still_finds_database_and_culprit(monkeypatch):
    # Real alarms name only the service. The agent must DISCOVER orders-db and reporting-job itself.
    monkeypatch.setenv("MOCK_SCENARIO", ""); reload_settings()
    inc = Incident(id="INC-CW-ORDERS", source="cloudwatch", title="Orders 5xx", service="orders-service",
                   alarm_name="orders-service-5xx-high")
    r = await run_incident_async(inc)
    assert r.status == "completed"
    assert "orders-db" in r.root_cause and "reporting-job" in r.root_cause
    targets = {a["target"] for a in r.proposed_actions}
    assert "rds:orders-db" in targets and "ecs:prod-cluster/reporting-job" in targets
